// @dive-file: Portproxy daemon entrypoint wiring gRPC services, TCP forwarding, and process supervision.
// @dive-rel: Composes child_tracker, daemon, services, and port_forward modules into one runtime.
// @dive-rel: Runs in guest VMs and exposes shell/daemon execution channels used by vmd control paths.

#![allow(clippy::collapsible_if)]
#![allow(clippy::default_constructed_unit_structs)]

mod child_tracker;
mod cli;
mod daemon;
mod port_forward;
mod services;

pub mod pb {
    pub mod bracket {
        pub mod portproxy {
            pub mod v1 {
                tonic::include_proto!("bracket.portproxy.v1");
            }
        }
    }

    pub mod google {
        pub mod protobuf {
            tonic::include_proto!("google.protobuf");
        }
    }
}

use std::net::SocketAddr;
use std::process;
use std::time::Duration;

use anyhow::Context;
use child_tracker::ChildTracker;
use clap::Parser;
use cli::Args;
use daemon::DaemonRegistry;
use nix::sys::signal::{Signal as UnixSignal, kill};
use nix::sys::wait::{WaitPidFlag, WaitStatus, waitpid};
use nix::unistd::Pid;
use services::{DaemonManagerService, PortProxyService, ShellExecService};
use tokio::signal::unix::{SignalKind, signal};
use tokio::time::sleep;
use tonic::transport::Server;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use crate::pb::bracket::portproxy::v1::daemon_manager_server::DaemonManagerServer;
use crate::pb::bracket::portproxy::v1::port_proxy_server::PortProxyServer;
use crate::pb::bracket::portproxy::v1::shell_exec_server::ShellExecServer;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let args = Args::parse();
    if let Err(err) = args.validate() {
        eprintln!("{err}");
        process::exit(1);
    }

    if args.server {
        run_server_mode(args).await
    } else {
        run_client_mode(args).await
    }
}

fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
}

async fn run_server_mode(args: Args) -> anyhow::Result<()> {
    let tracker = ChildTracker::new();
    let daemon_registry = DaemonRegistry::new(tracker.clone());

    tokio::spawn(reap_zombies(tracker.clone()));
    tokio::spawn({
        let tracker = tracker.clone();
        async move {
            if let Err(err) = forward_signals(tracker).await {
                warn!("forward signal task failed: {}", err);
            }
        }
    });

    let rpc_addr: SocketAddr = format!("0.0.0.0:{}", args.rpc_port).parse()?;

    let grpc = serve_grpc(rpc_addr, tracker.clone(), daemon_registry.clone());
    let port_proxy = port_forward::run_server(&args.server_addr);

    tokio::try_join!(grpc, port_proxy)?;
    Ok(())
}

async fn run_client_mode(args: Args) -> anyhow::Result<()> {
    let listen = args.listen_port.expect("validated");
    let forward = args.forward_port.expect("validated");
    port_forward::run_client(listen, forward, &args.server_addr).await
}

async fn serve_grpc(
    addr: SocketAddr,
    tracker: ChildTracker,
    daemon_registry: DaemonRegistry,
) -> anyhow::Result<()> {
    let shell_exec = ShellExecService::new(tracker.clone());
    let port_proxy = PortProxyService::new(tracker.clone());
    let daemon_manager = DaemonManagerService::new(daemon_registry);

    let (health_reporter, health_service) = tonic_health::server::health_reporter();

    health_reporter
        .set_serving::<ShellExecServer<ShellExecService>>()
        .await;
    health_reporter
        .set_serving::<PortProxyServer<PortProxyService>>()
        .await;
    health_reporter
        .set_serving::<DaemonManagerServer<DaemonManagerService>>()
        .await;

    info!("Starting gRPC server on {}", addr);

    Server::builder()
        .add_service(health_service)
        .add_service(ShellExecServer::new(shell_exec))
        .add_service(PortProxyServer::new(port_proxy))
        .add_service(DaemonManagerServer::new(daemon_manager))
        .serve(addr)
        .await
        .context("gRPC server failed")
}

async fn reap_zombies(tracker: ChildTracker) {
    use nix::errno::Errno;

    loop {
        match waitpid(Pid::from_raw(-1), Some(WaitPidFlag::WNOHANG)) {
            Ok(WaitStatus::StillAlive) => sleep(Duration::from_secs(5)).await,
            Ok(WaitStatus::Exited(pid, status)) => {
                info!("Reaped pid {} with status {}", pid, status);
                tracker.unregister(pid.as_raw()).await;
            }
            Ok(WaitStatus::Signaled(pid, signal, _)) => {
                info!("Reaped pid {} terminated by {:?}", pid, signal);
                tracker.unregister(pid.as_raw()).await;
            }
            Ok(_) => sleep(Duration::from_secs(1)).await,
            Err(Errno::ECHILD) => sleep(Duration::from_secs(5)).await,
            Err(err) => {
                warn!("waitpid failed: {}", err);
                sleep(Duration::from_secs(1)).await;
            }
        }
    }
}

async fn forward_signals(tracker: ChildTracker) -> anyhow::Result<()> {
    let mut term = signal(SignalKind::terminate())?;
    let mut int = signal(SignalKind::interrupt())?;

    let received = tokio::select! {
        _ = term.recv() => UnixSignal::SIGTERM,
        _ = int.recv() => UnixSignal::SIGINT,
    };

    info!(
        "Received signal {:?}, forwarding to child processes",
        received
    );

    let pids = tracker.snapshot().await;
    for pid in pids {
        if pid <= 0 {
            continue;
        }
        if let Err(err) = kill(Pid::from_raw(pid), received) {
            warn!("failed to send {:?} to pid {}: {}", received, pid, err);
        }
    }

    sleep(Duration::from_secs(2)).await;
    info!("Shutting down after signal");
    process::exit(0);
}
