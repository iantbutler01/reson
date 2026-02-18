//! Live sandbox demo using reson-sandbox from reson-rust.
//!
//! Default mode is local managed daemon.
//! Set `RESON_SANDBOX_MODE=remote` to use an external daemon endpoint.

use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use reson_sandbox::{
    ExecEvent, ExecInput, ExecOptions, ForkOptions, Sandbox, SandboxConfig, SandboxError, Session,
    SessionOptions,
};

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after unix epoch")
        .as_secs()
}

async fn run_exec(session: &Session, command: &str) -> Result<i32, Box<dyn std::error::Error>> {
    const EXEC_RETRY_DELAY_SECS: u64 = 2;
    let exec_ready_timeout_secs = std::env::var("RESON_EXEC_READY_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(180);
    let deadline = Instant::now() + Duration::from_secs(exec_ready_timeout_secs);
    let mut attempt = 1usize;

    loop {
        match run_exec_once(session, command).await {
            Ok(exit_code) => return Ok(exit_code),
            Err(err) if is_retryable_exec_error(&err) && Instant::now() < deadline => {
                eprintln!("exec channel not ready yet (attempt {attempt}): {err}");
                attempt += 1;
                tokio::time::sleep(Duration::from_secs(EXEC_RETRY_DELAY_SECS)).await;
            }
            Err(err) => return Err(Box::new(err)),
        }
    }
}

fn is_retryable_exec_error(err: &SandboxError) -> bool {
    match err {
        SandboxError::Grpc(status) => {
            let msg = status.message().to_lowercase();
            msg.contains("transport error")
                || msg.contains("connection reset")
                || msg.contains("connection refused")
        }
        _ => false,
    }
}

async fn run_exec_once(session: &Session, command: &str) -> Result<i32, SandboxError> {
    let handle = session.exec(command, ExecOptions::default()).await?;
    let _ = handle.input.send(ExecInput::Eof).await;

    let mut events = handle.events;
    let mut exit_code = -1;
    while let Some(event) = events.next().await {
        match event? {
            ExecEvent::Stdout(bytes) => print!("{}", String::from_utf8_lossy(&bytes)),
            ExecEvent::Stderr(bytes) => eprint!("{}", String::from_utf8_lossy(&bytes)),
            ExecEvent::Timeout => eprintln!("[exec timeout event]"),
            ExecEvent::Exit(code) => {
                exit_code = code;
                break;
            }
        }
    }
    println!();
    Ok(exit_code)
}

async fn discard_with_retry(
    session: Session,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    const DISCARD_RETRY_DELAY_SECS: u64 = 1;
    let discard_timeout_secs = std::env::var("RESON_DISCARD_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(30);
    let deadline = Instant::now() + Duration::from_secs(discard_timeout_secs);
    let mut attempt = 1usize;

    loop {
        match session.clone().discard().await {
            Ok(()) => return Ok(()),
            Err(SandboxError::Grpc(status))
                if status.message().to_lowercase().contains("invalid vm state")
                    && Instant::now() < deadline =>
            {
                eprintln!(
                    "discard pending state transition for {label} (attempt {attempt}): {status}"
                );
                attempt += 1;
                tokio::time::sleep(Duration::from_secs(DISCARD_RETRY_DELAY_SECS)).await;
            }
            Err(err) => return Err(Box::new(err)),
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mode = std::env::var("RESON_SANDBOX_MODE").unwrap_or_else(|_| "local".to_string());
    let endpoint =
        std::env::var("RESON_SANDBOX_ENDPOINT").unwrap_or_else(|_| "http://127.0.0.1:18072".to_string());
    let image = std::env::var("RESON_SANDBOX_IMAGE")
        .unwrap_or_else(|_| "ghcr.io/bracketdevelopers/uv-builder:main".to_string());
    let attach_session_id = std::env::var("RESON_ATTACH_SESSION_ID").ok();

    let mut cfg = SandboxConfig::default();
    cfg.connect_timeout = Duration::from_secs(20);
    cfg.daemon_start_timeout = Duration::from_secs(90);

    let sandbox = if mode == "remote" {
        println!("mode=remote endpoint={endpoint}");
        Sandbox::connect(endpoint.clone(), cfg).await?
    } else {
        let vmd_bin = std::env::var("RESON_VMD_BIN")
            .unwrap_or_else(|_| "../reson-sandbox/target/debug/vmd".to_string());
        let proxy_bin_dir = std::env::var("RESON_PROXY_BIN_DIR")
            .unwrap_or_else(|_| "../reson-sandbox/portproxy/bin".to_string());
        let data_dir = std::env::var("RESON_SANDBOX_DATA_DIR")
            .unwrap_or_else(|_| "/tmp/reson-agent-demo-vmd".to_string());

        unsafe {
            std::env::set_var("PROXY_BIN", &proxy_bin_dir);
        }

        cfg.endpoint = endpoint.clone();
        cfg.auto_spawn = true;
        cfg.daemon_listen = endpoint
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .to_string();
        cfg.daemon_bin = Some(PathBuf::from(vmd_bin));
        cfg.daemon_data_dir = Some(PathBuf::from(data_dir));

        println!("mode=local endpoint={} daemon_bin={:?}", cfg.endpoint, cfg.daemon_bin);
        Sandbox::new(cfg).await?
    };

    let suffix = unix_secs();
    let (parent_session_id, parent, attached_parent) = if let Some(session_id) = attach_session_id {
        println!("attaching existing session {session_id}");
        let attached = sandbox.attach_session(&session_id).await?;
        (session_id, attached, true)
    } else {
        let session_id = format!("demo-parent-{suffix}");
        println!("creating parent session {session_id}");
        let created = sandbox
            .session(SessionOptions {
                session_id: Some(session_id.clone()),
                name: Some(format!("reson-agent-demo-{suffix}")),
                image: Some(image),
                auto_start: true,
                ..SessionOptions::default()
            })
            .await?;
        (session_id, created, false)
    };
    println!("parent session_id={parent_session_id}");

    println!("parent vm_id={}", parent.vm_id());

    let exit = run_exec(
        &parent,
        "echo '[parent] hello from reson-rust'; uname -a; cat /etc/os-release | head -n 2",
    )
    .await?;
    if exit != 0 {
        return Err(format!("parent exec failed with exit code {exit}").into());
    }

    parent
        .write_file("/tmp/demo_note.txt", b"hello-from-parent\n".to_vec())
        .await?;
    let note = parent.read_file("/tmp/demo_note.txt").await?;
    println!("parent file readback={}", String::from_utf8_lossy(&note).trim());

    println!("forking parent session");
    let fork = parent.fork(ForkOptions::default()).await?;
    println!(
        "forked child_session_id={} fork_id={}",
        fork.child_session_id, fork.fork_id
    );

    let exit = run_exec(
        &fork.child,
        "echo '[child] branch write'; echo child-branch > /tmp/branch.txt; cat /tmp/branch.txt",
    )
    .await?;
    if exit != 0 {
        return Err(format!("child exec failed with exit code {exit}").into());
    }

    let exit = run_exec(
        &parent,
        "if [ -f /tmp/branch.txt ]; then echo parent_has_branch_file; else echo parent_isolated; fi",
    )
    .await?;
    if exit != 0 {
        return Err(format!("parent isolation check failed with exit code {exit}").into());
    }

    let sessions = sandbox.list_sessions().await?;
    println!("listed {} durable session(s)", sessions.len());

    println!("cleaning up child session");
    discard_with_retry(fork.child, "child").await?;
    if attached_parent {
        println!("leaving attached parent session intact");
    } else {
        println!("cleaning up parent session");
        discard_with_retry(parent, "parent").await?;
    }

    println!("sandbox agent demo complete");
    Ok(())
}
