use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use vmd_rs::fuse::{mount_remote_vfs_fuse, unmount_fuse};

#[derive(Debug, Parser)]
#[command(about = "Mount a Chevalier remote VFS endpoint as a foreground FUSE filesystem")]
struct Args {
    #[arg(long)]
    endpoint: String,
    #[arg(long, default_value = "")]
    scope: String,
    #[arg(long, default_value = "chevalier-vfs")]
    tag: String,
    #[arg(long)]
    token: Option<String>,
    #[arg(long, default_value = "CHEVALIER_SANDBOX_VFS_INTERNAL_SERVICE_TOKEN")]
    token_env: String,
    #[arg(long)]
    read_only: bool,
    #[arg(long)]
    mountpoint: Option<PathBuf>,
    #[arg(value_name = "MOUNTPOINT")]
    positional_mountpoint: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mountpoint = args
        .mountpoint
        .or(args.positional_mountpoint)
        .ok_or_else(|| anyhow!("mountpoint is required"))?;
    let token = args
        .token
        .or_else(|| std::env::var(args.token_env.as_str()).ok())
        .ok_or_else(|| anyhow!("missing VFS token; pass --token or set {}", args.token_env))?;
    let read_only =
        args.read_only || env_truthy("OC_MOUNT_READONLY") || env_truthy("CHEVALIER_VFS_READ_ONLY");

    let handle = mount_remote_vfs_fuse(
        args.endpoint.as_str(),
        token.as_str(),
        args.scope.as_str(),
        args.tag.as_str(),
        &mountpoint,
        read_only,
    )
    .await
    .with_context(|| format!("mount remote VFS at {}", mountpoint.display()))?;

    wait_for_shutdown_signal().await?;
    unmount_fuse(&handle).await
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

#[cfg(unix)]
async fn wait_for_shutdown_signal() -> Result<()> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .context("install SIGTERM handler")?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result.context("wait for ctrl-c")?,
        _ = terminate.recv() => {},
    }
    Ok(())
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() -> Result<()> {
    tokio::signal::ctrl_c().await.context("wait for ctrl-c")
}
