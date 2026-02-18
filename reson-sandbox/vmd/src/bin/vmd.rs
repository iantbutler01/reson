use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

use vmd_rs::app;
use vmd_rs::config::Config;

#[derive(Parser, Debug)]
#[command(name = "vmd", about = "Bracket VM daemon (Rust)")]
struct Args {
    #[arg(long)]
    listen: Option<String>,
    #[arg(long)]
    data_dir: Option<String>,
    #[arg(long)]
    qemu_bin: Option<String>,
    #[arg(long)]
    qemu_arm64_bin: Option<String>,
    #[arg(long)]
    qemu_img: Option<String>,
    #[arg(long)]
    docker_bin: Option<String>,
    #[arg(long)]
    log_level: Option<String>,
    /// Skip checking S3 for prebuilt VM images and always build locally
    #[arg(long)]
    force_local_build: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::default();

    if let Some(listen) = args.listen {
        cfg.listen_address = listen;
    }
    if let Some(dir) = args.data_dir {
        cfg.data_dir = dir;
    }
    if let Some(qemu) = args.qemu_bin {
        cfg.qemu_bin = qemu;
    }
    if let Some(qemu) = args.qemu_arm64_bin {
        cfg.qemu_arm64_bin = qemu;
    }
    if let Some(qemu_img) = args.qemu_img {
        cfg.qemu_img_bin = qemu_img;
    }
    if let Some(docker) = args.docker_bin {
        cfg.docker_bin = docker;
    }
    if let Some(level) = args.log_level {
        cfg.log_level = level;
    }
    if args.force_local_build {
        cfg.force_local_build = true;
    }

    init_tracing(&cfg.log_level)?;
    app::run_server(cfg).await
}

fn init_tracing(level: &str) -> Result<()> {
    let filter = EnvFilter::try_new(level)
        .or_else(|_| EnvFilter::try_new(format!("vmd_rs={level}")))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    Ok(())
}
