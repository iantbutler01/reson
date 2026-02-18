use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};

const ENV_PROXY_BIN: &str = "PROXY_BIN";
const DEFAULT_PROXY_BIN_DIR: &str = "../portproxy/bin";
const PORTPROXY_LINUX_AMD64: &str = "portproxy-linux-amd64";
const PORTPROXY_LINUX_ARM64: &str = "portproxy-linux-arm64";

pub fn binary(arch: &str) -> Result<Vec<u8>> {
    let filename = match arch {
        "amd64" => PORTPROXY_LINUX_AMD64,
        "arm64" => PORTPROXY_LINUX_ARM64,
        other => bail!("portproxy: unsupported arch {other}"),
    };

    let dir = env::var(ENV_PROXY_BIN).unwrap_or_else(|_| DEFAULT_PROXY_BIN_DIR.to_string());
    let path = PathBuf::from(dir).join(filename);
    fs::read(&path).with_context(|| format!("portproxy: read binary {}", path.to_string_lossy()))
}
