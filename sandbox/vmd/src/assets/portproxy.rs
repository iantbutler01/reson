// @dive-file: Locates and loads guest portproxy binaries used during VM bootstrap ISO assembly.
// @dive-rel: Called by vmd/src/bootstrap/mod.rs when embedding architecture-specific portproxy payloads.
// @dive-rel: Must remain robust to caller working-directory differences because vmd is launched from multiple entrypoints.
use std::env;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

const ENV_PROXY_BIN: &str = "PROXY_BIN";
const DEFAULT_PROXY_BIN_DIR: &str = "../portproxy/bin";
const PORTPROXY_LINUX_AMD64: &str = "portproxy-linux-amd64";
const PORTPROXY_LINUX_ARM64: &str = "portproxy-linux-arm64";
const GUEST_RUNTIME_FINGERPRINT_VERSION: &str = "bootstrap-v1";

pub fn binary(arch: &str) -> Result<Vec<u8>> {
    let filename = match arch {
        "amd64" => PORTPROXY_LINUX_AMD64,
        "arm64" => PORTPROXY_LINUX_ARM64,
        other => bail!("portproxy: unsupported arch {other}"),
    };

    if let Ok(dir) = env::var(ENV_PROXY_BIN) {
        let path = PathBuf::from(dir).join(filename);
        return fs::read(&path)
            .with_context(|| format!("portproxy: read binary {}", path.to_string_lossy()));
    }

    let mut attempted = Vec::new();
    for dir in default_candidate_dirs() {
        let path = dir.join(filename);
        attempted.push(path.clone());
        if let Ok(bytes) = fs::read(&path) {
            return Ok(bytes);
        }
    }

    let searched = attempted
        .iter()
        .map(|path| path.to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join(", ");
    bail!(
        "portproxy: read binary {filename} failed; searched: {searched}; set {ENV_PROXY_BIN} to override"
    );
}

pub fn guest_runtime_fingerprint(arch: &str) -> Result<String> {
    let bin = binary(arch)?;
    let digest = Sha256::digest(&bin);
    Ok(format!(
        "{GUEST_RUNTIME_FINGERPRINT_VERSION}:portproxy-sha256:{}",
        hex_lower(&digest)
    ))
}

pub fn binary_sha256_hex(arch: &str) -> Result<String> {
    let bin = binary(arch)?;
    Ok(hex_lower(&Sha256::digest(&bin)))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn default_candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = vec![
        PathBuf::from("portproxy/bin"),
        PathBuf::from(DEFAULT_PROXY_BIN_DIR),
    ];

    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            // @dive: Resolve from executable location so `target/debug/vmd` can find `<workspace>/portproxy/bin` regardless of cwd.
            dirs.push(exe_dir.join("../../portproxy/bin"));
            dirs.push(exe_dir.join("../../../portproxy/bin"));
            dirs.push(exe_dir.join("../portproxy/bin"));
        }
    }

    let mut deduped = Vec::new();
    for dir in dirs {
        if deduped.iter().any(|existing: &PathBuf| existing == &dir) {
            continue;
        }
        deduped.push(dir);
    }
    deduped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_candidate_dirs_include_legacy_and_workspace_paths() {
        let dirs = default_candidate_dirs();
        assert!(
            dirs.iter()
                .any(|path| path == &PathBuf::from(DEFAULT_PROXY_BIN_DIR))
        );
        assert!(
            dirs.iter()
                .any(|path| path == &PathBuf::from("portproxy/bin"))
        );
    }
}
