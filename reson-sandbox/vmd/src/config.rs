use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub const BASE_IMAGES_DIR_NAME: &str = "base_images";

#[derive(Clone, Debug)]
pub struct Config {
    pub listen_address: String,
    pub data_dir: String,
    pub qemu_bin: String,
    pub qemu_arm64_bin: String,
    pub qemu_img_bin: String,
    pub docker_bin: String,
    pub log_level: String,
    pub force_local_build: bool,
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".bracket")
            .join("vms");

        Self {
            listen_address: "127.0.0.1:8052".to_string(),
            data_dir: data_dir.to_string_lossy().to_string(),
            qemu_bin: "qemu-system-x86_64".to_string(),
            qemu_arm64_bin: "qemu-system-aarch64".to_string(),
            qemu_img_bin: "qemu-img".to_string(),
            docker_bin: "docker".to_string(),
            log_level: "info".to_string(),
            force_local_build: false,
        }
    }
}

impl Config {
    pub fn base_images_dir(&self) -> PathBuf {
        Path::new(&self.data_dir).join(BASE_IMAGES_DIR_NAME)
    }

    pub fn normalize(&mut self) -> Result<()> {
        if self.listen_address.trim().is_empty() {
            bail!("listen address must be provided");
        }
        if self.data_dir.trim().is_empty() {
            bail!("data dir must be provided");
        }

        let data_dir = expand_home(&self.data_dir)?;
        fs::create_dir_all(&data_dir)
            .with_context(|| format!("create data dir {}", data_dir.to_string_lossy()))?;
        let base_dir = data_dir.join(BASE_IMAGES_DIR_NAME);
        fs::create_dir_all(&base_dir)
            .with_context(|| format!("create base image dir {}", base_dir.to_string_lossy()))?;
        self.data_dir = canonical_string(&data_dir)?;

        self.qemu_bin = resolve_binary(&self.qemu_bin, "qemu-system-x86_64")?;
        self.qemu_arm64_bin = resolve_binary(&self.qemu_arm64_bin, "qemu-system-aarch64")?;
        self.qemu_img_bin = resolve_binary(&self.qemu_img_bin, "qemu-img")?;
        self.docker_bin = resolve_binary(&self.docker_bin, "docker")?;

        Ok(())
    }
}

fn expand_home(path: &str) -> Result<PathBuf> {
    if path.is_empty() {
        bail!("empty path");
    }
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = dirs::home_dir().context("resolve home directory")?;
        return Ok(home.join(stripped));
    }
    if path == "~" {
        let home = dirs::home_dir().context("resolve home directory")?;
        return Ok(home);
    }
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        Ok(candidate)
    } else {
        Ok(env::current_dir()?.join(candidate))
    }
}

fn resolve_binary(value: &str, fallback: &str) -> Result<String> {
    let candidate = {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            fallback
        } else {
            trimmed
        }
    };

    let path =
        lookup_executable(candidate).with_context(|| format!("locate executable {candidate}"))?;
    canonical_string(&path)
}

fn lookup_executable(candidate: &str) -> Result<PathBuf> {
    let path = PathBuf::from(candidate);
    if path.is_absolute() {
        fs::metadata(&path).with_context(|| format!("stat {}", path.to_string_lossy()))?;
        return Ok(path);
    }

    let paths = env::var_os("PATH").context("PATH environment variable unset")?;
    for dir in env::split_paths(&paths) {
        let full = dir.join(candidate);
        if full.is_file() {
            return Ok(full);
        }
    }
    bail!("{} not found in PATH", candidate);
}

fn canonical_string(path: &Path) -> Result<String> {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Ok(canonical.to_string_lossy().to_string())
}
