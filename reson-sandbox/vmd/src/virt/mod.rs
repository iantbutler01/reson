#[cfg(target_os = "macos")]
use std::ffi::CString;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::process::Command;
use tokio::task;
use tokio::time::sleep;
use tracing::{debug, trace};

#[derive(Clone, Debug)]
pub struct MonitorHandle {
    path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Platform {
    pub os: String,
    pub arch: String,
}

#[derive(Clone, Debug)]
pub struct D2VmOptions {
    pub image: String,
    pub output: String,
    pub disk_gb: i32,
    pub pull: bool,
    pub platform: Option<String>,
    pub include_bootstrap: bool,
}

#[derive(Clone, Debug, Default)]
pub struct StatusInfo {
    pub running: bool,
    pub status: String,
}

const D2VM_IMAGE: &str = "ghcr.io/bracketdevelopers/d2vm:latest";
const D2VM_CONTAINER_DIR: &str = "/workspace";

pub async fn run_d2vm(docker_bin: &str, host_dir: &Path, opts: D2VmOptions) -> Result<()> {
    if opts.image.trim().is_empty() {
        bail!("d2vm image reference is required");
    }
    let output_name = if opts.output.trim().is_empty() {
        "disk.qcow2".to_string()
    } else {
        opts.output.clone()
    };
    let disk_gb = if opts.disk_gb <= 0 { 10 } else { opts.disk_gb };
    let host_dir = host_dir
        .canonicalize()
        .with_context(|| format!("canonicalize {}", host_dir.display()))?;
    debug!(
        image = %opts.image,
        output = %output_name,
        disk_gb,
        platform = ?opts.platform,
        pull = opts.pull,
        include_bootstrap = opts.include_bootstrap,
        workdir = %host_dir.display(),
        "running d2vm conversion"
    );

    let build_cmd = |include_bootstrap: bool| {
        let mut cmd = Command::new(docker_bin);
        cmd.arg("run")
            .arg("--rm")
            .arg("--privileged")
            .arg("-v")
            .arg("/var/run/docker.sock:/var/run/docker.sock")
            .arg("-v")
            .arg(format!("{}:{}", host_dir.display(), D2VM_CONTAINER_DIR))
            .arg("-w")
            .arg(D2VM_CONTAINER_DIR)
            .arg(D2VM_IMAGE)
            .arg("--verbose")
            .arg("convert")
            .arg(&opts.image)
            .arg("--output")
            .arg(&output_name)
            .arg("--size")
            .arg(format!("{disk_gb}G"))
            .arg("--password")
            .arg("root");

        if opts.pull {
            cmd.arg("--pull");
        }
        if let Some(platform) = &opts.platform {
            if !platform.trim().is_empty() {
                cmd.arg("--platform").arg(platform);
            }
        }
        if include_bootstrap {
            cmd.arg("--include-bootstrap");
        }
        cmd
    };

    let mut cmd = build_cmd(opts.include_bootstrap);
    trace!(command = ?cmd, "spawning docker d2vm");
    let output = cmd.output().await.context("spawn docker d2vm")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        if opts.include_bootstrap && stderr.contains("Unknown flag: --include-bootstrap") {
            debug!("d2vm image does not support --include-bootstrap; retrying without it");
            let retry_output = build_cmd(false)
                .output()
                .await
                .context("spawn docker d2vm retry without bootstrap")?;
            if !retry_output.status.success() {
                bail!(
                    "d2vm convert failed: {}",
                    String::from_utf8_lossy(&retry_output.stderr)
                );
            }
        } else {
            bail!("d2vm convert failed: {stderr}");
        }
    }
    debug!(output = %output_name, "d2vm conversion complete");
    Ok(())
}

pub async fn create_overlay(
    qemu_img_bin: &str,
    base_path: &Path,
    overlay_path: &Path,
    size_gb: i32,
) -> Result<()> {
    if size_gb <= 0 {
        bail!("invalid overlay size: {size_gb}GB");
    }
    let size_arg = format!("{size_gb}G");
    let output = Command::new(qemu_img_bin)
        .arg("create")
        .arg("-f")
        .arg("qcow2")
        .arg("-F")
        .arg("qcow2")
        .arg("-b")
        .arg(base_path)
        .arg(overlay_path)
        .arg(size_arg)
        .output()
        .await
        .with_context(|| "spawn qemu-img create")?;
    if !output.status.success() {
        bail!(
            "qemu-img overlay create failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

pub async fn create_snapshot_offline(
    qemu_img_bin: &str,
    disk_path: &Path,
    name: &str,
) -> Result<()> {
    let args = [
        OsStr::new("snapshot"),
        OsStr::new("-c"),
        OsStr::new(name),
        disk_path.as_os_str(),
    ];
    run_qemu_img(qemu_img_bin, &args, "snapshot create").await
}

pub async fn delete_snapshot_offline(
    qemu_img_bin: &str,
    disk_path: &Path,
    name: &str,
) -> Result<()> {
    let args = [
        OsStr::new("snapshot"),
        OsStr::new("-d"),
        OsStr::new(name),
        disk_path.as_os_str(),
    ];
    run_qemu_img(qemu_img_bin, &args, "snapshot delete").await
}

pub async fn revert_snapshot_offline(
    qemu_img_bin: &str,
    disk_path: &Path,
    name: &str,
) -> Result<()> {
    let args = [
        OsStr::new("snapshot"),
        OsStr::new("-a"),
        OsStr::new(name),
        disk_path.as_os_str(),
    ];
    run_qemu_img(qemu_img_bin, &args, "snapshot apply").await
}

async fn run_qemu_img(qemu_img_bin: &str, args: &[&OsStr], context_label: &str) -> Result<()> {
    let mut cmd = Command::new(qemu_img_bin);
    for arg in args {
        cmd.arg(arg);
    }
    let output = cmd
        .output()
        .await
        .with_context(|| format!("spawn qemu-img {context_label}"))?;
    if !output.status.success() {
        bail!(
            "qemu-img {context_label} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

pub async fn wait_for_monitor(path: &Path, timeout: Duration) -> Result<MonitorHandle> {
    let deadline = Instant::now() + timeout;
    loop {
        match establish_connection(path).await {
            Ok(conn) => {
                drop(conn);
                return Ok(MonitorHandle {
                    path: path.to_path_buf(),
                });
            }
            Err(err) => {
                if Instant::now() >= deadline {
                    return Err(err);
                }
                sleep(Duration::from_millis(200)).await;
            }
        }
    }
}

pub async fn wait_for_running(monitor: &MonitorHandle, timeout: Duration) -> Result<()> {
    let deadline = Instant::now() + timeout;
    loop {
        let status = monitor.query_status().await?;
        if status.running {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("timeout waiting for VM to report running state");
        }
        sleep(Duration::from_millis(500)).await;
    }
}

pub async fn system_powerdown(monitor: &MonitorHandle) -> Result<()> {
    monitor.execute("system_powerdown", None).await.map(|_| ())
}

pub async fn quit(monitor: &MonitorHandle) -> Result<()> {
    monitor.execute("quit", None).await.map(|_| ())
}

pub async fn stop(monitor: &MonitorHandle) -> Result<()> {
    monitor.execute("stop", None).await.map(|_| ())
}

pub async fn cont(monitor: &MonitorHandle) -> Result<()> {
    monitor.execute("cont", None).await.map(|_| ())
}

pub async fn save_vm(monitor: &MonitorHandle, name: &str) -> Result<()> {
    let args = json!({ "command-line": format!("savevm {name}") });
    monitor
        .execute("human-monitor-command", Some(args))
        .await
        .map(|_| ())
}

pub async fn load_vm(monitor: &MonitorHandle, name: &str) -> Result<()> {
    let args = json!({ "command-line": format!("loadvm {name}") });
    monitor
        .execute("human-monitor-command", Some(args))
        .await
        .map(|_| ())
}

pub async fn delete_snapshot(monitor: &MonitorHandle, name: &str) -> Result<()> {
    let args = json!({ "command-line": format!("delvm {name}") });
    monitor
        .execute("human-monitor-command", Some(args))
        .await
        .map(|_| ())
}

pub async fn inspect_image_platforms(docker_bin: &str, reference: &str) -> Result<Vec<Platform>> {
    let output = Command::new(docker_bin)
        .arg("manifest")
        .arg("inspect")
        .arg(reference)
        .output()
        .await
        .with_context(|| "docker manifest inspect")?;
    if !output.status.success() {
        bail!(
            "docker manifest inspect failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let value: Value =
        serde_json::from_slice(&output.stdout).context("parse docker manifest inspect output")?;
    let mut platforms = Vec::new();
    if let Some(manifests) = value.get("manifests").and_then(|v| v.as_array()) {
        for manifest in manifests {
            if let Some(platform) = manifest.get("platform") {
                if let (Some(os), Some(arch)) = (platform.get("os"), platform.get("architecture")) {
                    if let (Some(os), Some(arch)) = (os.as_str(), arch.as_str()) {
                        platforms.push(Platform {
                            os: os.to_string(),
                            arch: arch.to_string(),
                        });
                    }
                }
            }
        }
    }
    if platforms.is_empty() {
        if let (Some(os), Some(arch)) = (
            value.get("os").and_then(|v| v.as_str()),
            value.get("architecture").and_then(|v| v.as_str()),
        ) {
            platforms.push(Platform {
                os: os.to_string(),
                arch: arch.to_string(),
            });
        }
    }
    if platforms.is_empty() {
        bail!("docker manifest inspect did not return platform information");
    }
    Ok(platforms)
}

pub async fn copy_file(src: &Path, dst: &Path) -> Result<()> {
    let src = src.to_owned();
    let dst = dst.to_owned();
    task::spawn_blocking(move || {
        std::fs::copy(&src, &dst)
            .with_context(|| format!("copy {} -> {}", src.display(), dst.display()))?;
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    Ok(())
}

pub async fn clone_file_cow(src: &Path, dst: &Path) -> Result<()> {
    let src = src.to_owned();
    let dst = dst.to_owned();
    task::spawn_blocking(move || clone_file_cow_sync(src.as_path(), dst.as_path())).await??;
    Ok(())
}

#[cfg(target_os = "linux")]
fn clone_file_cow_sync(src: &Path, dst: &Path) -> Result<()> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;

    const FICLONE: libc::c_ulong = 0x4004_9409;

    let src_file = OpenOptions::new()
        .read(true)
        .open(src)
        .with_context(|| format!("open clone source {}", src.display()))?;
    let dst_file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dst)
        .with_context(|| format!("open clone destination {}", dst.display()))?;

    // Enforce CoW clone semantics; never fall back to byte-for-byte copy.
    let rc = unsafe { libc::ioctl(dst_file.as_raw_fd(), FICLONE, src_file.as_raw_fd()) };
    if rc != 0 {
        bail!(
            "reflink clone failed {} -> {}: {}",
            src.display(),
            dst.display(),
            std::io::Error::last_os_error()
        );
    }

    Ok(())
}

#[cfg(target_os = "macos")]
fn clone_file_cow_sync(src: &Path, dst: &Path) -> Result<()> {
    use std::os::unix::ffi::OsStrExt;

    let src_c = CString::new(src.as_os_str().as_bytes())
        .with_context(|| format!("clone source path contains NUL: {}", src.display()))?;
    let dst_c = CString::new(dst.as_os_str().as_bytes())
        .with_context(|| format!("clone destination path contains NUL: {}", dst.display()))?;

    // Enforce CoW clone semantics; never fall back to byte-for-byte copy.
    let rc = unsafe { libc::clonefile(src_c.as_ptr(), dst_c.as_ptr(), 0) };
    if rc != 0 {
        bail!(
            "clonefile failed {} -> {}: {}",
            src.display(),
            dst.display(),
            std::io::Error::last_os_error()
        );
    }

    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn clone_file_cow_sync(src: &Path, dst: &Path) -> Result<()> {
    let _ = (src, dst);
    bail!("CoW file cloning is not supported on this platform")
}

impl MonitorHandle {
    pub async fn query_status(&self) -> Result<StatusInfo> {
        let mut conn = establish_connection(&self.path).await?;
        let value = execute_raw(&mut conn, "query-status", None).await?;
        #[derive(Deserialize)]
        struct Response {
            running: bool,
            status: String,
        }
        let resp: Response = serde_json::from_value(value)?;
        Ok(StatusInfo {
            running: resp.running,
            status: resp.status,
        })
    }

    async fn execute(&self, command: &str, args: Option<Value>) -> Result<Value> {
        let mut conn = establish_connection(&self.path).await?;
        execute_raw(&mut conn, command, args).await
    }
}

struct QmpConnection {
    reader: BufReader<tokio::net::unix::OwnedReadHalf>,
    writer: tokio::net::unix::OwnedWriteHalf,
}

async fn establish_connection(path: &Path) -> Result<QmpConnection> {
    let stream = UnixStream::connect(path)
        .await
        .with_context(|| format!("connect qmp {}", path.display()))?;
    let (read_half, mut write_half) = stream.into_split();
    let mut reader = BufReader::new(read_half);

    let greeting = read_message(&mut reader).await?;
    if !greeting.get("QMP").is_some() {
        bail!("unexpected QMP greeting: {greeting}");
    }
    write_command(&mut write_half, &json!({ "execute": "qmp_capabilities" })).await?;
    let response = read_message(&mut reader).await?;
    if response.get("return").is_none() {
        bail!("qmp_capabilities handshake failed: {response}");
    }

    Ok(QmpConnection {
        reader,
        writer: write_half,
    })
}

async fn execute_raw(
    conn: &mut QmpConnection,
    command: &str,
    args: Option<Value>,
) -> Result<Value> {
    let mut payload = json!({ "execute": command });
    if let Some(arguments) = args {
        payload
            .as_object_mut()
            .unwrap()
            .insert("arguments".to_string(), arguments);
    }
    write_command(&mut conn.writer, &payload).await?;
    let response = read_message(&mut conn.reader).await?;
    if let Some(err) = response.get("error") {
        bail!("qmp error: {err}");
    }
    Ok(response.get("return").cloned().unwrap_or(Value::Null))
}

async fn write_command(writer: &mut tokio::net::unix::OwnedWriteHalf, value: &Value) -> Result<()> {
    let mut data = serde_json::to_vec(value)?;
    data.push(b'\n');
    writer.write_all(&data).await?;
    writer.flush().await?;
    Ok(())
}

async fn read_message(reader: &mut BufReader<tokio::net::unix::OwnedReadHalf>) -> Result<Value> {
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line).await?;
        if bytes == 0 {
            bail!("qmp connection closed");
        }
        match serde_json::from_str::<Value>(&line) {
            Ok(value) => {
                if value.get("event").is_some() {
                    continue;
                }
                return Ok(value);
            }
            Err(err) => {
                // lines can be partial; continue reading
                if err.is_data() {
                    continue;
                }
                return Err(err.into());
            }
        }
    }
}
