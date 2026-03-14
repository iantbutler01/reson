// @dive-file: Tracks named long-lived guest-side daemon processes used for attachable command execution streams.
// @dive-rel: Consumed by portproxy/src/services.rs DaemonManagerService for exec/attach semantics.
// @dive-rel: Enables distributed stream producer reattachment by name across control-plane rebind attempts.
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use portable_pty::CommandBuilder;
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock, broadcast, watch};
use tracing::{error, info, warn};

use crate::child_tracker::ChildTracker;
use crate::pb::bracket::portproxy::v1::{ExecDaemonRequest, ExecDaemonResponse};

const CHANNEL_CAPACITY: usize = 100;
const BACKLOG_MAX_FRAMES: usize = 512;

#[derive(Error, Debug)]
pub enum DaemonError {
    #[error("daemon already running")]
    AlreadyRunning,
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("spawn error: {0}")]
    Spawn(#[from] std::io::Error),
}

#[derive(Clone)]
pub struct DaemonRegistry {
    daemons: Arc<RwLock<HashMap<String, Arc<DaemonEntry>>>>,
    tracker: ChildTracker,
}

impl DaemonRegistry {
    pub fn new(tracker: ChildTracker) -> Self {
        Self {
            daemons: Arc::new(RwLock::new(HashMap::new())),
            tracker,
        }
    }

    pub async fn exec_daemon(
        &self,
        req: ExecDaemonRequest,
    ) -> Result<ExecDaemonResponse, DaemonError> {
        if req.name.is_empty() {
            return Err(DaemonError::InvalidRequest("name is required".into()));
        }
        if req.args.is_empty() {
            return Err(DaemonError::InvalidRequest("args must not be empty".into()));
        }

        {
            let daemons = self.daemons.read().await;
            if daemons.contains_key(&req.name) {
                return Ok(ExecDaemonResponse { is_new: false });
            }
        }

        let mut command = Command::new(&req.args[0]);
        if req.args.len() > 1 {
            command.args(&req.args[1..]);
        }
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        command.env_clear();
        for (key, value) in &req.env {
            command.env(key, value);
        }

        let mut child = command.spawn()?;

        let pid = match child.id() {
            Some(id) => id as i32,
            None => {
                warn!("daemon {}: spawned process without pid", req.name);
                -1
            }
        };

        if pid > 0 {
            self.tracker.register(pid).await;
        }

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| DaemonError::InvalidRequest("failed to capture stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| DaemonError::InvalidRequest("failed to capture stdout".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| DaemonError::InvalidRequest("failed to capture stderr".into()))?;

        let (stdout_tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        let (stderr_tx, _) = broadcast::channel(CHANNEL_CAPACITY);
        let (exit_tx, _) = watch::channel(None::<i32>);
        let output_backlog = Arc::new(Mutex::new(OutputBacklog::default()));

        let entry = Arc::new(DaemonEntry {
            _name: req.name.clone(),
            stdin: Mutex::new(stdin),
            stdout_tx: stdout_tx.clone(),
            stderr_tx: stderr_tx.clone(),
            output_backlog: output_backlog.clone(),
            attach_lock: Arc::new(Mutex::new(())),
            exit_tx: exit_tx.clone(),
        });

        {
            let mut daemons = self.daemons.write().await;
            if daemons.contains_key(&req.name) {
                return Err(DaemonError::AlreadyRunning);
            }
            daemons.insert(req.name.clone(), entry.clone());
        }

        spawn_reader(
            stdout,
            stdout_tx,
            output_backlog.clone(),
            OutputKind::Stdout,
            format!("stdout({})", req.name),
        );
        spawn_reader(
            stderr,
            stderr_tx,
            output_backlog,
            OutputKind::Stderr,
            format!("stderr({})", req.name),
        );

        let daemons = self.daemons.clone();
        let tracker = self.tracker.clone();
        tokio::spawn(async move {
            let status = match child.wait().await {
                Ok(status) => status,
                Err(err) => {
                    error!("daemon {} wait failed: {}", req.name, err);
                    return;
                }
            };
            let _ = exit_tx.send(status.code());
            if pid > 0 {
                tracker.unregister(pid).await;
            }

            {
                let mut guard = daemons.write().await;
                guard.remove(&req.name);
            }
            info!("daemon {} exited with {}", req.name, status);
        });

        Ok(ExecDaemonResponse { is_new: true })
    }

    pub async fn get(&self, name: &str) -> Option<Arc<DaemonEntry>> {
        let guard = self.daemons.read().await;
        guard.get(name).cloned()
    }
}

pub struct DaemonEntry {
    pub(crate) _name: String,
    pub stdin: Mutex<tokio::process::ChildStdin>,
    pub stdout_tx: broadcast::Sender<Vec<u8>>,
    pub stderr_tx: broadcast::Sender<Vec<u8>>,
    pub output_backlog: Arc<Mutex<OutputBacklog>>,
    pub attach_lock: Arc<Mutex<()>>,
    pub exit_tx: watch::Sender<Option<i32>>,
}

#[derive(Clone, Copy, Debug)]
pub enum OutputKind {
    Stdout,
    Stderr,
}

#[derive(Clone, Debug)]
pub struct OutputFrame {
    pub kind: OutputKind,
    pub data: Vec<u8>,
}

#[derive(Default)]
pub struct OutputBacklog {
    frames: VecDeque<OutputFrame>,
}

impl OutputBacklog {
    fn push(&mut self, kind: OutputKind, data: Vec<u8>) {
        if self.frames.len() >= BACKLOG_MAX_FRAMES {
            let _ = self.frames.pop_front();
        }
        self.frames.push_back(OutputFrame { kind, data });
    }

    fn drain(&mut self) -> Vec<OutputFrame> {
        self.frames.drain(..).collect()
    }
}

impl DaemonEntry {
    pub async fn drain_output_backlog(&self) -> Vec<OutputFrame> {
        let mut guard = self.output_backlog.lock().await;
        guard.drain()
    }
}

fn spawn_reader<R>(
    mut reader: R,
    tx: broadcast::Sender<Vec<u8>>,
    backlog: Arc<Mutex<OutputBacklog>>,
    kind: OutputKind,
    label: String,
) where
    R: tokio::io::AsyncRead + Send + std::marker::Unpin + 'static,
{
    tokio::spawn(async move {
        let mut buf = vec![0u8; 4096];
        loop {
            match reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    let payload = buf[..n].to_vec();
                    if tx.receiver_count() == 0 {
                        let mut guard = backlog.lock().await;
                        guard.push(kind, payload.clone());
                    }
                    if tx.send(payload).is_err() {
                        // No active listeners; keep reading so buffer does not back up.
                    }
                }
                Err(err) => {
                    warn!("failed reading {}: {}", label, err);
                    break;
                }
            }
        }
    });
}

pub fn build_command_builder(
    shell: &str,
    args: &[String],
    env: &std::collections::HashMap<String, String>,
    cwd: Option<String>,
) -> CommandBuilder {
    let mut builder = CommandBuilder::new(shell);
    if !args.is_empty() {
        builder.args(args);
    }
    builder.env_clear();
    for (k, v) in env {
        builder.env(k, v);
    }
    if let Some(dir) = cwd {
        builder.cwd(PathBuf::from(dir));
    }
    builder
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use super::*;

    #[tokio::test]
    async fn output_backlog_replays_stdout_emitted_before_attach() {
        let tracker = ChildTracker::new();
        let registry = DaemonRegistry::new(tracker);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("unix time should be monotonic")
            .as_nanos();
        let daemon_name = format!("backlog-{nonce}");
        let req = ExecDaemonRequest {
            name: daemon_name.clone(),
            args: vec![
                "sh".to_string(),
                "-lc".to_string(),
                "echo preattach-output; sleep 1".to_string(),
            ],
            env: HashMap::new(),
        };

        registry
            .exec_daemon(req)
            .await
            .expect("exec_daemon should start process");
        tokio::time::sleep(Duration::from_millis(200)).await;

        let entry = registry
            .get(&daemon_name)
            .await
            .expect("daemon entry should exist before process exits");
        let frames = entry.drain_output_backlog().await;

        assert!(
            frames.iter().any(|frame| {
                matches!(frame.kind, OutputKind::Stdout)
                    && String::from_utf8_lossy(&frame.data).contains("preattach-output")
            }),
            "expected stdout emitted before attach to be retained in backlog"
        );
    }
}
