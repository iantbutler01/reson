use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use portable_pty::CommandBuilder;
use thiserror::Error;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{Mutex, RwLock, broadcast};
use tracing::{error, info, warn};

use crate::child_tracker::ChildTracker;
use crate::pb::bracket::portproxy::v1::{ExecDaemonRequest, ExecDaemonResponse};

const CHANNEL_CAPACITY: usize = 100;

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

        let entry = Arc::new(DaemonEntry {
            _name: req.name.clone(),
            stdin: Mutex::new(stdin),
            stdout_tx: stdout_tx.clone(),
            stderr_tx: stderr_tx.clone(),
            attach_lock: Arc::new(Mutex::new(())),
        });

        {
            let mut daemons = self.daemons.write().await;
            if daemons.contains_key(&req.name) {
                return Err(DaemonError::AlreadyRunning);
            }
            daemons.insert(req.name.clone(), entry.clone());
        }

        spawn_reader(stdout, stdout_tx, format!("stdout({})", req.name));
        spawn_reader(stderr, stderr_tx, format!("stderr({})", req.name));

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
    pub attach_lock: Arc<Mutex<()>>,
}

fn spawn_reader<R>(mut reader: R, tx: broadcast::Sender<Vec<u8>>, label: String)
where
    R: tokio::io::AsyncRead + Send + std::marker::Unpin + 'static,
{
    tokio::spawn(async move {
        let mut buf = vec![0u8; 4096];
        loop {
            match reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    let payload = buf[..n].to_vec();
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
