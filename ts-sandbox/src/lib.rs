//! Node-API bindings for the Chevalier sandbox CLIENT. Connects to an external
//! sandbox provider; never spawns one. Isolated from the core `chevalier`
//! binding so sandbox provider work can't break the core.
//!
//! v1 surface: connect/session/attachSession, exec (bidirectional ExecHandle),
//! readFile/writeFile, fork, sessionId/vmId. (shell, forwardPort, listDir,
//! snapshots/daemons are follow-ups — some need engine facade additions.)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chevalier_sandbox::{
    EventStream, ExecEvent, ExecInput, ExecOptions, ForkOptions, OpenComputerBackendConfig,
    OpenComputerMountConfig, Sandbox as EngineSandbox, SandboxConfig, SandboxError,
    SandboxProviderConfig, Session as EngineSession, SessionOptions, SharedMount,
    SharedMountAvailability, SharedMountContinuity,
};
use napi::bindgen_prelude::Buffer;
use napi_derive::napi;
use tokio::sync::Mutex;

fn sb_err(e: SandboxError) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, format!("Sandbox: {e}"))
}

// ---------------- exec streaming ----------------

/// A single exec event. `type` is `stdout` | `stderr` | `exit` | `timeout`.
#[napi(object)]
pub struct ExecEventJs {
    #[napi(js_name = "type")]
    pub kind: String,
    pub data: Option<Buffer>,
    pub code: Option<i32>,
}

impl From<ExecEvent> for ExecEventJs {
    fn from(e: ExecEvent) -> Self {
        match e {
            ExecEvent::Stdout(b) => ExecEventJs {
                kind: "stdout".into(),
                data: Some(Buffer::from(b)),
                code: None,
            },
            ExecEvent::Stderr(b) => ExecEventJs {
                kind: "stderr".into(),
                data: Some(Buffer::from(b)),
                code: None,
            },
            ExecEvent::Exit(c) => ExecEventJs {
                kind: "exit".into(),
                data: None,
                code: Some(c),
            },
            ExecEvent::Timeout => ExecEventJs {
                kind: "timeout".into(),
                data: None,
                code: None,
            },
        }
    }
}

/// Bidirectional handle to a running exec: write stdin, read events via `next()`.
#[napi]
pub struct ExecHandle {
    input: tokio::sync::mpsc::Sender<ExecInput>,
    events: Arc<Mutex<EventStream<ExecEvent>>>,
}

#[napi]
impl ExecHandle {
    #[napi]
    pub async fn write(&self, data: Buffer) -> napi::Result<()> {
        self.input
            .send(ExecInput::Data(data.to_vec()))
            .await
            .map_err(|_| napi::Error::from_reason("exec stdin closed"))
    }
    #[napi]
    pub async fn eof(&self) -> napi::Result<()> {
        self.input
            .send(ExecInput::Eof)
            .await
            .map_err(|_| napi::Error::from_reason("exec stdin closed"))
    }
    #[napi]
    pub async fn signal(&self, sig: i32) -> napi::Result<()> {
        self.input
            .send(ExecInput::Signal(sig))
            .await
            .map_err(|_| napi::Error::from_reason("exec stdin closed"))
    }
    #[napi]
    pub async fn resize(&self, cols: u32, rows: u32) -> napi::Result<()> {
        // Saturate rather than silently truncate (a terminal never exceeds u16).
        let clamp = |v: u32| u16::try_from(v).unwrap_or(u16::MAX);
        self.input
            .send(ExecInput::Resize {
                cols: clamp(cols),
                rows: clamp(rows),
            })
            .await
            .map_err(|_| napi::Error::from_reason("exec stdin closed"))
    }
    /// The next event, or `null` when the exec stream ends.
    #[napi]
    pub async fn next(&self) -> napi::Result<Option<ExecEventJs>> {
        use futures::StreamExt;
        let mut guard = self.events.lock().await;
        match guard.next().await {
            Some(Ok(ev)) => Ok(Some(ExecEventJs::from(ev))),
            Some(Err(e)) => Err(sb_err(e)),
            None => Ok(None),
        }
    }
}

// ---------------- options ----------------

#[napi(object)]
pub struct ExecOpts {
    pub env: Option<HashMap<String, String>>,
    pub timeout_secs: Option<i32>,
    pub detach: Option<bool>,
    pub shell: Option<String>,
    pub close_stdin_on_start: Option<bool>,
}

impl From<ExecOpts> for ExecOptions {
    fn from(o: ExecOpts) -> Self {
        ExecOptions {
            env: o.env.unwrap_or_default(),
            timeout_secs: o.timeout_secs,
            detach: o.detach.unwrap_or(false),
            shell: o.shell,
            close_stdin_on_start: o.close_stdin_on_start.unwrap_or(false),
        }
    }
}

#[napi(object)]
pub struct SharedMountOpts {
    pub host_path: Option<String>,
    pub guest_path: String,
    pub mount_tag: String,
    pub read_only: Option<bool>,
    pub availability: Option<String>,
    pub continuity: Option<String>,
    pub backend_profile: Option<String>,
    pub vfs_endpoint: Option<String>,
    pub vfs_scope_path: Option<String>,
}

fn shared_mount_availability(value: Option<String>) -> SharedMountAvailability {
    match value.as_deref() {
        Some("shared-storage") | Some("shared_storage") | Some("sharedStorage") => {
            SharedMountAvailability::SharedStorage
        }
        _ => SharedMountAvailability::NodeLocal,
    }
}

fn shared_mount_continuity(
    value: Option<String>,
    availability: &SharedMountAvailability,
) -> SharedMountContinuity {
    match value.as_deref() {
        Some("restore-cross-node") | Some("restore_cross_node") | Some("restoreCrossNode") => {
            SharedMountContinuity::RestoreCrossNode
        }
        Some("restart-same-node") | Some("restart_same_node") | Some("restartSameNode") => {
            SharedMountContinuity::RestartSameNode
        }
        _ => match availability {
            SharedMountAvailability::SharedStorage => SharedMountContinuity::RestoreCrossNode,
            SharedMountAvailability::NodeLocal => SharedMountContinuity::RestartSameNode,
        },
    }
}

impl SharedMountOpts {
    fn into_shared_mount(self) -> SharedMount {
        let availability = shared_mount_availability(self.availability);
        let continuity = shared_mount_continuity(self.continuity, &availability);
        SharedMount {
            host_path: self.host_path.unwrap_or_default(),
            guest_path: self.guest_path,
            mount_tag: self.mount_tag,
            read_only: self.read_only.unwrap_or(false),
            availability,
            continuity,
            backend_profile: self.backend_profile.unwrap_or_default(),
            vfs_endpoint: self.vfs_endpoint.unwrap_or_default(),
            vfs_scope_path: self.vfs_scope_path.unwrap_or_default(),
        }
    }
}

#[napi(object)]
pub struct SessionOpts {
    pub session_id: Option<String>,
    pub name: Option<String>,
    pub image: Option<String>,
    pub architecture: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
    pub auto_start: Option<bool>,
    pub shared_mounts: Option<Vec<SharedMountOpts>>,
    pub egress_allowlist: Option<Vec<String>>,
}

impl From<SessionOpts> for SessionOptions {
    fn from(o: SessionOpts) -> Self {
        SessionOptions {
            session_id: o.session_id,
            name: o.name,
            image: o.image,
            architecture: o.architecture,
            metadata: o.metadata.unwrap_or_default(),
            auto_start: o.auto_start.unwrap_or(true),
            shared_mounts: o
                .shared_mounts
                .unwrap_or_default()
                .into_iter()
                .map(SharedMountOpts::into_shared_mount)
                .collect(),
            egress_allowlist: o.egress_allowlist,
            ..Default::default()
        }
    }
}

#[napi(object)]
pub struct ForkOpts {
    pub child_name: Option<String>,
    pub child_metadata: Option<HashMap<String, String>>,
    pub auto_start_child: Option<bool>,
}

impl From<ForkOpts> for ForkOptions {
    fn from(o: ForkOpts) -> Self {
        ForkOptions {
            child_name: o.child_name,
            child_metadata: o.child_metadata.unwrap_or_default(),
            auto_start_child: o.auto_start_child.unwrap_or(true),
        }
    }
}

// ---------------- session ----------------

/// A sandbox session (one microVM).
#[napi]
pub struct Session {
    inner: EngineSession,
}

#[napi]
impl Session {
    #[napi(getter)]
    pub fn session_id(&self) -> String {
        self.inner.session_id().to_string()
    }
    #[napi(getter)]
    pub fn vm_id(&self) -> String {
        self.inner.vm_id().to_string()
    }

    /// Start a command; returns a bidirectional `ExecHandle`.
    #[napi]
    pub async fn exec(
        &self,
        command: String,
        options: Option<ExecOpts>,
    ) -> napi::Result<ExecHandle> {
        let opts = options.map(Into::into).unwrap_or_default();
        let h = self.inner.exec(&command, opts).await.map_err(sb_err)?;
        Ok(ExecHandle {
            input: h.input,
            events: Arc::new(Mutex::new(h.events)),
        })
    }

    /// Read a file from the guest.
    #[napi]
    pub async fn read_file(&self, path: String) -> napi::Result<Buffer> {
        let b = self.inner.read_file(&path).await.map_err(sb_err)?;
        Ok(Buffer::from(b))
    }

    /// Write a file to the guest.
    #[napi]
    pub async fn write_file(&self, path: String, data: Buffer) -> napi::Result<()> {
        self.inner
            .write_file(&path, data.to_vec())
            .await
            .map_err(sb_err)
    }

    /// Fork this session (CoW); returns the child session.
    #[napi]
    pub async fn fork(&self, options: Option<ForkOpts>) -> napi::Result<Session> {
        let opts = options.map(Into::into).unwrap_or(ForkOptions {
            child_name: None,
            child_metadata: HashMap::new(),
            auto_start_child: true,
        });
        let r = self.inner.fork(opts).await.map_err(sb_err)?;
        Ok(Session { inner: r.child })
    }
}

// ---------------- sandbox ----------------

#[napi(object)]
pub struct SandboxConnectOptions {
    pub auth_token: Option<String>,
    pub connect_timeout_ms: Option<f64>,
    pub default_image: Option<String>,
    pub provider: Option<String>,
    pub open_computer: Option<OpenComputerProviderOpts>,
}

#[napi(object)]
pub struct OpenComputerProviderOpts {
    pub api_url: Option<String>,
    pub api_key: Option<String>,
    pub template_id: Option<String>,
    pub timeout_secs: Option<f64>,
    pub default_cpu_count: Option<u32>,
    pub default_memory_mb: Option<u32>,
    pub default_disk_mb: Option<u32>,
    pub burst: Option<bool>,
    pub secret_store: Option<String>,
    pub egress_allowlist: Option<Vec<String>>,
    pub mounts: Option<Vec<OpenComputerMountOpts>>,
    pub shared_mounts: Option<HashMap<String, OpenComputerMountOpts>>,
}

#[napi(object)]
pub struct OpenComputerMountOpts {
    pub path: Option<String>,
    pub driver: Option<String>,
    pub remote: Option<String>,
    pub backend: Option<String>,
    pub command: Option<Vec<String>>,
    pub env: Option<HashMap<String, String>>,
    pub secrets: Option<HashMap<String, String>>,
    pub creds: Option<HashMap<String, String>>,
    pub rclone_config: Option<String>,
    pub read_only: Option<bool>,
    pub mount_options: Option<Vec<String>>,
}

impl OpenComputerMountOpts {
    fn into_config(self) -> OpenComputerMountConfig {
        OpenComputerMountConfig {
            path: self.path.unwrap_or_default(),
            driver: self.driver,
            remote: self.remote.unwrap_or_default(),
            backend: self.backend,
            command: self.command.unwrap_or_default(),
            env: self.env.unwrap_or_default(),
            secrets: self.secrets.unwrap_or_default(),
            creds: self.creds.unwrap_or_default(),
            rclone_config: self.rclone_config,
            read_only: self.read_only,
            mount_options: self.mount_options.unwrap_or_default(),
        }
    }
}

fn opencomputer_config_from_options(
    options: Option<OpenComputerProviderOpts>,
) -> Result<OpenComputerBackendConfig, SandboxError> {
    let mut cfg = OpenComputerBackendConfig::from_env()?;
    if let Some(options) = options {
        if let Some(api_url) = options.api_url {
            cfg.api_url = api_url;
        }
        if let Some(api_key) = options.api_key {
            cfg.api_key = api_key;
        }
        if let Some(template_id) = options.template_id {
            cfg.template_id = template_id;
        }
        if let Some(timeout_secs) = options.timeout_secs {
            cfg.timeout_secs = timeout_secs as u64;
        }
        cfg.default_cpu_count = options.default_cpu_count.or(cfg.default_cpu_count);
        cfg.default_memory_mb = options.default_memory_mb.or(cfg.default_memory_mb);
        cfg.default_disk_mb = options.default_disk_mb.or(cfg.default_disk_mb);
        cfg.burst = options.burst.or(cfg.burst);
        cfg.secret_store = options.secret_store.or(cfg.secret_store);
        cfg.egress_allowlist = options.egress_allowlist.or(cfg.egress_allowlist);
        if let Some(mounts) = options.mounts {
            cfg.mounts = mounts
                .into_iter()
                .map(OpenComputerMountOpts::into_config)
                .collect();
        }
        if let Some(shared_mounts) = options.shared_mounts {
            cfg.shared_mounts = shared_mounts
                .into_iter()
                .map(|(key, mount)| (key, mount.into_config()))
                .collect();
        }
    }
    Ok(cfg)
}

/// A connection to a Chevalier sandbox provider.
#[napi]
pub struct Sandbox {
    inner: EngineSandbox,
}

#[napi]
impl Sandbox {
    /// Connect to a sandbox provider.
    #[napi(factory)]
    pub async fn connect(
        endpoint: String,
        options: Option<SandboxConnectOptions>,
    ) -> napi::Result<Sandbox> {
        let mut cfg = SandboxConfig {
            endpoint: endpoint.clone(),
            ..Default::default()
        };
        if let Some(o) = options {
            if let Some(t) = o.auth_token {
                cfg.auth_token = Some(t);
            }
            if let Some(ms) = o.connect_timeout_ms {
                cfg.connect_timeout = Duration::from_millis(ms as u64);
            }
            if let Some(img) = o.default_image {
                cfg.default_image = img;
            }
            let provider = o.provider.unwrap_or_else(|| "chevalier".to_string());
            match provider.as_str() {
                "chevalier" | "local" | "vmd" => {}
                "opencomputer" | "open-computer" => {
                    cfg.provider = SandboxProviderConfig::OpenComputer(
                        opencomputer_config_from_options(o.open_computer).map_err(sb_err)?,
                    );
                }
                other => {
                    return Err(sb_err(SandboxError::InvalidConfig(format!(
                        "unsupported sandbox provider `{other}`"
                    ))));
                }
            }
        }
        let sb = EngineSandbox::connect(endpoint, cfg)
            .await
            .map_err(sb_err)?;
        Ok(Sandbox { inner: sb })
    }

    /// Create a new session (microVM).
    #[napi]
    pub async fn session(&self, options: Option<SessionOpts>) -> napi::Result<Session> {
        let opts = options.map(Into::into).unwrap_or_default();
        let s = self.inner.session(opts).await.map_err(sb_err)?;
        Ok(Session { inner: s })
    }

    /// Attach to an existing session by id.
    #[napi]
    pub async fn attach_session(&self, session_id: String) -> napi::Result<Session> {
        let s = self
            .inner
            .attach_session(&session_id)
            .await
            .map_err(sb_err)?;
        Ok(Session { inner: s })
    }
}
