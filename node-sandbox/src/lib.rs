//! Node-API bindings for the Chevalier sandbox CLIENT. Connects to an external
//! `vmd` daemon; never spawns it. Isolated from the core `chevalier` binding so
//! the in-progress provider work in the sandbox crate can't break the core.
//!
//! v1 surface: connect/session/attachSession, exec (bidirectional ExecHandle),
//! readFile/writeFile, fork, sessionId/vmId. (shell, forwardPort, listDir,
//! snapshots/daemons are follow-ups — some need engine facade additions.)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use chevalier_sandbox::{
    EventStream, ExecEvent, ExecInput, ExecOptions, ForkOptions, Sandbox as EngineSandbox,
    SandboxConfig, SandboxError, Session as EngineSession, SessionOptions,
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
            ExecEvent::Stdout(b) => ExecEventJs { kind: "stdout".into(), data: Some(Buffer::from(b)), code: None },
            ExecEvent::Stderr(b) => ExecEventJs { kind: "stderr".into(), data: Some(Buffer::from(b)), code: None },
            ExecEvent::Exit(c) => ExecEventJs { kind: "exit".into(), data: None, code: Some(c) },
            ExecEvent::Timeout => ExecEventJs { kind: "timeout".into(), data: None, code: None },
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
        self.input
            .send(ExecInput::Resize { cols: cols as u16, rows: rows as u16 })
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
pub struct SessionOpts {
    pub session_id: Option<String>,
    pub name: Option<String>,
    pub image: Option<String>,
    pub architecture: Option<String>,
    pub metadata: Option<HashMap<String, String>>,
    pub auto_start: Option<bool>,
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
    pub async fn exec(&self, command: String, options: Option<ExecOpts>) -> napi::Result<ExecHandle> {
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
        self.inner.write_file(&path, data.to_vec()).await.map_err(sb_err)
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
}

/// A connection to a `vmd` sandbox daemon.
#[napi]
pub struct Sandbox {
    inner: EngineSandbox,
}

#[napi]
impl Sandbox {
    /// Connect to an external `vmd` at `endpoint` (e.g. `http://127.0.0.1:8052`).
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
        }
        let sb = EngineSandbox::connect(endpoint, cfg).await.map_err(sb_err)?;
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
        let s = self.inner.attach_session(&session_id).await.map_err(sb_err)?;
        Ok(Session { inner: s })
    }
}
