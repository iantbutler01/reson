#![allow(clippy::large_enum_variant)]

use std::collections::HashMap;
use std::net::TcpListener;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::Stream;
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc};
use tokio_stream::wrappers::ReceiverStream;
use tonic::transport::Endpoint;
use tonic::Request;
use uuid::Uuid;

pub mod proto {
    pub mod bracket {
        pub mod portproxy {
            pub mod v1 {
                include!(concat!(env!("OUT_DIR"), "/bracket.portproxy.v1.rs"));
            }
        }
    }

    pub mod google {
        pub mod protobuf {
            include!(concat!(env!("OUT_DIR"), "/google.protobuf.rs"));
        }
    }

    pub mod vmd {
        pub mod v1 {
            include!(concat!(env!("OUT_DIR"), "/vmd.v1.rs"));
        }
    }
}

use proto::bracket::portproxy::v1::port_proxy_client::PortProxyClient;
use proto::bracket::portproxy::v1::shell_exec_client::ShellExecClient;
use proto::bracket::portproxy::v1::{
    DeletePathRequest, ExecRequest, ExecResponse, ExecStart, InteractiveShellRequest,
    InteractiveShellResponse, InteractiveShellStart, ListDirectoryRequest, ReadFileRequest,
    WriteFileRequest, exec_request, exec_response, interactive_shell_request,
    interactive_shell_response,
};
use proto::vmd::v1::vmd_service_client::VmdServiceClient;
use proto::vmd::v1::{
    CreateVmRequest, ForkVmRequest, GetVmRequest, ListVMsRequest, Metadata, ResourceSpec,
    Vm, VmActionRequest, VmSource, VmSourceType,
};

const META_SESSION_ID: &str = "reson.session_id";
const META_BRANCH_ID: &str = "reson.branch_id";
const META_PARENT_SESSION_ID: &str = "reson.parent_session_id";
const META_PARENT_VM_ID: &str = "reson.parent_vm_id";
const META_FORK_ID: &str = "reson.fork_id";

#[derive(thiserror::Error, Debug)]
pub enum SandboxError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("transport error: {0}")]
    Transport(#[from] tonic::transport::Error),
    #[error("gRPC status: {0}")]
    Grpc(#[from] tonic::Status),
    #[error("invalid endpoint: {0}")]
    InvalidEndpoint(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("daemon unavailable: {0}")]
    DaemonUnavailable(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

pub type Result<T> = std::result::Result<T, SandboxError>;

#[derive(Clone, Debug)]
pub struct ResourceLimits {
    pub vcpu: i32,
    pub memory_mb: i32,
    pub disk_gb: i32,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            vcpu: 2,
            memory_mb: 2048,
            disk_gb: 10,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SandboxConfig {
    pub endpoint: String,
    pub auto_spawn: bool,
    pub daemon_listen: String,
    pub daemon_bin: Option<PathBuf>,
    pub daemon_data_dir: Option<PathBuf>,
    pub daemon_start_timeout: Duration,
    pub portproxy_ready_timeout: Duration,
    pub connect_timeout: Duration,
    pub default_image: String,
    pub default_architecture: Option<String>,
    pub default_resources: ResourceLimits,
    pub default_shell: String,
    pub portproxy_client_bin: Option<PathBuf>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8052".to_string(),
            auto_spawn: true,
            daemon_listen: "127.0.0.1:8052".to_string(),
            daemon_bin: None,
            daemon_data_dir: None,
            daemon_start_timeout: Duration::from_secs(20),
            portproxy_ready_timeout: Duration::from_secs(90),
            connect_timeout: Duration::from_secs(5),
            default_image: std::env::var("BRACKET_VM_IMAGE")
                .unwrap_or_else(|_| "ghcr.io/bracketdevelopers/uv-builder:main".to_string()),
            default_architecture: None,
            default_resources: ResourceLimits::default(),
            default_shell: "bash".to_string(),
            portproxy_client_bin: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionOptions {
    pub session_id: Option<String>,
    pub name: Option<String>,
    pub image: Option<String>,
    pub architecture: Option<String>,
    pub metadata: HashMap<String, String>,
    pub auto_start: bool,
    pub resources: Option<ResourceLimits>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            session_id: None,
            name: None,
            image: None,
            architecture: None,
            metadata: HashMap::new(),
            auto_start: true,
            resources: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ForkOptions {
    pub child_name: Option<String>,
    pub child_metadata: HashMap<String, String>,
    pub auto_start_child: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ExecOptions {
    pub env: HashMap<String, String>,
    pub timeout_secs: Option<i32>,
    pub detach: bool,
    pub shell: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ShellOptions {
    pub shell: Option<String>,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub cwd: Option<String>,
}

#[derive(Clone, Debug)]
pub enum ExecInput {
    Data(Vec<u8>),
    Eof,
    Signal(i32),
    Resize { cols: u16, rows: u16 },
}

#[derive(Clone, Debug)]
pub enum ExecEvent {
    Stdout(Vec<u8>),
    Stderr(Vec<u8>),
    Exit(i32),
    Timeout,
}

#[derive(Clone, Debug)]
pub enum ShellInput {
    Data(Vec<u8>),
    Eof,
}

#[derive(Clone, Debug)]
pub enum ShellEvent {
    Output(Vec<u8>),
    Exit(i32),
}

pub type EventStream<T> = Pin<Box<dyn Stream<Item = Result<T>> + Send>>;

pub struct ExecHandle {
    pub input: mpsc::Sender<ExecInput>,
    pub events: EventStream<ExecEvent>,
}

pub struct ShellHandle {
    pub input: mpsc::Sender<ShellInput>,
    pub events: EventStream<ShellEvent>,
}

pub struct ForwardHandle {
    pub guest_port: u16,
    pub host_port: u16,
    child: Arc<Mutex<Option<Child>>>,
}

impl ForwardHandle {
    pub async fn close(&self) -> Result<()> {
        let mut guard = self.child.lock().await;
        if let Some(child) = guard.as_mut() {
            let _ = child.start_kill();
            let _ = child.wait().await;
        }
        *guard = None;
        Ok(())
    }
}

impl Drop for ForwardHandle {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.try_lock() {
            if let Some(child) = guard.as_mut() {
                let _ = child.start_kill();
            }
            *guard = None;
        }
    }
}

#[derive(Clone, Debug)]
pub struct SessionInfo {
    pub session_id: String,
    pub vm_id: String,
    pub name: String,
    pub state: i32,
    pub branch_id: Option<String>,
    pub parent_session_id: Option<String>,
    pub fork_id: Option<String>,
}

pub struct ForkResult {
    pub parent_session_id: String,
    pub child_session_id: String,
    pub fork_id: String,
    pub child: Session,
}

#[derive(Clone)]
pub struct Sandbox {
    inner: Arc<SandboxInner>,
}

struct ManagedDaemon {
    child: Child,
}

struct SandboxInner {
    cfg: SandboxConfig,
    managed_daemon: Mutex<Option<ManagedDaemon>>,
    ready_vm_rpc: Mutex<HashMap<String, i32>>,
}

#[derive(Clone)]
pub struct Session {
    sandbox: Sandbox,
    session_id: String,
    vm_id: String,
}

impl Session {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn vm_id(&self) -> &str {
        &self.vm_id
    }

    pub async fn exec(&self, command: &str, opts: ExecOptions) -> Result<ExecHandle> {
        let rpc_port = self.sandbox.ensure_vm_and_get_rpc_port(&self.vm_id).await?;
        let endpoint = format!("http://127.0.0.1:{rpc_port}");
        let mut client = ShellExecClient::connect(endpoint).await?;

        let shell = opts
            .shell
            .clone()
            .unwrap_or_else(|| self.sandbox.inner.cfg.default_shell.clone());

        let (req_tx, req_rx) = mpsc::channel(64);
        req_tx
            .send(ExecRequest {
                request: Some(exec_request::Request::Start(ExecStart {
                    args: vec![shell, "-lc".to_string(), command.to_string()],
                    env: opts.env,
                    detach: opts.detach,
                    timeout: opts.timeout_secs,
                })),
            })
            .await
            .map_err(|_| SandboxError::InvalidResponse("failed to enqueue exec start".into()))?;

        let mut stream = client
            .exec(Request::new(ReceiverStream::new(req_rx)))
            .await?
            .into_inner();

        let (input_tx, mut input_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::channel(128);

        let req_tx_for_input = req_tx.clone();
        let event_tx_input = event_tx.clone();
        tokio::spawn(async move {
            let req_tx = req_tx_for_input;
            while let Some(input) = input_rx.recv().await {
                match input {
                    ExecInput::Data(data) => {
                        if req_tx
                            .send(ExecRequest {
                                request: Some(exec_request::Request::StdinData(data)),
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ExecInput::Eof => break,
                    ExecInput::Signal(sig) => {
                        let _ = event_tx_input
                            .send(Err(SandboxError::Unsupported(format!(
                                "exec signal forwarding not supported by portproxy API: {sig}"
                            ))))
                            .await;
                    }
                    ExecInput::Resize { cols, rows } => {
                        let _ = event_tx_input
                            .send(Err(SandboxError::Unsupported(format!(
                                "exec resize not supported by portproxy API: {cols}x{rows}"
                            ))))
                            .await;
                    }
                }
            }
            drop(req_tx);
        });

        let event_tx_stream = event_tx.clone();
        tokio::spawn(async move {
            loop {
                match stream.message().await {
                    Ok(Some(ExecResponse {
                        response: Some(exec_response::Response::StdoutData(data)),
                    })) => {
                        if event_tx_stream.send(Ok(ExecEvent::Stdout(data))).await.is_err() {
                            break;
                        }
                    }
                    Ok(Some(ExecResponse {
                        response: Some(exec_response::Response::StderrData(data)),
                    })) => {
                        if event_tx_stream.send(Ok(ExecEvent::Stderr(data))).await.is_err() {
                            break;
                        }
                    }
                    Ok(Some(ExecResponse {
                        response: Some(exec_response::Response::ExitCode(code)),
                    })) => {
                        if code == 124 {
                            let _ = event_tx_stream.send(Ok(ExecEvent::Timeout)).await;
                        }
                        let _ = event_tx_stream.send(Ok(ExecEvent::Exit(code))).await;
                        break;
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(status) => {
                        let _ = event_tx_stream.send(Err(SandboxError::Grpc(status))).await;
                        break;
                    }
                }
            }
        });

        Ok(ExecHandle {
            input: input_tx,
            events: Box::pin(ReceiverStream::new(event_rx)),
        })
    }

    pub async fn shell(&self, opts: ShellOptions) -> Result<ShellHandle> {
        let rpc_port = self.sandbox.ensure_vm_and_get_rpc_port(&self.vm_id).await?;
        let endpoint = format!("http://127.0.0.1:{rpc_port}");
        let mut client = ShellExecClient::connect(endpoint).await?;

        let shell = opts
            .shell
            .clone()
            .unwrap_or_else(|| self.sandbox.inner.cfg.default_shell.clone());

        let (req_tx, req_rx) = mpsc::channel(64);
        req_tx
            .send(InteractiveShellRequest {
                request: Some(interactive_shell_request::Request::Start(
                    InteractiveShellStart {
                        shell,
                        args: opts.args,
                        env: opts.env,
                        cwd: opts.cwd.unwrap_or_default(),
                    },
                )),
            })
            .await
            .map_err(|_| SandboxError::InvalidResponse("failed to enqueue shell start".into()))?;

        let mut stream = client
            .interactive_shell(Request::new(ReceiverStream::new(req_rx)))
            .await?
            .into_inner();

        let (input_tx, mut input_rx) = mpsc::channel(64);
        let (event_tx, event_rx) = mpsc::channel(128);

        let req_tx_for_input = req_tx.clone();
        tokio::spawn(async move {
            let req_tx = req_tx_for_input;
            while let Some(input) = input_rx.recv().await {
                match input {
                    ShellInput::Data(data) => {
                        if req_tx
                            .send(InteractiveShellRequest {
                                request: Some(interactive_shell_request::Request::StdinData(data)),
                            })
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    ShellInput::Eof => break,
                }
            }
            drop(req_tx);
        });

        let event_tx_stream = event_tx.clone();
        tokio::spawn(async move {
            loop {
                match stream.message().await {
                    Ok(Some(InteractiveShellResponse {
                        response: Some(interactive_shell_response::Response::OutputData(data)),
                    })) => {
                        if event_tx_stream.send(Ok(ShellEvent::Output(data))).await.is_err() {
                            break;
                        }
                    }
                    Ok(Some(InteractiveShellResponse {
                        response: Some(interactive_shell_response::Response::ExitCode(code)),
                    })) => {
                        let _ = event_tx_stream.send(Ok(ShellEvent::Exit(code))).await;
                        break;
                    }
                    Ok(Some(_)) => {}
                    Ok(None) => break,
                    Err(status) => {
                        let _ = event_tx_stream.send(Err(SandboxError::Grpc(status))).await;
                        break;
                    }
                }
            }
        });

        Ok(ShellHandle {
            input: input_tx,
            events: Box::pin(ReceiverStream::new(event_rx)),
        })
    }

    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let mut client = self.sandbox.portproxy_client_for_vm(&self.vm_id).await?;
        let response = client
            .read_file(Request::new(ReadFileRequest {
                path: path.to_string(),
            }))
            .await?
            .into_inner();
        Ok(response.data)
    }

    pub async fn write_file(&self, path: &str, data: Vec<u8>) -> Result<()> {
        let mut client = self.sandbox.portproxy_client_for_vm(&self.vm_id).await?;
        client
            .write_file(Request::new(WriteFileRequest {
                path: path.to_string(),
                data,
                create_parents: true,
            }))
            .await?;
        Ok(())
    }

    pub async fn list_dir(
        &self,
        path: &str,
    ) -> Result<Vec<proto::bracket::portproxy::v1::DirectoryEntry>> {
        let mut client = self.sandbox.portproxy_client_for_vm(&self.vm_id).await?;
        let response = client
            .list_directory(Request::new(ListDirectoryRequest {
                path: path.to_string(),
            }))
            .await?
            .into_inner();
        Ok(response.entries)
    }

    pub async fn delete_path(&self, path: &str) -> Result<()> {
        let mut client = self.sandbox.portproxy_client_for_vm(&self.vm_id).await?;
        client
            .delete_path(Request::new(DeletePathRequest {
                path: path.to_string(),
            }))
            .await?;
        Ok(())
    }

    pub async fn forward_port(&self, guest_port: u16) -> Result<ForwardHandle> {
        let vm = self.sandbox.ensure_vm_running(&self.vm_id).await?;
        let proxy_port = vm
            .network
            .and_then(|n| n.portproxy_ports)
            .map(|p| p.proxy_port)
            .ok_or_else(|| SandboxError::InvalidResponse("VM missing proxy port".into()))?;

        let host_port = {
            let listener = TcpListener::bind("127.0.0.1:0")?;
            listener.local_addr()?.port()
        };

        let bin = self
            .sandbox
            .inner
            .cfg
            .portproxy_client_bin
            .clone()
            .unwrap_or_else(|| PathBuf::from("portproxy"));

        let child = Command::new(bin)
            .arg("--listen-port")
            .arg(host_port.to_string())
            .arg("--forward-port")
            .arg(guest_port.to_string())
            .arg("--server-addr")
            .arg(format!("0.0.0.0:{proxy_port}"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        Ok(ForwardHandle {
            guest_port,
            host_port,
            child: Arc::new(Mutex::new(Some(child))),
        })
    }

    pub async fn fork(&self, opts: ForkOptions) -> Result<ForkResult> {
        let auto_start_child = opts.auto_start_child;
        let child_session_id = Uuid::new_v4().to_string();
        let mut child_metadata = opts.child_metadata;
        child_metadata.insert(META_SESSION_ID.to_string(), child_session_id.clone());
        child_metadata.insert(META_PARENT_SESSION_ID.to_string(), self.session_id.clone());
        child_metadata.insert(META_PARENT_VM_ID.to_string(), self.vm_id.clone());
        child_metadata.insert(META_BRANCH_ID.to_string(), child_session_id.clone());

        let mut client = self.sandbox.vmd_client().await?;
        let response = client
            .fork_vm(Request::new(ForkVmRequest {
                parent_vm_id: self.vm_id.clone(),
                child_name: opts.child_name.unwrap_or_default(),
                child_metadata: Some(Metadata {
                    entries: child_metadata,
                }),
                auto_start_child,
            }))
            .await?
            .into_inner();

        let child_vm = response
            .child_vm
            .ok_or_else(|| SandboxError::InvalidResponse("fork response missing child VM".into()))?;

        if auto_start_child {
            let _ = self.sandbox.ensure_vm_and_get_rpc_port(&child_vm.id).await?;
        }

        let child = Session {
            sandbox: self.sandbox.clone(),
            session_id: child_session_id.clone(),
            vm_id: child_vm.id,
        };

        Ok(ForkResult {
            parent_session_id: self.session_id.clone(),
            child_session_id,
            fork_id: response.fork_id,
            child,
        })
    }

    pub async fn close(self) -> Result<()> {
        Ok(())
    }

    pub async fn discard(self) -> Result<()> {
        self.sandbox.discard_vm(&self.vm_id).await
    }
}

impl Sandbox {
    pub async fn new(mut config: SandboxConfig) -> Result<Self> {
        config.endpoint = normalize_endpoint(&config.endpoint)?;

        let sandbox = Self {
            inner: Arc::new(SandboxInner {
                cfg: config,
                managed_daemon: Mutex::new(None),
                ready_vm_rpc: Mutex::new(HashMap::new()),
            }),
        };

        sandbox.ensure_daemon_ready().await?;
        Ok(sandbox)
    }

    pub async fn connect(endpoint: impl Into<String>, mut config: SandboxConfig) -> Result<Self> {
        config.endpoint = normalize_endpoint(&endpoint.into())?;
        config.auto_spawn = false;
        Self::new(config).await
    }

    pub async fn session(&self, opts: SessionOptions) -> Result<Session> {
        let session_id = opts
            .session_id
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let auto_start = opts.auto_start;

        if let Some(vm) = self.find_vm_by_session_id(&session_id).await? {
            let vm = self.ensure_vm_running(&vm.id).await?;
            let _ = self.ensure_vm_and_get_rpc_port(&vm.id).await?;
            return Ok(Session {
                sandbox: self.clone(),
                session_id,
                vm_id: vm.id,
            });
        }

        let mut metadata = opts.metadata;
        metadata.insert(META_SESSION_ID.to_string(), session_id.clone());
        metadata.insert(META_BRANCH_ID.to_string(), session_id.clone());

        let resources = opts
            .resources
            .unwrap_or_else(|| self.inner.cfg.default_resources.clone());
        let name = opts
            .name
            .unwrap_or_else(|| format!("reson-session-{}", &session_id[..8]));
        let image = opts
            .image
            .unwrap_or_else(|| self.inner.cfg.default_image.clone());

        let request = CreateVmRequest {
            name,
            source: Some(VmSource {
                r#type: VmSourceType::Docker as i32,
                reference: image,
            }),
            resources: Some(ResourceSpec {
                vcpu: resources.vcpu,
                memory_mb: resources.memory_mb,
                disk_gb: resources.disk_gb,
            }),
            metadata: Some(Metadata { entries: metadata }),
            auto_start: opts.auto_start,
            architecture: opts.architecture.unwrap_or_else(|| {
                self.inner
                    .cfg
                    .default_architecture
                    .clone()
                    .unwrap_or_default()
            }),
        };

        let mut client = self.vmd_client().await?;
        let mut stream = client.create_vm(Request::new(request)).await?.into_inner();

        let mut final_vm: Option<Vm> = None;
        while let Some(update) = stream.message().await? {
            if let Some(proto::vmd::v1::create_vm_stream_response::Event::Vm(vm)) = update.event {
                final_vm = Some(vm);
            }
        }

        let vm = final_vm
            .ok_or_else(|| SandboxError::InvalidResponse("create_vm stream missing VM".into()))?;

        let running_state = proto::vmd::v1::VmState::Running as i32;
        if auto_start || vm.state == running_state {
            let _ = self.ensure_vm_and_get_rpc_port(&vm.id).await?;
        }

        Ok(Session {
            sandbox: self.clone(),
            session_id,
            vm_id: vm.id,
        })
    }

    pub async fn attach_session(&self, session_id: &str) -> Result<Session> {
        let vm = self
            .find_vm_by_session_id(session_id)
            .await?
            .ok_or_else(|| SandboxError::SessionNotFound(session_id.to_string()))?;
        let vm = self.ensure_vm_running(&vm.id).await?;
        let _ = self.ensure_vm_and_get_rpc_port(&vm.id).await?;

        Ok(Session {
            sandbox: self.clone(),
            session_id: session_id.to_string(),
            vm_id: vm.id,
        })
    }

    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let mut client = self.vmd_client().await?;
        let response = client
            .list_v_ms(Request::new(ListVMsRequest {
                include_snapshots: false,
            }))
            .await?
            .into_inner();

        let mut sessions = Vec::new();
        for vm in response.vms {
            let Some(session_id) = vm.metadata.get(META_SESSION_ID).cloned() else {
                continue;
            };
            sessions.push(SessionInfo {
                session_id,
                vm_id: vm.id,
                name: vm.name,
                state: vm.state,
                branch_id: vm.metadata.get(META_BRANCH_ID).cloned(),
                parent_session_id: vm.metadata.get(META_PARENT_SESSION_ID).cloned(),
                fork_id: vm.metadata.get(META_FORK_ID).cloned(),
            });
        }

        Ok(sessions)
    }

    async fn vmd_client(&self) -> Result<VmdServiceClient<tonic::transport::Channel>> {
        let endpoint = Endpoint::from_shared(self.inner.cfg.endpoint.clone())
            .map_err(|err| SandboxError::InvalidEndpoint(err.to_string()))?
            .connect_timeout(self.inner.cfg.connect_timeout)
            .timeout(self.inner.cfg.connect_timeout);
        Ok(VmdServiceClient::connect(endpoint).await?)
    }

    async fn ensure_daemon_ready(&self) -> Result<()> {
        if self.health_check().await.is_ok() {
            return Ok(());
        }

        if !self.inner.cfg.auto_spawn {
            return Err(SandboxError::DaemonUnavailable(format!(
                "unable to connect to sandbox daemon at {}",
                self.inner.cfg.endpoint
            )));
        }

        self.spawn_daemon_if_needed().await?;

        let start = Instant::now();
        while start.elapsed() < self.inner.cfg.daemon_start_timeout {
            if self.health_check().await.is_ok() {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        Err(SandboxError::DaemonUnavailable(format!(
            "sandbox daemon did not become ready at {}",
            self.inner.cfg.endpoint
        )))
    }

    async fn spawn_daemon_if_needed(&self) -> Result<()> {
        let mut guard = self.inner.managed_daemon.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        let bin = self
            .inner
            .cfg
            .daemon_bin
            .clone()
            .unwrap_or_else(|| PathBuf::from("vmd"));

        let mut cmd = Command::new(bin);
        cmd.arg("--listen").arg(&self.inner.cfg.daemon_listen);
        if let Some(data_dir) = &self.inner.cfg.daemon_data_dir {
            cmd.arg("--data-dir").arg(data_dir);
        }
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::null());

        let child = cmd.spawn()?;
        *guard = Some(ManagedDaemon { child });
        Ok(())
    }

    async fn health_check(&self) -> Result<()> {
        let mut client = self.vmd_client().await?;
        let _ = client
            .health(Request::new(proto::vmd::v1::HealthRequest {}))
            .await?;
        Ok(())
    }

    async fn find_vm_by_session_id(&self, session_id: &str) -> Result<Option<Vm>> {
        let mut client = self.vmd_client().await?;
        let response = client
            .list_v_ms(Request::new(ListVMsRequest {
                include_snapshots: false,
            }))
            .await?
            .into_inner();

        Ok(response
            .vms
            .into_iter()
            .find(|vm| vm.metadata.get(META_SESSION_ID).map(String::as_str) == Some(session_id)))
    }

    async fn ensure_vm_running(&self, vm_id: &str) -> Result<Vm> {
        let mut client = self.vmd_client().await?;
        let mut vm = client
            .get_vm(Request::new(GetVmRequest {
                vm_id: vm_id.to_string(),
            }))
            .await?
            .into_inner();

        let is_running = vm.state == proto::vmd::v1::VmState::Running as i32;
        if !is_running {
            let mut ready = self.inner.ready_vm_rpc.lock().await;
            ready.remove(vm_id);
            drop(ready);
            vm = client
                .start_vm(Request::new(VmActionRequest {
                    vm_id: vm_id.to_string(),
                }))
                .await?
                .into_inner();
        }

        Ok(vm)
    }

    async fn ensure_vm_and_get_rpc_port(&self, vm_id: &str) -> Result<i32> {
        let vm = self.ensure_vm_running(vm_id).await?;
        let rpc_port = vm
            .network
            .and_then(|network| network.portproxy_ports)
            .map(|ports| ports.rpc_port)
            .filter(|port| *port > 0)
            .ok_or_else(|| SandboxError::InvalidResponse("VM missing rpc port".into()))?;

        self.ensure_portproxy_ready(vm_id, rpc_port).await?;
        Ok(rpc_port)
    }

    async fn ensure_portproxy_ready(&self, vm_id: &str, rpc_port: i32) -> Result<()> {
        {
            let ready = self.inner.ready_vm_rpc.lock().await;
            if ready.get(vm_id).copied() == Some(rpc_port) {
                return Ok(());
            }
        }

        let endpoint = format!("http://127.0.0.1:{rpc_port}");
        let start = Instant::now();
        let mut consecutive_successes = 0u8;

        while start.elapsed() < self.inner.cfg.portproxy_ready_timeout {
            if let Ok(mut client) = ShellExecClient::connect(endpoint.clone()).await {
                let (req_tx, req_rx) = mpsc::channel(2);
                if req_tx
                    .send(ExecRequest {
                        request: Some(exec_request::Request::Start(ExecStart {
                            args: vec!["sh".to_string(), "-lc".to_string(), "true".to_string()],
                            env: HashMap::new(),
                            detach: false,
                            timeout: Some(5),
                        })),
                    })
                    .await
                    .is_err()
                {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    continue;
                }
                drop(req_tx);

                let probe = client.exec(Request::new(ReceiverStream::new(req_rx))).await;

                match probe {
                    Ok(response) => {
                        let mut stream = response.into_inner();
                        let mut saw_transport_error = false;
                        let mut saw_any_frame = false;
                        let mut saw_exit_code = false;
                        while let Some(frame) = stream.message().await.transpose() {
                            match frame {
                                Ok(ExecResponse {
                                    response: Some(exec_response::Response::ExitCode(_)),
                                }) => {
                                    saw_any_frame = true;
                                    saw_exit_code = true;
                                    break;
                                }
                                Ok(_) => {
                                    saw_any_frame = true;
                                }
                                Err(status) => {
                                    let status_message = status.message().to_lowercase();
                                    let is_transport_not_ready =
                                        status.code() == tonic::Code::Unavailable
                                            || status_message.contains("transport error")
                                            || status_message.contains("connection reset")
                                            || status_message.contains("broken pipe")
                                            || status_message.contains("connection refused");
                                    if is_transport_not_ready {
                                        saw_transport_error = true;
                                    }
                                    break;
                                }
                            }
                        }

                        if saw_transport_error {
                            consecutive_successes = 0;
                            tokio::time::sleep(Duration::from_millis(250)).await;
                            continue;
                        }

                        if !saw_any_frame || !saw_exit_code {
                            consecutive_successes = 0;
                            tokio::time::sleep(Duration::from_millis(250)).await;
                            continue;
                        }

                        consecutive_successes = consecutive_successes.saturating_add(1);
                        if consecutive_successes < 2 {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                            continue;
                        }

                        let mut ready = self.inner.ready_vm_rpc.lock().await;
                        ready.insert(vm_id.to_string(), rpc_port);
                        return Ok(());
                    }
                    Err(status) => {
                        let status_message = status.message().to_lowercase();
                        let is_transport_not_ready = status.code() == tonic::Code::Unavailable
                            || status_message.contains("transport error")
                            || status_message.contains("connection reset")
                            || status_message.contains("broken pipe")
                            || status_message.contains("connection refused");

                        if is_transport_not_ready {
                            // Guest RPC endpoint is not fully up yet.
                            consecutive_successes = 0;
                            continue;
                        }

                        // Any non-transport gRPC response proves service readiness.
                        consecutive_successes = consecutive_successes.saturating_add(1);
                        if consecutive_successes < 2 {
                            tokio::time::sleep(Duration::from_millis(200)).await;
                            continue;
                        }
                        let mut ready = self.inner.ready_vm_rpc.lock().await;
                        ready.insert(vm_id.to_string(), rpc_port);
                        return Ok(());
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        Err(SandboxError::DaemonUnavailable(format!(
            "sandbox guest RPC did not become ready for vm {vm_id} on {endpoint}"
        )))
    }

    async fn portproxy_client_for_vm(
        &self,
        vm_id: &str,
    ) -> Result<PortProxyClient<tonic::transport::Channel>> {
        let rpc_port = self.ensure_vm_and_get_rpc_port(vm_id).await?;
        let endpoint = format!("http://127.0.0.1:{rpc_port}");
        Ok(PortProxyClient::connect(endpoint).await?)
    }

    async fn discard_vm(&self, vm_id: &str) -> Result<()> {
        let mut client = self.vmd_client().await?;
        let _ = client
            .stop_vm(Request::new(VmActionRequest {
                vm_id: vm_id.to_string(),
            }))
            .await;
        client
            .delete_vm(Request::new(proto::vmd::v1::DeleteVmRequest {
                vm_id: vm_id.to_string(),
                purge_snapshots: true,
            }))
            .await?;
        let mut ready = self.inner.ready_vm_rpc.lock().await;
        ready.remove(vm_id);
        Ok(())
    }
}

impl Drop for SandboxInner {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.managed_daemon.try_lock() {
            if let Some(managed) = guard.as_mut() {
                let _ = managed.child.start_kill();
            }
            *guard = None;
        }
    }
}

fn normalize_endpoint(raw: &str) -> Result<String> {
    let value = raw.trim();
    if value.is_empty() {
        return Err(SandboxError::InvalidEndpoint(
            "endpoint must not be empty".to_string(),
        ));
    }
    if value.contains("://") {
        return Ok(value.to_string());
    }
    Ok(format!("http://{value}"))
}

pub type ResonSandboxResult<T> = Result<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_endpoint_adds_http_scheme() {
        let value = normalize_endpoint("127.0.0.1:8052").expect("endpoint should normalize");
        assert_eq!(value, "http://127.0.0.1:8052");
    }

    #[test]
    fn normalize_endpoint_keeps_scheme() {
        let value =
            normalize_endpoint("http://127.0.0.1:8052").expect("endpoint should stay unchanged");
        assert_eq!(value, "http://127.0.0.1:8052");
    }
}
