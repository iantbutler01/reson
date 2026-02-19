// @dive-file: Public sandbox facade exposing local/dynamic and distributed control flows for session, exec/shell, and fork operations.
// @dive-rel: Delegates distributed placement, admission, and routing semantics to crates/reson-sandbox/src/distributed.rs.
// @dive-rel: Preserves locked consumer API while layering HA/distributed policy and node-level multiplexer behavior.
#![allow(clippy::large_enum_variant)]

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::PathBuf;
use std::pin::Pin;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::Stream;
#[cfg(feature = "distributed-control")]
use serde_json::json;
use tokio::io::AsyncWriteExt;
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::transport::{Certificate, ClientTlsConfig, Endpoint, Identity};
use uuid::Uuid;

#[cfg(feature = "distributed-control")]
mod distributed;
pub mod slo;

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
    CreateVmRequest, ForkVmRequest, GetVmRequest, ListVMsRequest, Metadata, ResourceSpec, Vm,
    PreDownloadVmImageRequest, RestoreSnapshotRequest, VmActionRequest, VmSource, VmSourceType,
};

const META_SESSION_ID: &str = "reson.session_id";
const META_BRANCH_ID: &str = "reson.branch_id";
const META_PARENT_SESSION_ID: &str = "reson.parent_session_id";
const META_PARENT_VM_ID: &str = "reson.parent_vm_id";
const META_FORK_ID: &str = "reson.fork_id";
const META_FORK_SNAPSHOT: &str = "reson.fork_snapshot";
const META_EXEC_RESTORE_SNAPSHOT_ID: &str = "reson.execution_restore_snapshot_id";
const META_EXEC_RESTORE_SNAPSHOT_NAME: &str = "reson.execution_restore_snapshot_name";
const META_TIER_B_ELIGIBLE: &str = "reson.tier_b_eligible";
const META_EXECUTION_FIDELITY_REQUIREMENT: &str = "reson.execution_fidelity_requirement";

#[derive(Clone, Debug)]
pub struct DistributedControlConfig {
    pub etcd_endpoints: Vec<String>,
    pub etcd_prefix: String,
    pub nats_url: String,
    pub nats_subject_prefix: String,
    pub nats_stream_name: String,
    pub nats_stream_max_age_secs: u64,
    pub nats_stream_replicas: usize,
    pub nats_dead_letter_subject: String,
    pub required_storage_profile: Option<String>,
    pub required_continuity_tier: Option<String>,
    pub allow_tier_a_degraded: bool,
    pub tenant_session_quota: Option<usize>,
    pub workspace_session_quota: Option<usize>,
    pub admission_retry_after_ms: u64,
}

impl Default for DistributedControlConfig {
    fn default() -> Self {
        let subject_prefix = "reson.sandbox.control".to_string();
        Self {
            etcd_endpoints: vec!["http://127.0.0.1:2379".to_string()],
            etcd_prefix: "/reson-sandbox".to_string(),
            nats_url: "nats://127.0.0.1:4222".to_string(),
            nats_subject_prefix: subject_prefix.clone(),
            nats_stream_name: "RESON_SANDBOX_CONTROL".to_string(),
            nats_stream_max_age_secs: 60 * 60 * 24 * 7,
            nats_stream_replicas: 1,
            nats_dead_letter_subject: format!("{subject_prefix}.dlq.commands"),
            required_storage_profile: None,
            required_continuity_tier: Some("tier-b".to_string()),
            allow_tier_a_degraded: false,
            tenant_session_quota: Some(256),
            workspace_session_quota: Some(64),
            admission_retry_after_ms: 2_000,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TlsClientConfig {
    pub ca_cert_path: Option<PathBuf>,
    pub client_cert_path: Option<PathBuf>,
    pub client_key_path: Option<PathBuf>,
    pub domain_name: Option<String>,
}

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
    #[error("resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error("ownership fence conflict: {0}")]
    FenceConflict(String),
    #[error("unsupported operation: {0}")]
    Unsupported(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
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
pub struct WarmPoolProfile {
    pub image: String,
    pub architecture: Option<String>,
    pub min_inventory: usize,
}

impl WarmPoolProfile {
    fn normalized_architecture(&self) -> Option<String> {
        self.architecture
            .as_deref()
            .map(normalize_architecture_label)
            .filter(|value| !value.is_empty())
    }
}

#[derive(Clone, Debug)]
pub struct SandboxConfig {
    pub endpoint: String,
    pub control_gateway_endpoints: Vec<String>,
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
    pub warm_pool_profiles: Vec<WarmPoolProfile>,
    pub prewarm_on_start: bool,
    pub distributed_control: Option<DistributedControlConfig>,
    pub auth_token: Option<String>,
    pub tls: Option<TlsClientConfig>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            endpoint: "http://127.0.0.1:8052".to_string(),
            control_gateway_endpoints: Vec::new(),
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
            warm_pool_profiles: Vec::new(),
            prewarm_on_start: true,
            distributed_control: None,
            auth_token: None,
            tls: None,
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

#[cfg(feature = "distributed-control")]
#[derive(Clone)]
struct PortLifecycleContext {
    sandbox: Sandbox,
    session_id: String,
    vm_id: String,
    node_endpoint: String,
}

struct ForwardRegistration {
    multiplexer: Arc<NodePortMultiplexer>,
    host_port: u16,
    #[cfg(feature = "distributed-control")]
    port_lease: Option<distributed::PortAllocationLease>,
}

struct ForwardTask {
    shutdown_tx: oneshot::Sender<()>,
    join: tokio::task::JoinHandle<()>,
}

#[derive(Default)]
struct NodePortMultiplexer {
    forwards: Mutex<HashMap<u16, ForwardTask>>,
}

impl NodePortMultiplexer {
    async fn register(&self, guest_port: u16, server_addr: String) -> Result<u16> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let host_port = listener.local_addr()?.port();
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let join = tokio::spawn(async move {
            run_forward_listener(listener, guest_port, server_addr, shutdown_rx).await;
        });

        let mut guard = self.forwards.lock().await;
        guard.insert(host_port, ForwardTask { shutdown_tx, join });
        Ok(host_port)
    }

    async fn unregister(&self, host_port: u16) {
        let task = {
            let mut guard = self.forwards.lock().await;
            guard.remove(&host_port)
        };

        if let Some(task) = task {
            let _ = task.shutdown_tx.send(());
            let _ = task.join.await;
        }
    }
}

pub struct ForwardHandle {
    pub guest_port: u16,
    pub host_port: u16,
    registration: Arc<Mutex<Option<ForwardRegistration>>>,
    #[cfg(feature = "distributed-control")]
    port_context: Option<PortLifecycleContext>,
}

impl ForwardHandle {
    pub async fn close(&self) -> Result<()> {
        let registration = {
            let mut guard = self.registration.lock().await;
            guard.take()
        };
        let released = registration.is_some();

        if let Some(mut registration) = registration {
            registration
                .multiplexer
                .unregister(registration.host_port)
                .await;
            #[cfg(feature = "distributed-control")]
            if let Some(port_lease) = registration.port_lease.take() {
                port_lease.shutdown().await;
            }
        }

        #[cfg(feature = "distributed-control")]
        if released {
            self.publish_release().await;
        }
        #[cfg(not(feature = "distributed-control"))]
        let _ = released;

        Ok(())
    }
}

impl Drop for ForwardHandle {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.registration.try_lock() {
            let registration = guard.take();
            drop(guard);
            if let Some(mut registration) = registration {
                if let Ok(handle) = tokio::runtime::Handle::try_current() {
                    handle.spawn(async move {
                        registration
                            .multiplexer
                            .unregister(registration.host_port)
                            .await;
                        #[cfg(feature = "distributed-control")]
                        if let Some(port_lease) = registration.port_lease.take() {
                            port_lease.shutdown().await;
                        }
                    });
                }
            }
        }
    }
}

async fn run_forward_listener(
    listener: TcpListener,
    guest_port: u16,
    server_addr: String,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        tokio::select! {
            _ = &mut shutdown_rx => break,
            accept_result = listener.accept() => {
                let Ok((socket, _peer)) = accept_result else {
                    break;
                };
                let server_addr = server_addr.clone();
                tokio::spawn(async move {
                    let _ = forward_client_connection(socket, guest_port, &server_addr).await;
                });
            }
        }
    }
}

async fn forward_client_connection(
    mut socket: TcpStream,
    forward_port: u16,
    server_addr: &str,
) -> std::io::Result<()> {
    let mut server = TcpStream::connect(server_addr).await?;
    server.write_all(&forward_port.to_be_bytes()).await?;
    let _ = tokio::io::copy_bidirectional(&mut socket, &mut server).await?;
    Ok(())
}

impl ForwardHandle {
    #[cfg(feature = "distributed-control")]
    async fn publish_release(&self) {
        if let Some(ctx) = &self.port_context {
            let _ = ctx
                .sandbox
                .publish_control_command(
                    "port.release",
                    &ctx.vm_id,
                    json!({
                        "session_id": ctx.session_id.as_str(),
                        "vm_id": ctx.vm_id.as_str(),
                        "endpoint": ctx.node_endpoint.as_str(),
                        "guest_port": self.guest_port,
                        "host_port": self.host_port,
                    }),
                )
                .await;
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
    control_backend: ControlBackend,
    auth_header: Option<MetadataValue<Ascii>>,
    managed_daemon: Mutex<Option<ManagedDaemon>>,
    ready_vm_rpc: Mutex<HashMap<String, i32>>,
    node_multiplexers: Mutex<HashMap<String, Arc<NodePortMultiplexer>>>,
    warm_pool_ready: Mutex<HashSet<String>>,
}

enum ControlBackend {
    Direct,
    #[cfg(feature = "distributed-control")]
    Distributed(distributed::DistributedControlPlane),
}

#[derive(Clone)]
pub struct Session {
    sandbox: Sandbox,
    session_id: String,
    vm_id: String,
    node_endpoint: Arc<Mutex<String>>,
    ownership_fence: Arc<Mutex<Option<String>>>,
}

impl Session {
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn vm_id(&self) -> &str {
        &self.vm_id
    }

    async fn ownership_fence(&self) -> Option<String> {
        self.ownership_fence.lock().await.clone()
    }

    async fn current_node_endpoint(&self) -> String {
        self.node_endpoint.lock().await.clone()
    }

    async fn update_route_state(
        &self,
        previous_endpoint: &str,
        resolved_endpoint: &str,
        next_fence: Option<String>,
    ) {
        if previous_endpoint != resolved_endpoint {
            let mut guard = self.node_endpoint.lock().await;
            *guard = resolved_endpoint.to_string();
        }
        if let Some(fence) = next_fence {
            let mut guard = self.ownership_fence.lock().await;
            *guard = Some(fence);
        }
    }

    async fn resolve_session_endpoint(&self) -> Result<String> {
        let current_endpoint = self.current_node_endpoint().await;
        let expected_fence = self.ownership_fence().await;
        let (resolved_endpoint, next_fence) = self
            .sandbox
            .resolve_session_endpoint(
                &self.session_id,
                &self.vm_id,
                &current_endpoint,
                expected_fence.as_deref(),
            )
            .await?;
        self.update_route_state(&current_endpoint, &resolved_endpoint, next_fence)
            .await;
        Ok(resolved_endpoint)
    }

    async fn ensure_session_rpc_endpoint(&self) -> Result<String> {
        let current_endpoint = self.current_node_endpoint().await;
        let expected_fence = self.ownership_fence().await;
        let (resolved_endpoint, rpc_port, next_fence) = self
            .sandbox
            .ensure_vm_and_get_rpc_port_for_session(
                &self.session_id,
                &self.vm_id,
                &current_endpoint,
                expected_fence.as_deref(),
            )
            .await?;
        self.update_route_state(&current_endpoint, &resolved_endpoint, next_fence)
            .await;
        Ok(format!("http://127.0.0.1:{rpc_port}"))
    }

    pub async fn exec(&self, command: &str, opts: ExecOptions) -> Result<ExecHandle> {
        let started = Instant::now();
        let endpoint = self.ensure_session_rpc_endpoint().await?;
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
                        if event_tx_stream
                            .send(Ok(ExecEvent::Stdout(data)))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(Some(ExecResponse {
                        response: Some(exec_response::Response::StderrData(data)),
                    })) => {
                        if event_tx_stream
                            .send(Ok(ExecEvent::Stderr(data)))
                            .await
                            .is_err()
                        {
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

        let handle = ExecHandle {
            input: input_tx,
            events: Box::pin(ReceiverStream::new(event_rx)),
        };
        log_slo_observation("exec.stream.establish.warm_vm", started.elapsed(), "ok");
        Ok(handle)
    }

    pub async fn shell(&self, opts: ShellOptions) -> Result<ShellHandle> {
        let started = Instant::now();
        let endpoint = self.ensure_session_rpc_endpoint().await?;
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
                        if event_tx_stream
                            .send(Ok(ShellEvent::Output(data)))
                            .await
                            .is_err()
                        {
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

        let handle = ShellHandle {
            input: input_tx,
            events: Box::pin(ReceiverStream::new(event_rx)),
        };
        log_slo_observation("shell.stream.establish.warm_vm", started.elapsed(), "ok");
        Ok(handle)
    }

    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let endpoint = self.resolve_session_endpoint().await?;
        let mut client = self
            .sandbox
            .portproxy_client_for_vm(&self.vm_id, &endpoint)
            .await?;
        let response = client
            .read_file(Request::new(ReadFileRequest {
                path: path.to_string(),
            }))
            .await?
            .into_inner();
        Ok(response.data)
    }

    pub async fn write_file(&self, path: &str, data: Vec<u8>) -> Result<()> {
        let endpoint = self.resolve_session_endpoint().await?;
        let mut client = self
            .sandbox
            .portproxy_client_for_vm(&self.vm_id, &endpoint)
            .await?;
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
        let endpoint = self.resolve_session_endpoint().await?;
        let mut client = self
            .sandbox
            .portproxy_client_for_vm(&self.vm_id, &endpoint)
            .await?;
        let response = client
            .list_directory(Request::new(ListDirectoryRequest {
                path: path.to_string(),
            }))
            .await?
            .into_inner();
        Ok(response.entries)
    }

    pub async fn delete_path(&self, path: &str) -> Result<()> {
        let endpoint = self.resolve_session_endpoint().await?;
        let mut client = self
            .sandbox
            .portproxy_client_for_vm(&self.vm_id, &endpoint)
            .await?;
        client
            .delete_path(Request::new(DeletePathRequest {
                path: path.to_string(),
            }))
            .await?;
        Ok(())
    }

    pub async fn forward_port(&self, guest_port: u16) -> Result<ForwardHandle> {
        let started = Instant::now();
        let node_endpoint = self.resolve_session_endpoint().await?;
        let vm = self
            .sandbox
            .ensure_vm_running(&self.vm_id, &node_endpoint)
            .await?;
        let proxy_port = vm
            .network
            .and_then(|n| n.portproxy_ports)
            .map(|p| p.proxy_port)
            .ok_or_else(|| SandboxError::InvalidResponse("VM missing proxy port".into()))?;
        let proxy_port = u16::try_from(proxy_port).map_err(|_| {
            SandboxError::InvalidResponse(format!(
                "VM reported invalid proxy port for forwarding: {proxy_port}"
            ))
        })?;
        if proxy_port == 0 {
            return Err(SandboxError::InvalidResponse(
                "VM reported zero proxy port for forwarding".into(),
            ));
        }
        let proxy_addr = portproxy_server_addr(&node_endpoint, proxy_port)?;
        let multiplexer = self
            .sandbox
            .node_multiplexer_for_endpoint(&node_endpoint)
            .await;
        let host_port = multiplexer.register(guest_port, proxy_addr).await?;

        #[cfg(feature = "distributed-control")]
        let port_lease =
            if let ControlBackend::Distributed(control) = &self.sandbox.inner.control_backend {
                Some(
                    control
                        .acquire_port_lease(distributed::PortAllocation {
                            session_id: self.session_id.clone(),
                            vm_id: self.vm_id.clone(),
                            endpoint: node_endpoint.clone(),
                            guest_port,
                            host_port,
                        })
                        .await?,
                )
            } else {
                None
            };

        #[cfg(feature = "distributed-control")]
        self.sandbox
            .publish_control_command(
                "port.alloc",
                &self.vm_id,
                json!({
                    "session_id": self.session_id.as_str(),
                    "vm_id": self.vm_id.as_str(),
                    "endpoint": node_endpoint.as_str(),
                    "guest_port": guest_port,
                    "host_port": host_port,
                }),
            )
            .await?;

        let handle = ForwardHandle {
            guest_port,
            host_port,
            registration: Arc::new(Mutex::new(Some(ForwardRegistration {
                multiplexer,
                host_port,
                #[cfg(feature = "distributed-control")]
                port_lease,
            }))),
            #[cfg(feature = "distributed-control")]
            port_context: Some(PortLifecycleContext {
                sandbox: self.sandbox.clone(),
                session_id: self.session_id.clone(),
                vm_id: self.vm_id.clone(),
                node_endpoint,
            }),
        };
        log_slo_observation("port.forward.establish", started.elapsed(), "ok");
        Ok(handle)
    }

    pub async fn fork(&self, opts: ForkOptions) -> Result<ForkResult> {
        let auto_start_child = opts.auto_start_child;
        let node_endpoint = self.resolve_session_endpoint().await?;
        let child_session_id = Uuid::new_v4().to_string();
        let ownership_fence = self.ownership_fence().await;
        let mut child_metadata = opts.child_metadata;
        child_metadata.insert(META_SESSION_ID.to_string(), child_session_id.clone());
        child_metadata.insert(META_PARENT_SESSION_ID.to_string(), self.session_id.clone());
        child_metadata.insert(META_PARENT_VM_ID.to_string(), self.vm_id.clone());
        child_metadata.insert(META_BRANCH_ID.to_string(), child_session_id.clone());

        #[cfg(feature = "distributed-control")]
        self.sandbox
            .publish_control_command(
                "vm.fork",
                &self.vm_id,
                json!({
                    "session_id": self.session_id.as_str(),
                    "vm_id": self.vm_id.as_str(),
                    "child_session_id": child_session_id.as_str(),
                    "endpoint": node_endpoint.as_str(),
                    "auto_start_child": auto_start_child,
                    "expected_fence": ownership_fence.as_deref(),
                }),
            )
            .await?;

        let mut client = self
            .sandbox
            .vmd_client_for_endpoint(&node_endpoint)
            .await?;
        let response = client
            .fork_vm(self.sandbox.request_with_auth(ForkVmRequest {
                parent_vm_id: self.vm_id.clone(),
                child_name: opts.child_name.unwrap_or_default(),
                child_metadata: Some(Metadata {
                    entries: child_metadata,
                }),
                auto_start_child,
            }))
            .await?
            .into_inner();

        let child_vm = response.child_vm.ok_or_else(|| {
            SandboxError::InvalidResponse("fork response missing child VM".into())
        })?;

        if auto_start_child {
            let _ = self
                .sandbox
                .ensure_vm_and_get_rpc_port(&child_vm.id, &node_endpoint)
                .await?;
        }
        let (tenant_id, workspace_id) = self
            .sandbox
            .current_session_scope(&self.session_id)
            .await?;

        let child_fence = self
            .sandbox
            .bind_session_route(
                &child_session_id,
                &child_vm.id,
                &node_endpoint,
                Some(response.fork_id.as_str()),
                Some(tenant_id.as_str()),
                Some(workspace_id.as_str()),
                None,
            )
            .await?;

        let child = Session {
            sandbox: self.sandbox.clone(),
            session_id: child_session_id.clone(),
            vm_id: child_vm.id,
            node_endpoint: Arc::new(Mutex::new(node_endpoint)),
            ownership_fence: Arc::new(Mutex::new(child_fence)),
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
        let ownership_fence = self.ownership_fence().await;
        let node_endpoint = self.current_node_endpoint().await;
        self.sandbox
            .discard_vm(
                &self.vm_id,
                &node_endpoint,
                Some(&self.session_id),
                ownership_fence.as_deref(),
            )
            .await
    }
}

impl Sandbox {
    pub async fn new(mut config: SandboxConfig) -> Result<Self> {
        config.endpoint = normalize_endpoint(&config.endpoint)?;
        let mut normalized_gateways = Vec::new();
        for endpoint in std::mem::take(&mut config.control_gateway_endpoints) {
            if endpoint.trim().is_empty() {
                continue;
            }
            normalized_gateways.push(normalize_endpoint(&endpoint)?);
        }
        normalized_gateways.sort();
        normalized_gateways.dedup();
        normalized_gateways.retain(|endpoint| endpoint != &config.endpoint);
        config.control_gateway_endpoints = normalized_gateways;
        let auth_header = compile_auth_header(config.auth_token.as_deref())?;

        let control_backend = Self::build_control_backend(&config).await?;

        let sandbox = Self {
            inner: Arc::new(SandboxInner {
                cfg: config,
                control_backend,
                auth_header,
                managed_daemon: Mutex::new(None),
                ready_vm_rpc: Mutex::new(HashMap::new()),
                node_multiplexers: Mutex::new(HashMap::new()),
                warm_pool_ready: Mutex::new(HashSet::new()),
            }),
        };

        sandbox.ensure_daemon_ready().await?;
        sandbox.prewarm_warm_pool_profiles().await?;
        Ok(sandbox)
    }

    pub async fn connect(endpoint: impl Into<String>, mut config: SandboxConfig) -> Result<Self> {
        config.endpoint = normalize_endpoint(&endpoint.into())?;
        config.auto_spawn = false;
        Self::new(config).await
    }

    pub async fn session(&self, opts: SessionOptions) -> Result<Session> {
        let started = Instant::now();
        let session_id = opts
            .session_id
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let auto_start = opts.auto_start;

        let mut metadata = opts.metadata;
        let tenant_id = metadata
            .get("tenant_id")
            .cloned()
            .unwrap_or_else(|| "default".to_string());
        let workspace_id = metadata
            .get("workspace_id")
            .cloned()
            .unwrap_or_else(|| "default".to_string());
        metadata
            .entry("tenant_id".to_string())
            .or_insert_with(|| tenant_id.clone());
        metadata
            .entry("workspace_id".to_string())
            .or_insert_with(|| workspace_id.clone());
        let tier_b_eligible = resolve_tier_b_eligibility(&metadata);
        metadata.insert(
            META_TIER_B_ELIGIBLE.to_string(),
            if tier_b_eligible { "true" } else { "false" }.to_string(),
        );
        metadata.insert(
            META_EXECUTION_FIDELITY_REQUIREMENT.to_string(),
            if tier_b_eligible {
                "disk+memory".to_string()
            } else {
                "best-effort".to_string()
            },
        );

        if let Some((vm, node_endpoint)) = self.find_vm_by_session_id(&session_id).await? {
            let expected_fence = self.current_session_fence(&session_id).await?;
            let (attached_tenant_id, attached_workspace_id) =
                self.current_session_scope(&session_id).await?;
            #[cfg(feature = "distributed-control")]
            self.publish_control_command(
                "session.attach",
                &session_id,
                json!({
                    "session_id": session_id.as_str(),
                    "vm_id": vm.id.as_str(),
                    "endpoint": node_endpoint.as_str(),
                    "tenant_id": attached_tenant_id.as_str(),
                    "workspace_id": attached_workspace_id.as_str(),
                    "expected_fence": expected_fence.as_deref(),
                }),
            )
            .await?;

            // @dive: Reattach path replays execution snapshot restore hints before booting to preserve Tier-B state fidelity.
            let _ = self
                .restore_execution_state_if_needed(
                    &session_id,
                    &vm.id,
                    &node_endpoint,
                    &vm,
                    false,
                )
                .await?;
            let vm = self.ensure_vm_running(&vm.id, &node_endpoint).await?;
            let _ = self
                .ensure_vm_and_get_rpc_port(&vm.id, &node_endpoint)
                .await?;
            let next_fence = self
                .bind_session_route(
                    &session_id,
                    &vm.id,
                    &node_endpoint,
                    None,
                    Some(attached_tenant_id.as_str()),
                    Some(attached_workspace_id.as_str()),
                    expected_fence.as_deref(),
                )
                .await?;
            let session = Session {
                sandbox: self.clone(),
                session_id,
                vm_id: vm.id,
                node_endpoint: Arc::new(Mutex::new(node_endpoint)),
                ownership_fence: Arc::new(Mutex::new(next_fence)),
            };
            log_slo_observation("session.attach", started.elapsed(), "ok");
            return Ok(session);
        }

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
        let architecture = opts.architecture.unwrap_or_else(|| {
            self.inner
                .cfg
                .default_architecture
                .clone()
                .or_else(detect_host_architecture_label)
                .unwrap_or_default()
        });

        let request = CreateVmRequest {
            name,
            source: Some(VmSource {
                r#type: VmSourceType::Docker as i32,
                reference: image.clone(),
            }),
            resources: Some(ResourceSpec {
                vcpu: resources.vcpu,
                memory_mb: resources.memory_mb,
                disk_gb: resources.disk_gb,
            }),
            metadata: Some(Metadata { entries: metadata }),
            auto_start: opts.auto_start,
            architecture: architecture.clone(),
        };

        let node_endpoint = self
            .endpoint_for_new_session(&session_id, &tenant_id, &workspace_id, tier_b_eligible)
            .await?;
        let warm_pool_key = warm_pool_key(&node_endpoint, image.as_str(), architecture.as_str());
        let warm_pool_hit = self.warm_pool_contains_key(warm_pool_key.as_str()).await;
        #[cfg(feature = "distributed-control")]
        self.publish_control_command(
            "session.create",
            &session_id,
            json!({
                "session_id": session_id.as_str(),
                "endpoint": node_endpoint.as_str(),
                "tenant_id": tenant_id.as_str(),
                "workspace_id": workspace_id.as_str(),
                "auto_start": auto_start,
                "tier_b_eligible": tier_b_eligible,
                "execution_fidelity_requirement": if tier_b_eligible { "disk+memory" } else { "best-effort" },
                "warm_pool_hit": warm_pool_hit,
                "architecture": architecture.as_str(),
            }),
        )
        .await?;

        let mut client = self.vmd_client_for_endpoint(&node_endpoint).await?;
        let mut stream = client
            .create_vm(self.request_with_auth(request))
            .await?
            .into_inner();

        let mut final_vm: Option<Vm> = None;
        while let Some(update) = stream.message().await? {
            if let Some(proto::vmd::v1::create_vm_stream_response::Event::Vm(vm)) = update.event {
                final_vm = Some(vm);
            }
        }

        let vm = final_vm
            .ok_or_else(|| SandboxError::InvalidResponse("create_vm stream missing VM".into()))?;

        let next_fence = self
            .bind_session_route(
                &session_id,
                &vm.id,
                &node_endpoint,
                None,
                Some(tenant_id.as_str()),
                Some(workspace_id.as_str()),
                None,
            )
            .await?;

        let running_state = proto::vmd::v1::VmState::Running as i32;
        if auto_start || vm.state == running_state {
            let _ = self
                .ensure_vm_and_get_rpc_port(&vm.id, &node_endpoint)
                .await?;
        }

        let session = Session {
            sandbox: self.clone(),
            session_id,
            vm_id: vm.id,
            node_endpoint: Arc::new(Mutex::new(node_endpoint)),
            ownership_fence: Arc::new(Mutex::new(next_fence)),
        };

        if warm_pool_hit {
            log_slo_observation("session.create.warm_pool", started.elapsed(), "ok");
        } else {
            log_slo_observation("session.create.cold_cache_hit", started.elapsed(), "ok");
            let sandbox = self.clone();
            let endpoint = session.current_node_endpoint().await;
            let refill_profile = WarmPoolProfile {
                image,
                architecture: Some(architecture),
                min_inventory: 1,
            };
            tokio::spawn(async move {
                let profiles = vec![refill_profile];
                let _ = sandbox
                    .prewarm_profiles_on_endpoint(endpoint.as_str(), &profiles)
                    .await;
            });
        }

        log_slo_observation("session.create", started.elapsed(), "ok");
        Ok(session)
    }

    pub async fn attach_session(&self, session_id: &str) -> Result<Session> {
        let started = Instant::now();
        let (vm, node_endpoint) = self
            .find_vm_by_session_id(session_id)
            .await?
            .ok_or_else(|| SandboxError::SessionNotFound(session_id.to_string()))?;
        let expected_fence = self.current_session_fence(session_id).await?;
        let (tenant_id, workspace_id) = self.current_session_scope(session_id).await?;

        #[cfg(feature = "distributed-control")]
        self.publish_control_command(
            "session.attach",
            session_id,
            json!({
                "session_id": session_id,
                "vm_id": vm.id.as_str(),
                "endpoint": node_endpoint.as_str(),
                "tenant_id": tenant_id.as_str(),
                "workspace_id": workspace_id.as_str(),
                "expected_fence": expected_fence.as_deref(),
            }),
        )
        .await?;

        // @dive: Attach uses the same restore primitive as failover rebinding so execution-state recovery is deterministic.
        let _ = self
            .restore_execution_state_if_needed(session_id, &vm.id, &node_endpoint, &vm, false)
            .await?;
        let vm = self.ensure_vm_running(&vm.id, &node_endpoint).await?;
        let _ = self
            .ensure_vm_and_get_rpc_port(&vm.id, &node_endpoint)
            .await?;
        let next_fence = self
            .bind_session_route(
                session_id,
                &vm.id,
                &node_endpoint,
                None,
                Some(tenant_id.as_str()),
                Some(workspace_id.as_str()),
                expected_fence.as_deref(),
            )
            .await?;

        let session = Session {
            sandbox: self.clone(),
            session_id: session_id.to_string(),
            vm_id: vm.id,
            node_endpoint: Arc::new(Mutex::new(node_endpoint)),
            ownership_fence: Arc::new(Mutex::new(next_fence)),
        };
        log_slo_observation("session.attach", started.elapsed(), "ok");
        Ok(session)
    }

    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        let mut sessions = Vec::new();
        let mut seen = HashSet::new();
        for endpoint in self.candidate_endpoints().await? {
            let mut client = match self.vmd_client_for_endpoint(&endpoint).await {
                Ok(client) => client,
                Err(_) => continue,
            };
            let response = match client
                .list_v_ms(self.request_with_auth(ListVMsRequest {
                    include_snapshots: false,
                }))
                .await
            {
                Ok(response) => response.into_inner(),
                Err(_) => continue,
            };
            for vm in response.vms {
                let Some(session_id) = vm.metadata.get(META_SESSION_ID).cloned() else {
                    continue;
                };
                if !seen.insert(session_id.clone()) {
                    continue;
                }
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
        }

        Ok(sessions)
    }

    async fn build_control_backend(config: &SandboxConfig) -> Result<ControlBackend> {
        if let Some(dist_cfg) = config.distributed_control.clone() {
            #[cfg(feature = "distributed-control")]
            {
                let control = distributed::DistributedControlPlane::connect(dist_cfg).await?;
                return Ok(ControlBackend::Distributed(control));
            }

            #[cfg(not(feature = "distributed-control"))]
            {
                let _ = dist_cfg;
                return Err(SandboxError::Unsupported(
                    "distributed control requested but crate feature `distributed-control` is disabled"
                        .to_string(),
                ));
            }
        }

        Ok(ControlBackend::Direct)
    }

    async fn vmd_client_for_endpoint(
        &self,
        endpoint_raw: &str,
    ) -> Result<VmdServiceClient<tonic::transport::Channel>> {
        let endpoint_raw = normalize_endpoint(endpoint_raw)?;
        let mut endpoint = Endpoint::from_shared(endpoint_raw.clone())
            .map_err(|err| SandboxError::InvalidEndpoint(err.to_string()))?
            .connect_timeout(self.inner.cfg.connect_timeout)
            .timeout(self.inner.cfg.connect_timeout);
        if endpoint_raw.starts_with("https://") {
            endpoint = endpoint
                .tls_config(self.build_client_tls_config(endpoint_raw.as_str())?)
                .map_err(|err| SandboxError::InvalidConfig(err.to_string()))?;
        }
        Ok(VmdServiceClient::connect(endpoint).await?)
    }

    async fn ensure_daemon_ready(&self) -> Result<()> {
        match &self.inner.control_backend {
            ControlBackend::Direct => {
                for endpoint in self.candidate_endpoints().await? {
                    if self.health_check_endpoint(&endpoint).await.is_ok() {
                        return Ok(());
                    }
                }

                if !self.inner.cfg.auto_spawn {
                    return Err(SandboxError::DaemonUnavailable(format!(
                        "unable to connect to sandbox daemon at any configured control endpoint (primary: {})",
                        self.inner.cfg.endpoint
                    )));
                }

                self.spawn_daemon_if_needed().await?;

                let start = Instant::now();
                while start.elapsed() < self.inner.cfg.daemon_start_timeout {
                    if self
                        .health_check_endpoint(&self.inner.cfg.endpoint)
                        .await
                        .is_ok()
                    {
                        return Ok(());
                    }
                    tokio::time::sleep(Duration::from_millis(250)).await;
                }

                Err(SandboxError::DaemonUnavailable(format!(
                    "sandbox daemon did not become ready at {}",
                    self.inner.cfg.endpoint
                )))
            }
            #[cfg(feature = "distributed-control")]
            ControlBackend::Distributed(_) => self.ensure_distributed_ready().await,
        }
    }

    async fn prewarm_warm_pool_profiles(&self) -> Result<()> {
        if !self.inner.cfg.prewarm_on_start {
            return Ok(());
        }

        let profiles = self.resolved_warm_pool_profiles();
        if profiles.is_empty() {
            return Ok(());
        }

        for endpoint in self.candidate_endpoints().await? {
            if self.health_check_endpoint(&endpoint).await.is_err() {
                continue;
            }
            self.prewarm_profiles_on_endpoint(&endpoint, &profiles).await?;
        }
        Ok(())
    }

    fn resolved_warm_pool_profiles(&self) -> Vec<WarmPoolProfile> {
        if !self.inner.cfg.warm_pool_profiles.is_empty() {
            return self.inner.cfg.warm_pool_profiles.clone();
        }

        let architecture = self
            .inner
            .cfg
            .default_architecture
            .as_deref()
            .map(normalize_architecture_label)
            .filter(|value| !value.is_empty())
            .or_else(detect_host_architecture_label);

        vec![WarmPoolProfile {
            image: self.inner.cfg.default_image.clone(),
            architecture,
            min_inventory: 1,
        }]
    }

    async fn prewarm_profiles_on_endpoint(
        &self,
        endpoint: &str,
        profiles: &[WarmPoolProfile],
    ) -> Result<()> {
        for profile in profiles {
            if profile.image.trim().is_empty() {
                continue;
            }
            let architecture = profile
                .normalized_architecture()
                .or_else(detect_host_architecture_label)
                .unwrap_or_default();
            let key = warm_pool_key(endpoint, profile.image.as_str(), architecture.as_str());
            if self.warm_pool_contains_key(key.as_str()).await {
                continue;
            }
            if self
                .prewarm_profile_on_endpoint(endpoint, profile, architecture.as_str())
                .await
                .is_ok()
            {
                self.warm_pool_mark_ready(key).await;
            }
        }
        Ok(())
    }

    async fn prewarm_profile_on_endpoint(
        &self,
        endpoint: &str,
        profile: &WarmPoolProfile,
        architecture: &str,
    ) -> Result<()> {
        let mut client = self.vmd_client_for_endpoint(endpoint).await?;
        let mut stream = client
            .pre_download_vm_image(self.request_with_auth(PreDownloadVmImageRequest {
                reference: profile.image.clone(),
                architecture: architecture.to_string(),
                force: false,
            }))
            .await?
            .into_inner();
        while stream.message().await?.is_some() {}
        Ok(())
    }

    async fn warm_pool_contains_key(&self, key: &str) -> bool {
        self.inner.warm_pool_ready.lock().await.contains(key)
    }

    async fn warm_pool_mark_ready(&self, key: String) {
        self.inner.warm_pool_ready.lock().await.insert(key);
    }

    fn build_client_tls_config(&self, endpoint: &str) -> Result<ClientTlsConfig> {
        let mut tls = ClientTlsConfig::new();
        let mut domain_set = false;
        if let Some(cfg) = self.inner.cfg.tls.as_ref() {
            if let Some(ca_path) = cfg.ca_cert_path.as_ref() {
                let ca_pem = fs::read(ca_path).map_err(|err| {
                    SandboxError::InvalidConfig(format!(
                        "read tls ca cert {}: {err}",
                        ca_path.to_string_lossy()
                    ))
                })?;
                tls = tls.ca_certificate(Certificate::from_pem(ca_pem));
            }
            match (cfg.client_cert_path.as_ref(), cfg.client_key_path.as_ref()) {
                (Some(cert_path), Some(key_path)) => {
                    let cert_pem = fs::read(cert_path).map_err(|err| {
                        SandboxError::InvalidConfig(format!(
                            "read tls client cert {}: {err}",
                            cert_path.to_string_lossy()
                        ))
                    })?;
                    let key_pem = fs::read(key_path).map_err(|err| {
                        SandboxError::InvalidConfig(format!(
                            "read tls client key {}: {err}",
                            key_path.to_string_lossy()
                        ))
                    })?;
                    tls = tls.identity(Identity::from_pem(cert_pem, key_pem));
                }
                (None, None) => {}
                _ => {
                    return Err(SandboxError::InvalidConfig(
                        "both tls client cert and key must be configured together".to_string(),
                    ));
                }
            }
            if let Some(domain_name) = cfg.domain_name.as_ref() {
                let trimmed = domain_name.trim();
                if !trimmed.is_empty() {
                    tls = tls.domain_name(trimmed.to_string());
                    domain_set = true;
                }
            }
        }
        if !domain_set {
            let host = endpoint_host(endpoint)?;
            if !host.trim().is_empty() {
                tls = tls.domain_name(host);
            }
        }
        Ok(tls)
    }

    fn request_with_auth<T>(&self, message: T) -> Request<T> {
        let mut request = Request::new(message);
        if let Some(value) = self.inner.auth_header.as_ref() {
            request
                .metadata_mut()
                .insert("authorization", value.clone());
        }
        request
    }

    #[cfg(feature = "distributed-control")]
    async fn ensure_distributed_ready(&self) -> Result<()> {
        let start = Instant::now();
        while start.elapsed() < self.inner.cfg.daemon_start_timeout {
            for endpoint in self.candidate_endpoints().await? {
                if self.health_check_endpoint(&endpoint).await.is_ok() {
                    return Ok(());
                }
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }

        Err(SandboxError::DaemonUnavailable(
            "no healthy distributed sandbox node discovered".to_string(),
        ))
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

    async fn health_check_endpoint(&self, endpoint: &str) -> Result<()> {
        let mut client = self.vmd_client_for_endpoint(endpoint).await?;
        let _ = client
            .health(self.request_with_auth(proto::vmd::v1::HealthRequest {}))
            .await?;
        Ok(())
    }

    async fn node_multiplexer_for_endpoint(&self, endpoint: &str) -> Arc<NodePortMultiplexer> {
        let mut guard = self.inner.node_multiplexers.lock().await;
        guard
            .entry(endpoint.to_string())
            .or_insert_with(|| Arc::new(NodePortMultiplexer::default()))
            .clone()
    }

    async fn candidate_endpoints(&self) -> Result<Vec<String>> {
        let mut endpoints = Vec::new();
        endpoints.push(self.inner.cfg.endpoint.clone());
        endpoints.extend(self.inner.cfg.control_gateway_endpoints.iter().cloned());

        #[cfg(feature = "distributed-control")]
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            for node in control.list_node_routes().await? {
                endpoints.push(node.endpoint);
            }
        }

        endpoints.sort();
        endpoints.dedup();
        Ok(endpoints)
    }

    async fn endpoint_for_new_session(
        &self,
        _session_id: &str,
        _tenant_id: &str,
        _workspace_id: &str,
        _tier_b_eligible: bool,
    ) -> Result<String> {
        #[cfg(feature = "distributed-control")]
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            if let Some(endpoint) = control
                .get_session_route(_session_id)
                .await?
                .map(|route| route.endpoint)
                .filter(|endpoint| !endpoint.trim().is_empty())
            {
                return normalize_endpoint(&endpoint);
            }
            let node = control
                .select_node_for_session_with_eligibility(
                    _session_id,
                    _tenant_id,
                    _workspace_id,
                    _tier_b_eligible,
                )
                .await?;
            return normalize_endpoint(&node.endpoint);
        }

        let _ = _tier_b_eligible;
        for endpoint in self.candidate_endpoints().await? {
            if self.health_check_endpoint(&endpoint).await.is_ok() {
                return Ok(endpoint);
            }
        }

        Ok(self.inner.cfg.endpoint.clone())
    }

    #[cfg(feature = "distributed-control")]
    async fn publish_control_command(
        &self,
        command_type: &str,
        ordering_key: &str,
        payload: serde_json::Value,
    ) -> Result<()> {
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            control
                .publish_command(command_type, ordering_key, payload)
                .await?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn bind_session_route(
        &self,
        _session_id: &str,
        _vm_id: &str,
        _endpoint: &str,
        _fork_id: Option<&str>,
        _tenant_id: Option<&str>,
        _workspace_id: Option<&str>,
        _expected_fence: Option<&str>,
    ) -> Result<Option<String>> {
        #[cfg(feature = "distributed-control")]
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            let route = control
                .put_session_route(
                    distributed::SessionRoute {
                        session_id: _session_id.to_string(),
                        vm_id: _vm_id.to_string(),
                        endpoint: _endpoint.to_string(),
                        node_id: None,
                        fork_id: _fork_id.map(ToOwned::to_owned),
                        ownership_fence: None,
                        tenant_id: _tenant_id.unwrap_or("default").to_string(),
                        workspace_id: _workspace_id.unwrap_or("default").to_string(),
                    },
                    _expected_fence,
                )
                .await?;
            let fence = route.ownership_fence.clone();
            let tenant_id = route.tenant_id.clone();
            let workspace_id = route.workspace_id.clone();
            let _ = control
                .publish_event(
                    "session.bound",
                    serde_json::json!({
                        "session_id": _session_id,
                        "vm_id": _vm_id,
                        "endpoint": _endpoint,
                        "fork_id": _fork_id,
                        "tenant_id": tenant_id,
                        "workspace_id": workspace_id,
                        "ownership_fence": fence,
                    }),
                )
                .await;
            return Ok(route.ownership_fence);
        }
        let _ = _expected_fence;
        Ok(None)
    }

    async fn current_session_scope(&self, session_id: &str) -> Result<(String, String)> {
        #[cfg(not(feature = "distributed-control"))]
        let _ = session_id;
        #[cfg(feature = "distributed-control")]
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            if let Some(route) = control.get_session_route(session_id).await? {
                return Ok((route.tenant_id, route.workspace_id));
            }
        }
        Ok(("default".to_string(), "default".to_string()))
    }

    async fn clear_session_route(
        &self,
        _session_id: Option<&str>,
        _vm_id: &str,
        _expected_fence: Option<&str>,
    ) -> Result<()> {
        #[cfg(feature = "distributed-control")]
        if let (ControlBackend::Distributed(control), Some(session_id)) =
            (&self.inner.control_backend, _session_id)
        {
            control
                .delete_session_route(session_id, _expected_fence)
                .await?;
            let _ = control
                .publish_event(
                    "session.discarded",
                    serde_json::json!({
                        "session_id": session_id,
                        "vm_id": _vm_id,
                        "ownership_fence": _expected_fence,
                    }),
                )
                .await;
        }
        Ok(())
    }

    async fn current_session_fence(&self, session_id: &str) -> Result<Option<String>> {
        #[cfg(feature = "distributed-control")]
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            return Ok(control
                .get_session_route(session_id)
                .await?
                .and_then(|route| route.ownership_fence));
        }
        let _ = session_id;
        Ok(None)
    }

    async fn ensure_vm_and_get_rpc_port_for_session(
        &self,
        session_id: &str,
        vm_id: &str,
        endpoint: &str,
        expected_fence: Option<&str>,
    ) -> Result<(String, i32, Option<String>)> {
        let (resolved_endpoint, next_fence) = self
            .resolve_session_endpoint(session_id, vm_id, endpoint, expected_fence)
            .await?;
        match self
            .ensure_vm_and_get_rpc_port(vm_id, &resolved_endpoint)
            .await
        {
            Ok(rpc_port) => Ok((resolved_endpoint, rpc_port, next_fence)),
            Err(err) => {
                if !is_rebind_candidate_error(&err) {
                    return Err(err);
                }

                // @dive: If guest RPC readiness fails due transport loss, try cross-node rebind before surfacing failure.
                if let Some((candidate_endpoint, rebound_fence)) = self
                    .rebind_session_endpoint(
                        session_id,
                        vm_id,
                        &resolved_endpoint,
                        next_fence.as_deref().or(expected_fence),
                        "ensure_vm_and_get_rpc_port",
                    )
                    .await?
                {
                    let rpc_port = self
                        .ensure_vm_and_get_rpc_port(vm_id, &candidate_endpoint)
                        .await?;
                    return Ok((candidate_endpoint, rpc_port, rebound_fence.or(next_fence)));
                }

                Err(err)
            }
        }
    }

    async fn resolve_session_endpoint(
        &self,
        session_id: &str,
        vm_id: &str,
        endpoint: &str,
        expected_fence: Option<&str>,
    ) -> Result<(String, Option<String>)> {
        let normalized_endpoint = normalize_endpoint(endpoint)?;
        match self.ensure_vm_running(vm_id, &normalized_endpoint).await {
            Ok(_) => Ok((normalized_endpoint, None)),
            Err(initial_err) => {
                if let Some((rebound_endpoint, next_fence)) = self
                    .rebind_session_endpoint(
                        session_id,
                        vm_id,
                        &normalized_endpoint,
                        expected_fence,
                        "resolve_session_endpoint",
                    )
                    .await?
                {
                    return Ok((rebound_endpoint, next_fence));
                }
                Err(initial_err)
            }
        }
    }

    async fn rebind_session_endpoint(
        &self,
        session_id: &str,
        vm_id: &str,
        from_endpoint: &str,
        expected_fence: Option<&str>,
        reason: &str,
    ) -> Result<Option<(String, Option<String>)>> {
        for candidate in self.candidate_endpoints().await? {
            if candidate == from_endpoint {
                continue;
            }
            let Some(candidate_vm) = self.find_vm_by_id_on_endpoint(vm_id, &candidate).await? else {
                continue;
            };

            #[cfg(feature = "distributed-control")]
            if let ControlBackend::Distributed(control) = &self.inner.control_backend {
                // @dive: Rebind events include the trigger reason so failover behavior can be audited per stage.
                let _ = control
                    .publish_event(
                        "stream.rebinding",
                        json!({
                            "session_id": session_id,
                            "vm_id": vm_id,
                            "from_endpoint": from_endpoint,
                            "to_endpoint": candidate.clone(),
                            "expected_fence": expected_fence,
                            "reason": reason,
                        }),
                    )
                    .await;
            }

            let restore_snapshot_id = match self
                .restore_execution_state_if_needed(session_id, vm_id, &candidate, &candidate_vm, true)
                .await
            {
                Ok(snapshot_id) => snapshot_id,
                Err(err) => {
                    if matches!(err, SandboxError::InvalidResponse(_)) {
                        return Err(err);
                    }
                    #[cfg(feature = "distributed-control")]
                    if let ControlBackend::Distributed(control) = &self.inner.control_backend {
                        let _ = control
                            .publish_event(
                                "stream.failed",
                                json!({
                                    "session_id": session_id,
                                    "vm_id": vm_id,
                                    "from_endpoint": from_endpoint,
                                    "to_endpoint": candidate.clone(),
                                    "stage": "execution_state_restore",
                                    "reason": reason,
                                    "error": err.to_string(),
                                }),
                            )
                            .await;
                    }
                    continue;
                }
            };

            if self.ensure_vm_running(vm_id, &candidate).await.is_err() {
                #[cfg(feature = "distributed-control")]
                if let ControlBackend::Distributed(control) = &self.inner.control_backend {
                    let _ = control
                        .publish_event(
                            "stream.failed",
                            json!({
                                "session_id": session_id,
                                "vm_id": vm_id,
                                "from_endpoint": from_endpoint,
                                "to_endpoint": candidate.clone(),
                                "stage": "ensure_vm_running",
                                "reason": reason,
                            }),
                        )
                        .await;
                }
                continue;
            }

            let mut next_fence = None;
            #[cfg(feature = "distributed-control")]
            if let ControlBackend::Distributed(_control) = &self.inner.control_backend {
                let (tenant_id, workspace_id) = self.current_session_scope(session_id).await?;
                next_fence = self
                    .bind_session_route(
                        session_id,
                        vm_id,
                        &candidate,
                        None,
                        Some(tenant_id.as_str()),
                        Some(workspace_id.as_str()),
                        expected_fence,
                    )
                    .await?;
            }

            let mut ready = self.inner.ready_vm_rpc.lock().await;
            ready.remove(&ready_key(from_endpoint, vm_id));
            drop(ready);

            #[cfg(feature = "distributed-control")]
            if let ControlBackend::Distributed(control) = &self.inner.control_backend {
                let _ = control
                    .publish_event(
                        "session.rebound",
                        json!({
                            "session_id": session_id,
                            "vm_id": vm_id,
                            "from_endpoint": from_endpoint,
                            "to_endpoint": candidate.clone(),
                            "expected_fence": expected_fence,
                            "reason": reason,
                        }),
                    )
                    .await;
                let _ = control
                    .publish_event(
                        "stream.rebound",
                        json!({
                            "session_id": session_id,
                            "vm_id": vm_id,
                            "from_endpoint": from_endpoint,
                            "to_endpoint": candidate.clone(),
                            "restored_snapshot_id": restore_snapshot_id,
                            "expected_fence": expected_fence,
                            "reason": reason,
                        }),
                    )
                    .await;
            } else {
                let _ = restore_snapshot_id;
            }

            return Ok(Some((candidate, next_fence)));
        }

        let _ = (expected_fence, reason);
        Ok(None)
    }

    async fn restore_execution_state_if_needed(
        &self,
        _session_id: &str,
        vm_id: &str,
        endpoint: &str,
        vm: &Vm,
        enforce_tier_b: bool,
    ) -> Result<Option<String>> {
        let tier_b_eligible = vm_tier_b_eligible(vm);
        let Some(snapshot_id) = execution_restore_snapshot_id(vm) else {
            if enforce_tier_b && tier_b_eligible {
                return Err(SandboxError::InvalidResponse(format!(
                    "tier_b_eligible session `{_session_id}` requires execution restore snapshot marker"
                )));
            }
            return Ok(None);
        };

        let mut client = self.vmd_client_for_endpoint(endpoint).await?;
        client
            .restore_snapshot(self.request_with_auth(RestoreSnapshotRequest {
                vm_id: vm_id.to_string(),
                snapshot_id: snapshot_id.clone(),
            }))
            .await?;

        #[cfg(feature = "distributed-control")]
        if let ControlBackend::Distributed(control) = &self.inner.control_backend {
            let _ = control
                .publish_event(
                    "execution_state.restored",
                    json!({
                        "session_id": _session_id,
                        "vm_id": vm_id,
                        "endpoint": endpoint,
                        "snapshot_id": snapshot_id,
                        "tier_b_eligible": tier_b_eligible,
                    }),
                )
                .await;
        }
        let _ = (_session_id, enforce_tier_b);
        Ok(Some(snapshot_id))
    }

    async fn find_vm_by_id_on_endpoint(&self, vm_id: &str, endpoint: &str) -> Result<Option<Vm>> {
        let mut client = match self.vmd_client_for_endpoint(endpoint).await {
            Ok(client) => client,
            Err(_) => return Ok(None),
        };
        match client
            .get_vm(self.request_with_auth(GetVmRequest {
                vm_id: vm_id.to_string(),
            }))
            .await
        {
            Ok(response) => Ok(Some(response.into_inner())),
            Err(status) if status.code() == tonic::Code::NotFound => Ok(None),
            Err(_) => Ok(None),
        }
    }

    async fn find_vm_by_session_id(&self, session_id: &str) -> Result<Option<(Vm, String)>> {
        #[cfg(feature = "distributed-control")]
        match &self.inner.control_backend {
            ControlBackend::Distributed(control) => {
                if let Some(endpoint) = control
                    .get_session_route(session_id)
                    .await?
                    .map(|route| route.endpoint)
                    .filter(|endpoint| !endpoint.trim().is_empty())
                {
                    let endpoint = normalize_endpoint(&endpoint)?;
                    if let Some(vm) = self
                        .find_vm_by_session_id_on_endpoint(session_id, &endpoint)
                        .await?
                    {
                        return Ok(Some((vm, endpoint)));
                    }
                }
            }
            ControlBackend::Direct => {}
        }

        for endpoint in self.candidate_endpoints().await? {
            if let Some(vm) = self
                .find_vm_by_session_id_on_endpoint(session_id, &endpoint)
                .await?
            {
                return Ok(Some((vm, endpoint)));
            }
        }

        Ok(None)
    }

    async fn find_vm_by_session_id_on_endpoint(
        &self,
        session_id: &str,
        endpoint: &str,
    ) -> Result<Option<Vm>> {
        let mut client = match self.vmd_client_for_endpoint(endpoint).await {
            Ok(client) => client,
            Err(_) => return Ok(None),
        };
        let response = match client
            .list_v_ms(self.request_with_auth(ListVMsRequest {
                include_snapshots: false,
            }))
            .await
        {
            Ok(response) => response.into_inner(),
            Err(_) => return Ok(None),
        };

        Ok(response
            .vms
            .into_iter()
            .find(|vm| vm.metadata.get(META_SESSION_ID).map(String::as_str) == Some(session_id)))
    }

    async fn ensure_vm_running(&self, vm_id: &str, endpoint: &str) -> Result<Vm> {
        let mut client = self.vmd_client_for_endpoint(endpoint).await?;
        let mut vm = client
            .get_vm(self.request_with_auth(GetVmRequest {
                vm_id: vm_id.to_string(),
            }))
            .await?
            .into_inner();

        let is_running = vm.state == proto::vmd::v1::VmState::Running as i32;
        if !is_running {
            let mut ready = self.inner.ready_vm_rpc.lock().await;
            ready.remove(&ready_key(endpoint, vm_id));
            drop(ready);
            vm = client
                .start_vm(self.request_with_auth(VmActionRequest {
                    vm_id: vm_id.to_string(),
                }))
                .await?
                .into_inner();
        }

        Ok(vm)
    }

    async fn ensure_vm_and_get_rpc_port(&self, vm_id: &str, endpoint: &str) -> Result<i32> {
        let vm = self.ensure_vm_running(vm_id, endpoint).await?;
        let rpc_port = vm
            .network
            .and_then(|network| network.portproxy_ports)
            .map(|ports| ports.rpc_port)
            .filter(|port| *port > 0)
            .ok_or_else(|| SandboxError::InvalidResponse("VM missing rpc port".into()))?;

        self.ensure_portproxy_ready(vm_id, endpoint, rpc_port)
            .await?;
        Ok(rpc_port)
    }

    async fn ensure_portproxy_ready(
        &self,
        vm_id: &str,
        endpoint: &str,
        rpc_port: i32,
    ) -> Result<()> {
        let cache_key = ready_key(endpoint, vm_id);
        let cached_ready = {
            let ready = self.inner.ready_vm_rpc.lock().await;
            ready.get(&cache_key).copied() == Some(rpc_port)
        };
        if cached_ready {
            let probe_endpoint = format!("http://127.0.0.1:{rpc_port}");
            if ShellExecClient::connect(probe_endpoint).await.is_ok() {
                return Ok(());
            }
            // @dive: Guest RPC readiness cache is invalidated on failed probe so failover/rebind logic can recover.
            let mut ready = self.inner.ready_vm_rpc.lock().await;
            ready.remove(&cache_key);
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
                                    let is_transport_not_ready = status.code()
                                        == tonic::Code::Unavailable
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
                        ready.insert(cache_key.clone(), rpc_port);
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
                        ready.insert(cache_key.clone(), rpc_port);
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
        endpoint: &str,
    ) -> Result<PortProxyClient<tonic::transport::Channel>> {
        let rpc_port = self.ensure_vm_and_get_rpc_port(vm_id, endpoint).await?;
        let endpoint = format!("http://127.0.0.1:{rpc_port}");
        Ok(PortProxyClient::connect(endpoint).await?)
    }

    async fn discard_vm(
        &self,
        vm_id: &str,
        endpoint: &str,
        session_id: Option<&str>,
        expected_fence: Option<&str>,
    ) -> Result<()> {
        #[cfg(feature = "distributed-control")]
        self.publish_control_command(
            "session.discard",
            session_id.unwrap_or(vm_id),
            json!({
                "session_id": session_id,
                "vm_id": vm_id,
                "endpoint": endpoint,
                "expected_fence": expected_fence,
            }),
        )
        .await?;

        let mut client = self.vmd_client_for_endpoint(endpoint).await?;
        let _ = client
            .stop_vm(self.request_with_auth(VmActionRequest {
                vm_id: vm_id.to_string(),
            }))
            .await;
        client
            .delete_vm(self.request_with_auth(proto::vmd::v1::DeleteVmRequest {
                vm_id: vm_id.to_string(),
                purge_snapshots: true,
            }))
            .await?;
        let mut ready = self.inner.ready_vm_rpc.lock().await;
        ready.remove(&ready_key(endpoint, vm_id));
        drop(ready);
        self.clear_session_route(session_id, vm_id, expected_fence)
            .await?;
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

fn compile_auth_header(raw_token: Option<&str>) -> Result<Option<MetadataValue<Ascii>>> {
    let Some(raw_token) = raw_token else {
        return Ok(None);
    };
    let trimmed = raw_token.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    let value = if trimmed.starts_with("Bearer ") {
        trimmed.to_string()
    } else {
        format!("Bearer {trimmed}")
    };
    let metadata = MetadataValue::try_from(value.as_str()).map_err(|err| {
        SandboxError::InvalidConfig(format!("authorization token is not valid ASCII: {err}"))
    })?;
    Ok(Some(metadata))
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

fn portproxy_server_addr(endpoint: &str, proxy_port: u16) -> Result<String> {
    let host = endpoint_host(endpoint)?;
    if host.contains(':') {
        return Ok(format!("[{host}]:{proxy_port}"));
    }
    Ok(format!("{host}:{proxy_port}"))
}

fn endpoint_host(endpoint: &str) -> Result<String> {
    let normalized = normalize_endpoint(endpoint)?;
    let without_scheme = normalized
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(normalized.as_str());
    let authority = without_scheme.split('/').next().ok_or_else(|| {
        SandboxError::InvalidEndpoint(format!("endpoint missing authority: {normalized}"))
    })?;
    if authority.is_empty() {
        return Err(SandboxError::InvalidEndpoint(format!(
            "endpoint missing authority: {normalized}"
        )));
    }
    let authority = authority.rsplit('@').next().unwrap_or(authority);

    if authority.starts_with('[') {
        if let Some(close_idx) = authority.find(']') {
            let host = &authority[1..close_idx];
            if host.is_empty() {
                return Err(SandboxError::InvalidEndpoint(format!(
                    "endpoint has empty host: {normalized}"
                )));
            }
            return Ok(normalize_dial_host(host).to_string());
        }
        return Err(SandboxError::InvalidEndpoint(format!(
            "endpoint has malformed ipv6 host: {normalized}"
        )));
    }

    if let Some((host, _port)) = authority.rsplit_once(':') {
        if !host.is_empty() && !host.contains(':') {
            return Ok(normalize_dial_host(host).to_string());
        }
    }

    Ok(normalize_dial_host(authority).to_string())
}

fn normalize_dial_host(host: &str) -> &str {
    match host {
        "0.0.0.0" => "127.0.0.1",
        "::" => "::1",
        _ => host,
    }
}

fn ready_key(endpoint: &str, vm_id: &str) -> String {
    format!("{endpoint}::{vm_id}")
}

fn warm_pool_key(endpoint: &str, image: &str, architecture: &str) -> String {
    format!("{endpoint}::{image}::{}", normalize_architecture_label(architecture))
}

fn normalize_architecture_label(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "x86_64" | "amd64" => "amd64".to_string(),
        "aarch64" | "arm64" => "arm64".to_string(),
        other => other.to_string(),
    }
}

fn detect_host_architecture_label() -> Option<String> {
    let detected = normalize_architecture_label(std::env::consts::ARCH);
    if detected.is_empty() {
        None
    } else {
        Some(detected.to_string())
    }
}

fn resolve_tier_b_eligibility(metadata: &HashMap<String, String>) -> bool {
    if let Some(raw) = metadata.get(META_TIER_B_ELIGIBLE).or_else(|| metadata.get("tier_b_eligible")) {
        return parse_bool_like(raw).unwrap_or(true);
    }
    true
}

fn vm_tier_b_eligible(vm: &Vm) -> bool {
    resolve_tier_b_eligibility(&vm.metadata)
}

fn parse_bool_like(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}

fn is_rebind_candidate_error(err: &SandboxError) -> bool {
    match err {
        SandboxError::Grpc(status) => {
            let message = status.message().to_ascii_lowercase();
            status.code() == tonic::Code::Unavailable
                || status.code() == tonic::Code::Unknown
                || message.contains("transport error")
                || message.contains("connection reset")
                || message.contains("broken pipe")
                || message.contains("connection refused")
        }
        SandboxError::DaemonUnavailable(message) => {
            let lower = message.to_ascii_lowercase();
            lower.contains("did not become ready")
                || lower.contains("transport")
                || lower.contains("connection reset")
                || lower.contains("broken pipe")
                || lower.contains("connection refused")
        }
        _ => false,
    }
}

fn execution_restore_snapshot_id(vm: &Vm) -> Option<String> {
    // @dive: Restore selection prefers explicit snapshot IDs, then resolves snapshot names to IDs for compatibility.
    if let Some(snapshot_id) = vm
        .metadata
        .get(META_EXEC_RESTORE_SNAPSHOT_ID)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        return Some(snapshot_id.to_string());
    }

    let mut candidate_names = Vec::new();
    if let Some(snapshot_name) = vm
        .metadata
        .get(META_EXEC_RESTORE_SNAPSHOT_NAME)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        candidate_names.push(snapshot_name.to_string());
    }
    if let Some(fork_snapshot_name) = vm
        .metadata
        .get(META_FORK_SNAPSHOT)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        candidate_names.push(fork_snapshot_name.to_string());
    }
    for snapshot_name in candidate_names {
        if let Some(snapshot_id) = vm
            .snapshots
            .iter()
            .find(|snapshot| snapshot.name == snapshot_name)
            .map(|snapshot| snapshot.id.clone())
            .filter(|id| !id.trim().is_empty())
        {
            return Some(snapshot_id);
        }
    }
    None
}

fn log_slo_observation(metric: &str, elapsed: Duration, outcome: &str) {
    let elapsed_ms = elapsed.as_millis().min(u128::from(u64::MAX)) as u64;
    tracing::info!(
        target: "reson_sandbox::slo",
        metric = metric,
        elapsed_ms = elapsed_ms,
        outcome = outcome,
        "slo observation"
    );
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

    #[test]
    fn portproxy_server_addr_uses_endpoint_host() {
        let value = portproxy_server_addr("http://sandbox-node.internal:18072", 3001)
            .expect("address should derive host");
        assert_eq!(value, "sandbox-node.internal:3001");
    }

    #[test]
    fn portproxy_server_addr_rewrites_unspecified_hosts_for_local_dial() {
        let v4 = portproxy_server_addr("http://0.0.0.0:18072", 3001)
            .expect("v4 unspecified host should normalize");
        assert_eq!(v4, "127.0.0.1:3001");

        let v6 = portproxy_server_addr("http://[::]:18072", 3001)
            .expect("v6 unspecified host should normalize");
        assert_eq!(v6, "[::1]:3001");
    }

    #[test]
    fn portproxy_server_addr_supports_ipv6_endpoints() {
        let value = portproxy_server_addr("http://[2001:db8::42]:18072", 3001)
            .expect("ipv6 host should preserve brackets");
        assert_eq!(value, "[2001:db8::42]:3001");
    }

    #[test]
    fn compile_auth_header_adds_bearer_prefix() {
        let value = compile_auth_header(Some("token-value"))
            .expect("auth header should compile")
            .expect("auth header should be present");
        assert_eq!(value.to_str().expect("ascii header"), "Bearer token-value");
    }

    #[test]
    fn compile_auth_header_keeps_existing_bearer_prefix() {
        let value = compile_auth_header(Some("Bearer token-value"))
            .expect("auth header should compile")
            .expect("auth header should be present");
        assert_eq!(value.to_str().expect("ascii header"), "Bearer token-value");
    }
}
