// @dive-file: Implements gRPC PortProxy, ShellExec, and DaemonManager services used inside guest VMs.
// @dive-rel: Uses portproxy/src/daemon.rs to provide named daemon exec streams that can be reattached.
// @dive-rel: Conforms to proto/bracket/portproxy/v1/portproxy.proto service contracts.
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, Mutex as StdMutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::Stream;
use tokio::fs;
use tokio::io::{self, AsyncReadExt, AsyncWriteExt};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use tracing::{debug, error, warn};

use crate::child_tracker::ChildTracker;
use crate::daemon::{DaemonError, DaemonRegistry, build_command_builder};
use crate::pb::bracket::portproxy::v1::daemon_manager_server::DaemonManager;
use crate::pb::bracket::portproxy::v1::port_proxy_server::PortProxy;
use crate::pb::bracket::portproxy::v1::shell_exec_server::ShellExec;
use crate::pb::bracket::portproxy::v1::{
    AttachDaemonRequest, AttachDaemonResponse, DeletePathRequest, DirectoryEntry, ExecRequest,
    ExecResponse, ExecStart, InteractiveShellRequest, InteractiveShellResponse,
    ListDirectoryRequest, ListDirectoryResponse, ReadFileRequest, ReadFileResponse,
    WriteFileRequest,
};
use crate::pb::bracket::portproxy::v1::{ExecDaemonRequest, ExecDaemonResponse};
use crate::pb::google::protobuf::Empty;

type ExecResponseStream = ReceiverStream<Result<ExecResponse, Status>>;
type InteractiveResponseStream = ReceiverStream<Result<InteractiveShellResponse, Status>>;
type AttachDaemonResponseStream =
    GuardedStream<ReceiverStream<Result<AttachDaemonResponse, Status>>>;

pub struct GuardedStream<S> {
    _guard: tokio::sync::OwnedMutexGuard<()>,
    inner: S,
}

impl<S> GuardedStream<S> {
    fn new(guard: tokio::sync::OwnedMutexGuard<()>, inner: S) -> Self {
        Self {
            _guard: guard,
            inner,
        }
    }
}

impl<S> Stream for GuardedStream<S>
where
    S: Stream + Unpin,
{
    type Item = S::Item;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let inner = unsafe { self.map_unchecked_mut(|s| &mut s.inner) };
        inner.poll_next(cx)
    }
}

const READ_BUFFER: usize = 4096;

#[derive(Clone)]
pub struct ShellExecService {
    tracker: ChildTracker,
}

impl ShellExecService {
    pub fn new(tracker: ChildTracker) -> Self {
        Self { tracker }
    }
}

#[tonic::async_trait]
impl ShellExec for ShellExecService {
    type ExecStream = ExecResponseStream;
    type InteractiveShellStream = InteractiveResponseStream;

    async fn exec(
        &self,
        request: Request<tonic::Streaming<ExecRequest>>,
    ) -> Result<Response<Self::ExecStream>, Status> {
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream ended before ExecStart"))?;

        let ExecRequest {
            request: Some(crate::pb::bracket::portproxy::v1::exec_request::Request::Start(start)),
        } = first
        else {
            return Err(Status::invalid_argument("first message must be ExecStart"));
        };

        let args = validate_args(&start)?;

        let mut command = Command::new(&args[0]);
        if args.len() > 1 {
            command.args(&args[1..]);
        }
        command.stdin(std::process::Stdio::piped());
        command.stdout(std::process::Stdio::piped());
        command.stderr(std::process::Stdio::piped());
        command.env_clear();
        for (key, value) in &start.env {
            command.env(key, value);
        }

        let mut child = command
            .spawn()
            .map_err(|err| Status::internal(format!("failed to start command: {err}")))?;

        let pid = child.id().unwrap_or(0) as i32;
        if pid > 0 {
            self.tracker.register(pid).await;
        }

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| Status::internal("missing stdin handle"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Status::internal("missing stdout handle"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| Status::internal("missing stderr handle"))?;

        let timeout = start.timeout.map(|secs| Duration::from_secs(secs as u64));

        let (tx, rx) = mpsc::channel(32);

        spawn_reader(stdout, tx.clone(), |data| ExecResponse {
            response: Some(
                crate::pb::bracket::portproxy::v1::exec_response::Response::StdoutData(data),
            ),
        });
        spawn_reader(stderr, tx.clone(), |data| ExecResponse {
            response: Some(
                crate::pb::bracket::portproxy::v1::exec_response::Response::StderrData(data),
            ),
        });

        tokio::spawn(async move {
            loop {
                match stream.message().await {
                    Ok(Some(ExecRequest {
                        request:
                            Some(crate::pb::bracket::portproxy::v1::exec_request::Request::StdinData(
                                data,
                            )),
                    })) => {
                        if let Err(err) = stdin.write_all(&data).await {
                            debug!("failed writing to stdin: {err}");
                            break;
                        }
                    }
                    Ok(Some(_)) => {
                        debug!("ignoring unexpected exec message");
                    }
                    Ok(None) => break,
                    Err(err) => {
                        debug!("stdin stream error: {err}");
                        break;
                    }
                }
            }
            let _ = stdin.shutdown().await;
        });

        let tracker = self.tracker.clone();
        tokio::spawn(async move {
            let status_result = if let Some(duration) = timeout {
                tokio::time::timeout(duration, child.wait()).await
            } else {
                match child.wait().await {
                    Ok(status) => return finalize_exec(status, pid, tracker, tx).await,
                    Err(err) => {
                        error!("failed waiting for child: {err}");
                        if pid > 0 {
                            tracker.unregister(pid).await;
                        }
                        return;
                    }
                }
            };

            match status_result {
                Ok(Ok(status)) => finalize_exec(status, pid, tracker, tx).await,
                Ok(Err(err)) => {
                    error!("failed waiting for child: {err}");
                    if pid > 0 {
                        tracker.unregister(pid).await;
                    }
                }
                Err(_) => {
                    debug!("command timed out, killing pid {pid}");
                    if let Err(err) = child.start_kill() {
                        debug!("failed to kill timed out process: {err}");
                    }
                    let _ = tx
                        .send(Ok(ExecResponse {
                            response: Some(
                                crate::pb::bracket::portproxy::v1::exec_response::Response::StderrData(
                                    b"\n(Command timed out)".to_vec(),
                                ),
                            ),
                        }))
                        .await;

                    match child.wait().await {
                        Ok(_) => {
                            finalize_exec_with_code(pid, tracker, tx, 124).await;
                        }
                        Err(err) => {
                            error!("failed waiting after timeout: {err}");
                            if pid > 0 {
                                tracker.unregister(pid).await;
                            }
                        }
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn interactive_shell(
        &self,
        request: Request<tonic::Streaming<InteractiveShellRequest>>,
    ) -> Result<Response<Self::InteractiveShellStream>, Status> {
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream ended before start"))?;

        let InteractiveShellRequest {
            request:
                Some(crate::pb::bracket::portproxy::v1::interactive_shell_request::Request::Start(
                    start,
                )),
        } = first
        else {
            return Err(Status::invalid_argument(
                "first message must be InteractiveShellStart",
            ));
        };

        let shell = if start.shell.is_empty() {
            "/bin/sh".to_string()
        } else {
            start.shell.clone()
        };

        let pty_system = portable_pty::native_pty_system();
        let pair = pty_system
            .openpty(Default::default())
            .map_err(|err| Status::internal(format!("failed to open pty: {err}")))?;

        let builder = build_command_builder(&shell, &start.args, &start.env, {
            if start.cwd.is_empty() {
                None
            } else {
                Some(start.cwd.clone())
            }
        });

        let mut child = pair
            .slave
            .spawn_command(builder)
            .map_err(|err| Status::internal(format!("failed to spawn interactive shell: {err}")))?;

        let killer = child.clone_killer();
        let killer = Arc::new(StdMutex::new(Some(killer)));

        let pid = child.process_id().unwrap_or(0) as i32;
        if pid > 0 {
            self.tracker.register(pid).await;
        }

        let mut reader = pair
            .master
            .try_clone_reader()
            .map_err(|err| Status::internal(format!("failed to clone pty reader: {err}")))?;
        let writer = pair
            .master
            .take_writer()
            .map_err(|err| Status::internal(format!("failed to take pty writer: {err}")))?;

        let (tx, rx) = mpsc::channel(32);

        tokio::task::spawn_blocking({
            let tx = tx.clone();
            move || {
                let mut buf = vec![0u8; READ_BUFFER];
                loop {
                    match reader.read(&mut buf) {
                        Ok(0) => break,
                        Ok(n) => {
                            if tx
                                .blocking_send(Ok(InteractiveShellResponse {
                                    response: Some(
                                        crate::pb::bracket::portproxy::v1::interactive_shell_response::Response::OutputData(
                                            buf[..n].to_vec(),
                                        ),
                                    ),
                                }))
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(err) => {
                            warn!("pty read error: {err}");
                            break;
                        }
                    }
                }
            }
        });

        let writer = Arc::new(StdMutex::new(writer));
        {
            let writer = writer.clone();
            let killer = killer.clone();
            tokio::spawn(async move {
                loop {
                    match stream.message().await {
                        Ok(Some(InteractiveShellRequest {
                            request:
                                Some(
                                    crate::pb::bracket::portproxy::v1::interactive_shell_request::Request::StdinData(
                                        data,
                                    ),
                                ),
                        })) => {
                            let writer = writer.clone();
                            let data = data.to_vec();
                            let result = tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                                let mut guard = writer
                                    .lock()
                                    .map_err(|_| std::io::Error::other("writer poisoned"))?;
                                guard.write_all(&data)?;
                                guard.flush()
                            })
                            .await;
                            match result {
                                Ok(Ok(())) => {}
                                _ => break,
                            }
                        }
                        Ok(Some(_)) => {}
                        Ok(None) => break,
                        Err(err) => {
                            debug!("interactive shell input error: {err}");
                            break;
                        }
                    }
                }

                if let Ok(mut guard) = killer.lock() {
                    if let Some(mut killer) = guard.take() {
                        if let Err(err) = killer.kill() {
                            debug!("interactive shell kill failed: {err}");
                        }
                    }
                }
            });
        }

        let tracker = self.tracker.clone();
        let killer_for_wait = killer.clone();
        tokio::spawn(async move {
            let status = match tokio::task::spawn_blocking(move || child.wait()).await {
                Ok(res) => res,
                Err(err) => Err(std::io::Error::other(err)),
            };

            if pid > 0 {
                tracker.unregister(pid).await;
            }

            if let Ok(mut guard) = killer_for_wait.lock() {
                guard.take();
            }

            let exit_code = match status {
                Ok(exit_status) => {
                    if exit_status.signal().is_some() {
                        -1
                    } else {
                        exit_status.exit_code() as i32
                    }
                }
                Err(err) => {
                    error!("interactive shell wait failed: {err}");
                    -1
                }
            };

            let _ = tx
                .send(Ok(InteractiveShellResponse {
                    response: Some(
                        crate::pb::bracket::portproxy::v1::interactive_shell_response::Response::ExitCode(exit_code),
                    ),
                }))
                .await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[derive(Clone)]
pub struct PortProxyService {
    tracker: ChildTracker,
}

impl PortProxyService {
    pub fn new(tracker: ChildTracker) -> Self {
        Self { tracker }
    }
}

#[tonic::async_trait]
impl PortProxy for PortProxyService {
    async fn prepare_shutdown(&self, _request: Request<Empty>) -> Result<Response<Empty>, Status> {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;

        let pids = self.tracker.snapshot().await;
        for pid in &pids {
            if *pid <= 0 {
                continue;
            }
            if let Err(err) = kill(Pid::from_raw(*pid), Signal::SIGTERM) {
                warn!("failed to send SIGTERM to pid {}: {}", pid, err);
            }
        }

        tokio::time::sleep(Duration::from_secs(2)).await;

        for pid in pids {
            if pid <= 0 {
                continue;
            }
            if let Err(err) = kill(Pid::from_raw(pid), Signal::SIGKILL) {
                warn!("failed to send SIGKILL to pid {}: {}", pid, err);
            }
        }
        Ok(Response::new(Empty {}))
    }

    async fn read_file(
        &self,
        request: Request<ReadFileRequest>,
    ) -> Result<Response<ReadFileResponse>, Status> {
        let path = validate_path(&request.get_ref().path)?;
        match fs::read(&path).await {
            Ok(data) => Ok(Response::new(ReadFileResponse { data })),
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                Err(Status::not_found(format!("path {:?} not found", path)))
            }
            Err(err) => Err(Status::internal(format!(
                "failed to read file {:?}: {err}",
                path
            ))),
        }
    }

    async fn write_file(
        &self,
        request: Request<WriteFileRequest>,
    ) -> Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let path = validate_path(&req.path)?;
        if req.create_parents {
            if let Some(parent) = path.parent() {
                if let Err(err) = fs::create_dir_all(parent).await {
                    return Err(Status::internal(format!(
                        "failed to create parents for {:?}: {err}",
                        path
                    )));
                }
            }
        }
        if let Err(err) = fs::write(&path, req.data).await {
            return Err(Status::internal(format!(
                "failed to write {:?}: {err}",
                path
            )));
        }
        Ok(Response::new(Empty {}))
    }

    async fn list_directory(
        &self,
        request: Request<ListDirectoryRequest>,
    ) -> Result<Response<ListDirectoryResponse>, Status> {
        let path = validate_path(&request.get_ref().path)?;
        let mut entries = fs::read_dir(&path)
            .await
            .map_err(|err| map_fs_error(err, &path))?;

        let mut resp = ListDirectoryResponse { entries: vec![] };
        while let Some(entry) = entries.next_entry().await? {
            let file_type = entry.file_type().await.map_err(|err| {
                Status::internal(format!(
                    "failed to read metadata for {:?}: {err}",
                    entry.path()
                ))
            })?;
            resp.entries.push(DirectoryEntry {
                name: entry.file_name().to_string_lossy().into_owned(),
                is_dir: file_type.is_dir(),
                is_symlink: file_type.is_symlink(),
            });
        }
        Ok(Response::new(resp))
    }

    async fn delete_path(
        &self,
        request: Request<DeletePathRequest>,
    ) -> Result<Response<Empty>, Status> {
        let path = validate_path(&request.get_ref().path)?;
        let path_clone = path.clone();
        tokio::task::spawn_blocking(move || {
            if path_clone.is_dir() {
                std::fs::remove_dir_all(&path_clone)
            } else {
                std::fs::remove_file(&path_clone)
            }
        })
        .await
        .map_err(|err| Status::internal(format!("delete task failed: {err}")))?
        .map_err(|err| Status::internal(format!("failed to delete {:?}: {err}", path)))?;

        Ok(Response::new(Empty {}))
    }
}

#[derive(Clone)]
pub struct DaemonManagerService {
    registry: DaemonRegistry,
}

impl DaemonManagerService {
    pub fn new(registry: DaemonRegistry) -> Self {
        Self { registry }
    }
}

#[tonic::async_trait]
impl DaemonManager for DaemonManagerService {
    type AttachDaemonStream = AttachDaemonResponseStream;

    async fn exec_daemon(
        &self,
        request: Request<ExecDaemonRequest>,
    ) -> Result<Response<ExecDaemonResponse>, Status> {
        match self.registry.exec_daemon(request.into_inner()).await {
            Ok(resp) => Ok(Response::new(resp)),
            Err(DaemonError::AlreadyRunning) => {
                Ok(Response::new(ExecDaemonResponse { is_new: false }))
            }
            Err(DaemonError::InvalidRequest(msg)) => Err(Status::invalid_argument(msg)),
            Err(DaemonError::Spawn(err)) => Err(Status::internal(format!("{err}"))),
        }
    }

    async fn attach_daemon(
        &self,
        request: Request<tonic::Streaming<AttachDaemonRequest>>,
    ) -> Result<Response<Self::AttachDaemonStream>, Status> {
        let mut stream = request.into_inner();
        let first = stream
            .message()
            .await?
            .ok_or_else(|| Status::invalid_argument("stream ended before start"))?;

        let AttachDaemonRequest {
            request:
                Some(crate::pb::bracket::portproxy::v1::attach_daemon_request::Request::Start(start)),
        } = first
        else {
            return Err(Status::invalid_argument(
                "first message must be AttachDaemonStart",
            ));
        };

        let Some(entry) = self.registry.get(&start.name).await else {
            return Err(Status::not_found(format!(
                "daemon {} not found",
                start.name
            )));
        };

        let guard = entry.attach_lock.clone().lock_owned().await;

        let (tx, rx) = mpsc::channel(32);
        let mut stdout_rx = entry.stdout_tx.subscribe();
        let mut stderr_rx = entry.stderr_tx.subscribe();
        let mut exit_rx = entry.exit_tx.subscribe();

        tokio::spawn({
            let tx = tx.clone();
            async move {
                loop {
                    match stdout_rx.recv().await {
                        Ok(data) => {
                            if tx
                                .send(Ok(AttachDaemonResponse {
                                    response: Some(
                                        crate::pb::bracket::portproxy::v1::attach_daemon_response::Response::StdoutData(data),
                                    ),
                                }))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    }
                }
            }
        });

        tokio::spawn({
            let tx = tx.clone();
            async move {
                loop {
                    match stderr_rx.recv().await {
                        Ok(data) => {
                            if tx
                                .send(Ok(AttachDaemonResponse {
                                    response: Some(
                                        crate::pb::bracket::portproxy::v1::attach_daemon_response::Response::StderrData(data),
                                    ),
                                }))
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    }
                }
            }
        });

        tokio::spawn({
            let tx = tx.clone();
            async move {
                loop {
                    let exit_code = *exit_rx.borrow_and_update();
                    if let Some(exit_code) = exit_code {
                        let _ = tx
                            .send(Ok(AttachDaemonResponse {
                                response: Some(
                                    crate::pb::bracket::portproxy::v1::attach_daemon_response::Response::ExitCode(exit_code),
                                ),
                            }))
                            .await;
                        break;
                    }
                    if exit_rx.changed().await.is_err() {
                        break;
                    }
                }
            }
        });

        {
            let entry = entry.clone();
            tokio::spawn(async move {
                loop {
                    match stream.message().await {
                        Ok(Some(AttachDaemonRequest {
                            request:
                                Some(
                                    crate::pb::bracket::portproxy::v1::attach_daemon_request::Request::StdinData(
                                        data,
                                    ),
                                ),
                        })) => {
                            if let Err(err) = entry.stdin.lock().await.write_all(&data).await {
                                debug!("daemon stdin write failed: {err}");
                                break;
                            }
                        }
                        Ok(Some(_)) => {}
                        Ok(None) => break,
                        Err(err) => {
                            debug!("attach daemon stream error: {err}");
                            break;
                        }
                    }
                }
            });
        }

        let stream = GuardedStream::new(guard, ReceiverStream::new(rx));
        Ok(Response::new(stream))
    }
}

fn validate_args(start: &ExecStart) -> Result<Vec<String>, Status> {
    if start.args.is_empty() {
        return Err(Status::invalid_argument("args must not be empty"));
    }
    Ok(start.args.clone())
}

fn validate_path(path: &str) -> Result<PathBuf, Status> {
    if path.is_empty() {
        return Err(Status::invalid_argument("path is required"));
    }
    Ok(PathBuf::from(path))
}

fn map_fs_error(err: std::io::Error, path: &PathBuf) -> Status {
    match err.kind() {
        io::ErrorKind::NotFound => Status::not_found(format!("path {:?} not found", path)),
        io::ErrorKind::PermissionDenied => {
            Status::permission_denied(format!("permission denied for {:?}", path))
        }
        _ => Status::internal(format!("failed to list {:?}: {err}", path)),
    }
}

fn spawn_reader<R, F>(mut reader: R, tx: mpsc::Sender<Result<ExecResponse, Status>>, build: F)
where
    R: io::AsyncRead + Send + std::marker::Unpin + 'static,
    F: Fn(Vec<u8>) -> ExecResponse + Send + Sync + 'static,
{
    tokio::spawn(async move {
        let mut buf = vec![0u8; READ_BUFFER];
        loop {
            match reader.read(&mut buf).await {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(Ok(build(buf[..n].to_vec()))).await.is_err() {
                        break;
                    }
                }
                Err(err) => {
                    warn!("io read error: {err}");
                    break;
                }
            }
        }
    });
}

async fn finalize_exec(
    status: std::process::ExitStatus,
    pid: i32,
    tracker: ChildTracker,
    tx: mpsc::Sender<Result<ExecResponse, Status>>,
) {
    finalize_exec_with_code(pid, tracker, tx, exit_code_from_status(&status)).await;
}

async fn finalize_exec_with_code(
    pid: i32,
    tracker: ChildTracker,
    tx: mpsc::Sender<Result<ExecResponse, Status>>,
    code: i32,
) {
    if pid > 0 {
        tracker.unregister(pid).await;
    }
    if tx
        .send(Ok(ExecResponse {
            response: Some(
                crate::pb::bracket::portproxy::v1::exec_response::Response::ExitCode(code),
            ),
        }))
        .await
        .is_err()
    {
        debug!("failed to send exit code for pid {}", pid);
    }
}

fn exit_code_from_status(status: &std::process::ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    if let Some(code) = status.code() {
        code
    } else {
        status.signal().unwrap_or(-1)
    }
}
