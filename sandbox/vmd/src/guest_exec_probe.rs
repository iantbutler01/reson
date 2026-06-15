// @dive-file: Shared guest exec readiness probe used by command dispatch and VM health reconciliation.
// @dive-rel: Keeps portproxy auth handling and guest-exec probe semantics consistent across vmd subsystems.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use anyhow::{Context, Result, anyhow, bail};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::metadata::{Ascii, MetadataValue};
use tonic::{Code, Request, Status};

use crate::proto::bracket::portproxy::v1::shell_exec_client::ShellExecClient;
use crate::proto::bracket::portproxy::v1::{
    ExecRequest, ExecResponse, ExecStart, exec_request, exec_response,
};

pub const META_PORTPROXY_AUTH_TOKEN: &str = "chevalier.portproxy_auth_token";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuestExecProbeFailureKind {
    PermanentAuth,
    Transient,
}

#[derive(Debug)]
pub struct GuestExecProbeFailure {
    kind: GuestExecProbeFailureKind,
    message: String,
}

impl GuestExecProbeFailure {
    pub fn transient(message: impl Into<String>) -> Self {
        Self {
            kind: GuestExecProbeFailureKind::Transient,
            message: message.into(),
        }
    }

    fn from_status(context: &str, status: Status) -> Self {
        let kind = match status.code() {
            Code::Unauthenticated | Code::PermissionDenied => {
                GuestExecProbeFailureKind::PermanentAuth
            }
            _ => GuestExecProbeFailureKind::Transient,
        };
        Self {
            kind,
            message: format!("{context}: {status}"),
        }
    }

    pub fn kind(&self) -> GuestExecProbeFailureKind {
        self.kind
    }
}

impl fmt::Display for GuestExecProbeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for GuestExecProbeFailure {}

pub fn portproxy_auth_header_from_metadata(
    metadata: &HashMap<String, String>,
) -> Result<Option<MetadataValue<Ascii>>> {
    let token = metadata
        .get(META_PORTPROXY_AUTH_TOKEN)
        .map(String::as_str)
        .map(str::trim)
        .filter(|token| !token.is_empty());
    portproxy_auth_header_from_token(token)
}

pub fn portproxy_auth_header_from_token(
    token: Option<&str>,
) -> Result<Option<MetadataValue<Ascii>>> {
    let Some(token) = token.map(str::trim).filter(|token| !token.is_empty()) else {
        return Ok(None);
    };
    let value = if token.starts_with("Bearer ") {
        token.to_string()
    } else {
        format!("Bearer {token}")
    };
    MetadataValue::try_from(value.as_str())
        .map(Some)
        .context("build portproxy authorization metadata")
}

pub fn request_with_portproxy_auth<T>(
    message: T,
    auth_header: Option<&MetadataValue<Ascii>>,
) -> Request<T> {
    let mut request = Request::new(message);
    if let Some(value) = auth_header {
        request
            .metadata_mut()
            .insert("authorization", value.clone());
    }
    request
}

pub async fn probe_guest_exec_ready(
    endpoint: &str,
    auth_header: Option<&MetadataValue<Ascii>>,
    command_timeout_secs: i32,
) -> Result<(), GuestExecProbeFailure> {
    let mut client = ShellExecClient::connect(endpoint.to_string())
        .await
        .map_err(|error| {
            GuestExecProbeFailure::transient(format!("connect shell exec readiness probe: {error}"))
        })?;
    let (req_tx, req_rx) = mpsc::channel(2);
    req_tx
        .send(ExecRequest {
            request: Some(exec_request::Request::Start(ExecStart {
                args: vec!["/bin/sh".to_string(), "-lc".to_string(), "true".to_string()],
                env: HashMap::new(),
                detach: false,
                timeout: Some(command_timeout_secs),
            })),
        })
        .await
        .map_err(|_| GuestExecProbeFailure::transient("enqueue shell exec readiness probe"))?;
    drop(req_tx);

    let response = client
        .exec(request_with_portproxy_auth(
            ReceiverStream::new(req_rx),
            auth_header,
        ))
        .await
        .map_err(|status| {
            GuestExecProbeFailure::from_status("invoke shell exec readiness probe", status)
        })?;
    let mut stream = response.into_inner();

    while let Some(frame) = stream.message().await.map_err(|status| {
        GuestExecProbeFailure::from_status("read shell exec readiness probe frame", status)
    })? {
        if let ExecResponse {
            response: Some(exec_response::Response::ExitCode(code)),
        } = frame
        {
            if code == 0 {
                return Ok(());
            }
            return Err(GuestExecProbeFailure::transient(format!(
                "shell exec readiness probe exited with status {code}"
            )));
        }
    }

    Err(GuestExecProbeFailure::transient(
        "shell exec readiness probe stream ended before exit code",
    ))
}

pub async fn probe_guest_exec_ready_anyhow(
    endpoint: &str,
    auth_header: Option<&MetadataValue<Ascii>>,
    command_timeout_secs: i32,
) -> Result<()> {
    probe_guest_exec_ready(endpoint, auth_header, command_timeout_secs)
        .await
        .map_err(|error| anyhow!(error.to_string()))
}

pub async fn run_guest_shell_exec(
    endpoint: &str,
    auth_header: Option<&MetadataValue<Ascii>>,
    command: &str,
    stdin: Option<&[u8]>,
    command_timeout_secs: i32,
) -> Result<GuestShellExecOutput> {
    let mut client = ShellExecClient::connect(endpoint.to_string())
        .await
        .context("connect shell exec client")?;
    let (req_tx, req_rx) = mpsc::channel(8);
    req_tx
        .send(ExecRequest {
            request: Some(exec_request::Request::Start(ExecStart {
                args: vec![
                    "/bin/sh".to_string(),
                    "-lc".to_string(),
                    command.to_string(),
                ],
                env: HashMap::new(),
                detach: false,
                timeout: Some(command_timeout_secs),
            })),
        })
        .await
        .map_err(|_| anyhow!("enqueue shell exec start request"))?;
    if let Some(stdin) = stdin {
        for chunk in stdin.chunks(64 * 1024) {
            req_tx
                .send(ExecRequest {
                    request: Some(exec_request::Request::StdinData(chunk.to_vec())),
                })
                .await
                .map_err(|_| anyhow!("enqueue shell exec stdin request"))?;
        }
    }
    drop(req_tx);

    let response = client
        .exec(request_with_portproxy_auth(
            ReceiverStream::new(req_rx),
            auth_header,
        ))
        .await
        .context("invoke shell exec")?;
    let mut stream = response.into_inner();
    let mut output = GuestShellExecOutput::default();

    while let Some(frame) = stream.message().await.context("read shell exec frame")? {
        match frame {
            ExecResponse {
                response: Some(exec_response::Response::StdoutData(bytes)),
            } => output.stdout.extend(bytes),
            ExecResponse {
                response: Some(exec_response::Response::StderrData(bytes)),
            } => output.stderr.extend(bytes),
            ExecResponse {
                response: Some(exec_response::Response::ExitCode(code)),
            } => {
                output.exit_code = Some(code);
                return Ok(output);
            }
            ExecResponse { response: None } => {}
        }
    }

    Ok(output)
}

#[derive(Debug, Default)]
pub struct GuestShellExecOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: Option<i32>,
}

impl GuestShellExecOutput {
    pub fn stdout_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    pub fn stderr_lossy(&self) -> String {
        String::from_utf8_lossy(&self.stderr).into_owned()
    }
}

pub async fn ensure_guest_portproxy_binary(
    endpoint: &str,
    auth_header: Option<&MetadataValue<Ascii>>,
    auth_token: Option<&str>,
    expected_sha256: &str,
    binary: &[u8],
) -> Result<bool> {
    let current = run_guest_shell_exec(
        endpoint,
        auth_header,
        "sha256sum /usr/sbin/portproxy 2>/dev/null | awk '{print $1}' || true",
        None,
        5,
    )
    .await
    .context("read guest portproxy sha256")?;
    let current_sha = current.stdout_lossy().trim().to_string();
    if current_sha == expected_sha256 {
        return Ok(false);
    }

    let token_setup = match auth_token.map(str::trim).filter(|token| !token.is_empty()) {
        Some(token) => format!(
            "mkdir -p /etc/chevalier\nprintf 'CHEVALIER_PORTPROXY_AUTH_TOKEN=%s\\n' {} >/etc/chevalier/portproxy.env\nchmod 0600 /etc/chevalier/portproxy.env\n",
            shell_single_quote(token)
        ),
        None => String::new(),
    };
    let install_script = format!(
        r#"set -eu
tmp=/tmp/chevalier-portproxy.new
cat > "$tmp"
actual="$(sha256sum "$tmp" | awk '{{print $1}}')"
test "$actual" = "{expected_sha256}"
install -m 0755 "$tmp" /usr/sbin/portproxy
{token_setup}
cat >/etc/systemd/system/portproxy.service <<'EOF'
[Unit]
Description=Bracket PortProxy
After=network.target

[Service]
Type=simple
Environment=RUST_LOG=trace
EnvironmentFile=-/etc/chevalier/portproxy.env
ExecStartPre=-/usr/local/sbin/chevalier-apply-tap-network.sh
ExecStart=/usr/sbin/portproxy --server
Restart=on-failure
RestartSec=2
StandardOutput=journal+console
StandardError=journal+console

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload || true
systemctl enable portproxy.service || true
"#
    );
    let installed =
        run_guest_shell_exec(endpoint, auth_header, &install_script, Some(binary), 20).await?;
    if installed.exit_code != Some(0) {
        bail!(
            "install guest portproxy failed exit={:?} stderr={}",
            installed.exit_code,
            installed.stderr_lossy()
        );
    }

    let restart_script = r#"set -eu
nohup /bin/sh -c 'sleep 0.2; pkill -x portproxy || true; sleep 0.1; systemctl start portproxy.service || /usr/sbin/portproxy --server' >/tmp/chevalier-portproxy-upgrade.log 2>&1 &
"#;
    let _ = run_guest_shell_exec(endpoint, auth_header, restart_script, None, 5).await;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if probe_guest_exec_ready(endpoint, auth_header, 3)
            .await
            .is_ok()
        {
            let verified = run_guest_shell_exec(
                endpoint,
                auth_header,
                "sha256sum /usr/sbin/portproxy 2>/dev/null | awk '{print $1}' || true",
                None,
                5,
            )
            .await
            .context("verify upgraded guest portproxy sha256")?;
            if verified.stdout_lossy().trim() == expected_sha256 {
                return Ok(true);
            }
        }
        if tokio::time::Instant::now() >= deadline {
            bail!(
                "guest portproxy upgrade did not become ready with expected sha256 {expected_sha256}"
            );
        }
    }
}

fn shell_single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portproxy_auth_header_uses_vm_metadata_token() {
        let metadata = HashMap::from([(
            META_PORTPROXY_AUTH_TOKEN.to_string(),
            "guest-token".to_string(),
        )]);
        let value = portproxy_auth_header_from_metadata(&metadata)
            .expect("portproxy auth metadata should compile")
            .expect("portproxy auth metadata should be present");
        assert_eq!(value.to_str().expect("ascii header"), "Bearer guest-token");
    }

    #[test]
    fn request_with_portproxy_auth_inserts_authorization_metadata() {
        let metadata = HashMap::from([(
            META_PORTPROXY_AUTH_TOKEN.to_string(),
            "guest-token".to_string(),
        )]);
        let value = portproxy_auth_header_from_metadata(&metadata)
            .expect("portproxy auth metadata should compile")
            .expect("portproxy auth metadata should be present");
        let request = request_with_portproxy_auth((), Some(&value));
        assert_eq!(
            request
                .metadata()
                .get("authorization")
                .and_then(|value| value.to_str().ok()),
            Some("Bearer guest-token")
        );
    }

    #[test]
    fn guest_exec_probe_failure_classifies_auth_statuses_as_permanent() {
        for code in [Code::Unauthenticated, Code::PermissionDenied] {
            let failure = GuestExecProbeFailure::from_status("probe", Status::new(code, "nope"));
            assert_eq!(failure.kind(), GuestExecProbeFailureKind::PermanentAuth);
        }
        let failure =
            GuestExecProbeFailure::from_status("probe", Status::new(Code::Unavailable, "down"));
        assert_eq!(failure.kind(), GuestExecProbeFailureKind::Transient);
    }
}
