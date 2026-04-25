// @dive-file: Daemon-wide network service lifecycle for guest egress helpers such as the explicit HTTP/HTTPS proxy.
// @dive-rel: Owned by vmd startup/shutdown so guest bootstrap and QEMU guestfwd have a real pod-local upstream to target.
// @dive-rel: Generates Envoy/CoreDNS config on disk and supervises both child processes with readiness probing.
mod firewall;
mod vm_counters;

use std::collections::{BTreeMap, VecDeque};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::process::Command;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{debug, warn};
use uuid::Uuid;

use crate::config::Config;
use vm_counters::{VmCounters, VmProxyLimitSnapshot};
pub use vm_counters::{VmProxyAccessEvent, VmProxyActivitySnapshot};

const ENVOY_READY_TIMEOUT: Duration = Duration::from_secs(5);
const ENVOY_READY_POLL_INTERVAL: Duration = Duration::from_millis(50);
const ENVOY_CLUSTER_NAME: &str = "dynamic_forward_proxy_cluster";
const ENVOY_DNS_CACHE_NAME: &str = "dynamic_forward_proxy_cache_config";
const COREDNS_READY_TIMEOUT: Duration = Duration::from_secs(5);
const COREDNS_READY_POLL_INTERVAL: Duration = Duration::from_millis(50);
const LIMIT_ENFORCER_POLL_INTERVAL: Duration = Duration::from_secs(5);
const FALLBACK_THREAT_HOSTS: &[&str] = &[
    "dns.google",
    "cloudflare-dns.com",
    "dns.quad9.net",
    "one.one.one.one",
    "1.1.1.1",
    "8.8.8.8",
    "9.9.9.9",
];
static NETWORK_CONTROLLER: OnceLock<Arc<tokio::sync::Mutex<Option<NetworkController>>>> =
    OnceLock::new();

pub struct NetworkServicesHandle {
    active: bool,
}

impl NetworkServicesHandle {
    pub async fn shutdown(self) {
        if !self.active {
            return;
        }
        if let Some(controller) = NETWORK_CONTROLLER.get() {
            let mut guard = controller.lock().await;
            if let Some(mut state) = guard.take() {
                if let Some(handle) = state.envoy.take() {
                    handle.shutdown().await;
                }
                if let Some(handle) = state.coredns.take() {
                    handle.shutdown().await;
                }
                if let Some(handle) = state.firewall.take() {
                    handle.shutdown();
                }
            }
        }
    }
}

struct ManagedProcessHandle {
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
    service_firewall: Option<firewall::ServiceProcessFirewallHandle>,
}

impl ManagedProcessHandle {
    async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
        if let Some(handle) = self.service_firewall.take() {
            handle.shutdown();
        }
    }
}

struct NetworkController {
    config: Config,
    firewall: Option<firewall::FirewallHandle>,
    coredns: Option<ManagedProcessHandle>,
    envoy: Option<ManagedProcessHandle>,
    vm_proxy_policies: BTreeMap<String, VmProxyListener>,
    vm_guardrails: BTreeMap<String, VmGuardrail>,
    vm_counters: VmCounters,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VmProxyPolicyConfig {
    #[serde(default)]
    pub nym_id: Option<String>,
    #[serde(default)]
    pub domain_allowlist: Option<Vec<String>>,
    #[serde(default)]
    pub domain_blocklist: Vec<String>,
    #[serde(default)]
    pub custom_port_allowlist: Vec<u16>,
    pub bandwidth_cap_mb_per_hour: u32,
    pub max_connections_per_minute: u32,
}

#[derive(Clone, Debug)]
struct VmProxyListener {
    listen_addr: SocketAddr,
    policy: VmProxyPolicyConfig,
}

#[derive(Clone, Debug)]
struct VmGuardrail {
    pid: Option<u32>,
    cgroup_path: Option<String>,
    max_connections_per_minute: u32,
    bandwidth_cap_mb_per_hour: u32,
    blocked_until: Option<DateTime<Utc>>,
    last_conn_counter: Option<u64>,
    last_byte_counter: Option<u64>,
    conn_deltas: VecDeque<CounterDeltaEvent>,
    byte_deltas: VecDeque<CounterDeltaEvent>,
}

struct EnvoyRuntimeConfig {
    default_listen_addr: Option<SocketAddr>,
    probe_addr: SocketAddr,
    dns_resolver_addr: SocketAddr,
    admin_addr: SocketAddr,
}

struct EnvoyRuntimePaths {
    work_dir: PathBuf,
    config_path: PathBuf,
    access_log_path: PathBuf,
    process_log_path: PathBuf,
}

#[derive(Clone, Debug)]
struct CounterDeltaEvent {
    occurred_at: DateTime<Utc>,
    value: u64,
}

pub async fn start(config: &Config) -> Result<Option<NetworkServicesHandle>> {
    let firewall = Some(firewall::install(config).context("install qemu guest firewall")?);
    let coredns = if needs_coredns(config) {
        Some(start_coredns(config).await?)
    } else {
        None
    };
    let envoy = if config.guest_network.http_proxy_upstream_addr.is_some() {
        Some(start_envoy(config, &BTreeMap::new()).await?)
    } else {
        None
    };

    if firewall.is_none() && coredns.is_none() && envoy.is_none() {
        return Ok(None);
    }

    let controller = NETWORK_CONTROLLER
        .get_or_init(|| Arc::new(tokio::sync::Mutex::new(None)))
        .clone();
    let mut guard = controller.lock().await;
    let access_log_path = Path::new(&config.data_dir)
        .join("_network")
        .join("envoy")
        .join("access.log");
    *guard = Some(NetworkController {
        config: config.clone(),
        firewall,
        coredns,
        envoy,
        vm_proxy_policies: BTreeMap::new(),
        vm_guardrails: BTreeMap::new(),
        vm_counters: VmCounters::new(access_log_path),
    });
    drop(guard);
    spawn_limit_enforcer(controller);

    Ok(Some(NetworkServicesHandle { active: true }))
}

fn needs_coredns(config: &Config) -> bool {
    config.guest_network.dns_server.is_some()
        || config.guest_network.http_proxy_upstream_addr.is_some()
}

async fn start_coredns(config: &Config) -> Result<ManagedProcessHandle> {
    let bind_addr = config
        .network_services
        .coredns_bind_addr
        .parse::<SocketAddr>()
        .with_context(|| {
            format!(
                "parse coredns bind addr {}",
                config.network_services.coredns_bind_addr
            )
        })?;

    let work_dir = Path::new(&config.data_dir).join("_network").join("coredns");
    tokio::fs::create_dir_all(&work_dir)
        .await
        .with_context(|| format!("create coredns work dir {}", work_dir.display()))?;

    let config_path = work_dir.join("Corefile");
    let threat_hosts_path = resolve_coredns_threat_hosts_path(config, &work_dir).await?;
    let process_log_path = work_dir.join("process.log");
    write_coredns_config(&config_path, bind_addr, &threat_hosts_path).await?;

    let process_log = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&process_log_path)
        .await
        .with_context(|| format!("open coredns process log {}", process_log_path.display()))?;
    let process_log_err = process_log
        .try_clone()
        .await
        .with_context(|| format!("clone coredns process log {}", process_log_path.display()))?;

    let mut child = Command::new(&config.network_services.coredns_bin);
    child
        .arg("-conf")
        .arg(&config_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(process_log.into_std().await))
        .stderr(Stdio::from(process_log_err.into_std().await));

    let mut child = child.spawn().with_context(|| {
        format!(
            "spawn coredns binary {}",
            config.network_services.coredns_bin
        )
    })?;

    wait_for_listener(
        bind_addr,
        &mut child,
        "coredns",
        COREDNS_READY_TIMEOUT,
        COREDNS_READY_POLL_INTERVAL,
    )
    .await?;

    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        tokio::select! {
            _ = &mut stop_rx => {
                if let Some(pid) = child.id() {
                    debug!(pid, "stopping coredns");
                }
                if let Err(err) = child.start_kill() {
                    warn!(error = %err, "failed to signal coredns for shutdown");
                }
                if let Err(err) = child.wait().await {
                    warn!(error = %err, "failed waiting for coredns shutdown");
                }
            }
            status = child.wait() => {
                match status {
                    Ok(status) => warn!(?status, "coredns exited"),
                    Err(err) => warn!(error = %err, "coredns wait failed"),
                }
            }
        }
    });

    Ok(ManagedProcessHandle {
        stop_tx: Some(stop_tx),
        join: Some(join),
        service_firewall: None,
    })
}

async fn start_envoy(
    config: &Config,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
) -> Result<ManagedProcessHandle> {
    let runtime = envoy_runtime_config(config, vm_proxy_policies)?;
    let paths = envoy_runtime_paths(config);
    tokio::fs::create_dir_all(&paths.work_dir)
        .await
        .with_context(|| format!("create envoy work dir {}", paths.work_dir.display()))?;

    write_envoy_config(
        &paths.config_path,
        runtime.default_listen_addr,
        vm_proxy_policies,
        runtime.admin_addr,
        runtime.dns_resolver_addr,
        &paths.access_log_path,
    )
    .await?;

    let process_log = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&paths.process_log_path)
        .await
        .with_context(|| {
            format!(
                "open envoy process log {}",
                paths.process_log_path.display()
            )
        })?;
    let process_log_err = process_log.try_clone().await.with_context(|| {
        format!(
            "clone envoy process log {}",
            paths.process_log_path.display()
        )
    })?;

    let mut service_cgroup = if config.ha_mode {
        Some(
            firewall::prepare_service_process_cgroup("envoy")?
                .context("envoy service cgroup is required in ha mode")?,
        )
    } else {
        None
    };

    let mut child = Command::new(&config.network_services.envoy_bin);
    child
        .arg("-c")
        .arg(&paths.config_path)
        .arg("--log-level")
        .arg(&config.network_services.envoy_log_level)
        .stdin(Stdio::null())
        .stdout(Stdio::from(process_log.into_std().await))
        .stderr(Stdio::from(process_log_err.into_std().await));

    let mut service_firewall = None;
    if let Some(cgroup) = service_cgroup.take() {
        if let Err(err) = configure_child_cgroup(&mut child, cgroup.cgroup_procs_path()) {
            cgroup.cleanup();
            return Err(err);
        }
        service_firewall = Some(firewall::protect_service_cgroup_private_egress(
            cgroup,
            Some(runtime.dns_resolver_addr),
        )?);
    }

    let mut child = match child
        .spawn()
        .with_context(|| format!("spawn envoy binary {}", config.network_services.envoy_bin))
    {
        Ok(child) => child,
        Err(err) => {
            if let Some(handle) = service_firewall.take() {
                handle.shutdown();
            }
            return Err(err);
        }
    };
    if service_firewall.is_some() {
        if let Some(pid) = child.id() {
            debug!(pid, "installed envoy cgroup private-egress firewall rules");
        }
    }

    if let Err(err) = wait_for_listener(
        runtime.probe_addr,
        &mut child,
        "envoy",
        ENVOY_READY_TIMEOUT,
        ENVOY_READY_POLL_INTERVAL,
    )
    .await
    {
        let _ = child.start_kill();
        let _ = child.wait().await;
        if let Some(handle) = service_firewall.take() {
            handle.shutdown();
        }
        return Err(err);
    }

    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        tokio::select! {
            _ = &mut stop_rx => {
                if let Some(pid) = child.id() {
                    debug!(pid, "stopping envoy");
                }
                if let Err(err) = child.start_kill() {
                    warn!(error = %err, "failed to signal envoy for shutdown");
                }
                if let Err(err) = child.wait().await {
                    warn!(error = %err, "failed waiting for envoy shutdown");
                }
            }
            status = child.wait() => {
                match status {
                    Ok(status) => warn!(?status, "envoy exited"),
                    Err(err) => warn!(error = %err, "envoy wait failed"),
                }
            }
        }
    });

    Ok(ManagedProcessHandle {
        stop_tx: Some(stop_tx),
        join: Some(join),
        service_firewall,
    })
}

#[cfg(target_os = "linux")]
fn configure_child_cgroup(command: &mut Command, cgroup_procs_path: &Path) -> Result<()> {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;

    let path = CString::new(cgroup_procs_path.as_os_str().as_bytes()).with_context(|| {
        format!(
            "convert cgroup.procs path to cstring {}",
            cgroup_procs_path.display()
        )
    })?;
    unsafe {
        command.pre_exec(move || {
            let fd = libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC);
            if fd < 0 {
                return Err(std::io::Error::last_os_error());
            }

            let mut buf = [0_u8; 32];
            let len = encode_pid_line(libc::getpid() as u64, &mut buf);
            let mut offset = 0;
            while offset < len {
                let written = libc::write(
                    fd,
                    buf[offset..len].as_ptr().cast(),
                    (len - offset) as libc::size_t,
                );
                if written < 0 {
                    let err = std::io::Error::last_os_error();
                    let _ = libc::close(fd);
                    return Err(err);
                }
                offset += written as usize;
            }

            if libc::close(fd) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn encode_pid_line(mut pid: u64, buf: &mut [u8; 32]) -> usize {
    let mut digits = [0_u8; 20];
    let mut len = 0;
    if pid == 0 {
        digits[len] = b'0';
        len += 1;
    } else {
        while pid > 0 {
            digits[len] = b'0' + (pid % 10) as u8;
            pid /= 10;
            len += 1;
        }
    }
    for index in 0..len {
        buf[index] = digits[len - index - 1];
    }
    buf[len] = b'\n';
    len + 1
}

#[cfg(not(target_os = "linux"))]
fn configure_child_cgroup(_command: &mut Command, _cgroup_procs_path: &Path) -> Result<()> {
    Ok(())
}

async fn wait_for_listener(
    addr: SocketAddr,
    child: &mut tokio::process::Child,
    label: &str,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<()> {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().context("poll child readiness")? {
            bail!("{label} exited before becoming ready: {status}");
        }
        if TcpStream::connect(addr).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(poll_interval).await;
    }
    bail!("{label} did not become ready on {addr} within 5s");
}

fn envoy_runtime_config(
    config: &Config,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
) -> Result<EnvoyRuntimeConfig> {
    let default_listen_addr = config
        .guest_network
        .http_proxy_upstream_addr
        .as_deref()
        .map(|listen_addr| {
            listen_addr
                .parse::<SocketAddr>()
                .with_context(|| format!("parse envoy listen addr {listen_addr}"))
        })
        .transpose()?;
    let probe_addr = default_listen_addr
        .or_else(|| {
            vm_proxy_policies
                .values()
                .next()
                .map(|listener| listener.listen_addr)
        })
        .ok_or_else(|| anyhow!("at least one envoy listener is required"))?;
    let dns_resolver_addr = config
        .network_services
        .coredns_bind_addr
        .parse::<SocketAddr>()
        .with_context(|| {
            format!(
                "parse envoy dns resolver addr {}",
                config.network_services.coredns_bind_addr
            )
        })?;
    let admin_addr = config
        .network_services
        .envoy_admin_addr
        .parse::<SocketAddr>()
        .with_context(|| {
            format!(
                "parse envoy admin addr {}",
                config.network_services.envoy_admin_addr
            )
        })?;

    Ok(EnvoyRuntimeConfig {
        default_listen_addr,
        probe_addr,
        dns_resolver_addr,
        admin_addr,
    })
}

fn envoy_runtime_paths(config: &Config) -> EnvoyRuntimePaths {
    let work_dir = Path::new(&config.data_dir).join("_network").join("envoy");
    EnvoyRuntimePaths {
        config_path: work_dir.join("envoy.yaml"),
        access_log_path: work_dir.join("access.log"),
        process_log_path: work_dir.join("process.log"),
        work_dir,
    }
}

fn should_run_envoy(
    config: &Config,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
) -> bool {
    config.guest_network.http_proxy_upstream_addr.is_some() || !vm_proxy_policies.is_empty()
}

async fn validate_envoy_candidate_config(
    config: &Config,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
) -> Result<()> {
    let runtime = envoy_runtime_config(config, vm_proxy_policies)?;
    let paths = envoy_runtime_paths(config);
    tokio::fs::create_dir_all(&paths.work_dir)
        .await
        .with_context(|| format!("create envoy work dir {}", paths.work_dir.display()))?;
    let validation_path = paths
        .work_dir
        .join(format!(".envoy-validate-{}.yaml", Uuid::new_v4()));

    let result = async {
        write_envoy_config(
            &validation_path,
            runtime.default_listen_addr,
            vm_proxy_policies,
            runtime.admin_addr,
            runtime.dns_resolver_addr,
            &paths.access_log_path,
        )
        .await?;

        let output = Command::new(&config.network_services.envoy_bin)
            .arg("--mode")
            .arg("validate")
            .arg("-c")
            .arg(&validation_path)
            .arg("--log-level")
            .arg(&config.network_services.envoy_log_level)
            .stdin(Stdio::null())
            .output()
            .await
            .with_context(|| {
                format!(
                    "validate envoy config with {}",
                    config.network_services.envoy_bin
                )
            })?;
        if !output.status.success() {
            bail!(
                "envoy config validation failed: status={} stderr={} stdout={}",
                output.status,
                process_output_excerpt(&output.stderr),
                process_output_excerpt(&output.stdout)
            );
        }
        Ok(())
    }
    .await;

    let _ = tokio::fs::remove_file(&validation_path).await;
    result
}

fn process_output_excerpt(bytes: &[u8]) -> String {
    const MAX_CHARS: usize = 4096;
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    if trimmed.chars().count() <= MAX_CHARS {
        return trimmed.to_string();
    }
    let mut excerpt = trimmed.chars().take(MAX_CHARS).collect::<String>();
    excerpt.push_str("...<truncated>");
    excerpt
}

async fn write_coredns_config(
    path: &Path,
    bind_addr: SocketAddr,
    threat_hosts_path: &Path,
) -> Result<()> {
    let contents = render_coredns_config(bind_addr, threat_hosts_path);
    write_text_file_atomic(path, &contents, "coredns config").await
}

async fn write_coredns_threat_hosts(path: &Path) -> Result<()> {
    let contents = FALLBACK_THREAT_HOSTS
        .iter()
        .map(|host| format!("0.0.0.0 {host}\n"))
        .collect::<String>();
    write_text_file_atomic(path, &contents, "coredns threat hosts").await
}

async fn resolve_coredns_threat_hosts_path(
    config: &Config,
    work_dir: &Path,
) -> Result<std::path::PathBuf> {
    let configured_path = Path::new(&config.network_services.coredns_threat_hosts_path);
    if configured_path.exists() {
        return Ok(configured_path.to_path_buf());
    }

    let fallback_path = work_dir.join("threat-domains.hosts");
    write_coredns_threat_hosts(&fallback_path).await?;
    Ok(fallback_path)
}

async fn write_envoy_config(
    path: &Path,
    default_listen_addr: Option<SocketAddr>,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
    admin_addr: SocketAddr,
    dns_resolver_addr: SocketAddr,
    access_log_path: &Path,
) -> Result<()> {
    let contents = render_envoy_config(
        default_listen_addr,
        vm_proxy_policies,
        admin_addr,
        dns_resolver_addr,
        access_log_path,
    );
    write_text_file_atomic(path, &contents, "envoy config").await
}

async fn write_text_file_atomic(path: &Path, contents: &str, label: &str) -> Result<()> {
    let parent = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("config");
    let tmp_path = parent.join(format!(".{file_name}.{}.tmp", Uuid::new_v4()));
    let write_result = async {
        let mut file = tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp_path)
            .await
            .with_context(|| format!("create temporary {label} {}", tmp_path.display()))?;
        file.write_all(contents.as_bytes())
            .await
            .with_context(|| format!("write temporary {label} {}", tmp_path.display()))?;
        file.flush()
            .await
            .with_context(|| format!("flush temporary {label} {}", tmp_path.display()))?;
        drop(file);
        tokio::fs::rename(&tmp_path, path)
            .await
            .with_context(|| format!("publish {label} {}", path.display()))?;
        Ok(())
    }
    .await;
    if write_result.is_err() {
        let _ = tokio::fs::remove_file(&tmp_path).await;
    }
    write_result
}

fn render_coredns_config(bind_addr: SocketAddr, threat_hosts_path: &Path) -> String {
    format!(
        r#".:{port} {{
    bind {ip}
    hosts {threat_hosts_path} {{
        fallthrough
    }}
    forward . 1.1.1.1 8.8.8.8
    cache 30
    errors
    log
    reload
}}
"#,
        ip = bind_addr.ip(),
        port = bind_addr.port(),
        threat_hosts_path = threat_hosts_path.display(),
    )
}

fn render_envoy_config(
    default_listen_addr: Option<SocketAddr>,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
    admin_addr: SocketAddr,
    dns_resolver_addr: SocketAddr,
    access_log_path: &Path,
) -> String {
    let admin_ip = admin_addr.ip();
    let dns_resolver_ip = dns_resolver_addr.ip();
    let access_log = access_log_path.display();
    let mut listeners = String::new();
    if let Some(listen_addr) = default_listen_addr {
        listeners.push_str(&render_envoy_listener_block(
            "listener_http_proxy",
            listen_addr,
            None,
            None,
            None,
            dns_resolver_addr,
            access_log.to_string().as_str(),
        ));
    }
    for (vm_id, listener) in vm_proxy_policies {
        listeners.push_str(&render_envoy_listener_block(
            format!("listener_vm_{}", sanitize_listener_name(vm_id)).as_str(),
            listener.listen_addr,
            Some(listener.policy.domain_blocklist.as_slice()),
            Some(&listener.policy),
            Some(vm_id.as_str()),
            dns_resolver_addr,
            access_log.to_string().as_str(),
        ));
    }
    format!(
        r#"admin:
  access_log_path: {access_log}
  address:
    socket_address:
      address: {admin_ip}
      port_value: {admin_port}
static_resources:
  listeners:
{listeners}
  clusters:
  - name: {cluster_name}
    connect_timeout: 10s
    lb_policy: CLUSTER_PROVIDED
    cluster_type:
      name: envoy.clusters.dynamic_forward_proxy
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.clusters.dynamic_forward_proxy.v3.ClusterConfig
        dns_cache_config:
          name: {dns_cache_name}
          dns_lookup_family: V4_ONLY
          typed_dns_resolver_config:
            name: envoy.network.dns_resolver.cares
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.network.dns_resolver.cares.v3.CaresDnsResolverConfig
              resolvers:
              - socket_address:
                  address: {dns_resolver_ip}
                  port_value: {dns_resolver_port}
              dns_resolver_options:
                use_tcp_for_dns_lookups: true
                no_default_search_domain: true
"#,
        access_log = access_log,
        admin_ip = admin_ip,
        admin_port = admin_addr.port(),
        dns_resolver_ip = dns_resolver_ip,
        dns_resolver_port = dns_resolver_addr.port(),
        listeners = listeners,
        cluster_name = ENVOY_CLUSTER_NAME,
        dns_cache_name = ENVOY_DNS_CACHE_NAME,
    )
}

fn render_envoy_listener_block(
    name: &str,
    listen_addr: SocketAddr,
    domain_blocklist: Option<&[String]>,
    policy: Option<&VmProxyPolicyConfig>,
    vm_id: Option<&str>,
    dns_resolver_addr: SocketAddr,
    access_log_path: &str,
) -> String {
    let additional_block_regex = domain_blocklist
        .filter(|entries| !entries.is_empty())
        .map(|entries| domain_list_regex(entries, None))
        .unwrap_or_default();
    let allow_regex = policy.map(|policy| {
        let mut allowed_ports = vec![80_u16, 443_u16];
        allowed_ports.extend(policy.custom_port_allowlist.iter().copied());
        allowed_ports.sort_unstable();
        allowed_ports.dedup();
        authority_allow_regex(policy.domain_allowlist.as_deref(), &allowed_ports)
    });
    let additional_block_regex = escape_yaml_double_quoted(&additional_block_regex);
    let allow_regex = allow_regex.as_deref().map(escape_yaml_double_quoted);
    let system_block_routes = render_envoy_system_block_routes();
    let allow_routes = if let Some(allow_regex) = allow_regex {
        format!(
            r#"              - match:
                  connect_matcher: {{}}
                  headers:
                  - name: ":authority"
                    string_match:
                      safe_regex:
                        regex: "{allow_regex}"
                route:
                  cluster: {cluster_name}
                  upgrade_configs:
                  - upgrade_type: CONNECT
                    connect_config: {{}}
              - match:
                  prefix: "/"
                  headers:
                  - name: ":authority"
                    string_match:
                      safe_regex:
                        regex: "{allow_regex}"
                route:
                  cluster: {cluster_name}
                  timeout: 0s
              - match:
                  connect_matcher: {{}}
                direct_response:
                  status: 403
              - match:
                  prefix: "/"
                direct_response:
                  status: 403
"#,
            allow_regex = allow_regex,
            cluster_name = ENVOY_CLUSTER_NAME,
        )
    } else {
        format!(
            r#"              - match:
                  connect_matcher: {{}}
                route:
                  cluster: {cluster_name}
                  upgrade_configs:
                  - upgrade_type: CONNECT
                    connect_config: {{}}
              - match:
                  prefix: "/"
                route:
                  cluster: {cluster_name}
                  timeout: 0s
"#,
            cluster_name = ENVOY_CLUSTER_NAME,
        )
    };
    let blocklist_routes = if additional_block_regex.is_empty() {
        String::new()
    } else {
        format!(
            r#"              - match:
                  connect_matcher: {{}}
                  headers:
                  - name: ":authority"
                    string_match:
                      safe_regex:
                        regex: "{additional_block_regex}"
                direct_response:
                  status: 403
              - match:
                  prefix: "/"
                  headers:
                  - name: ":authority"
                    string_match:
                      safe_regex:
                        regex: "{additional_block_regex}"
                direct_response:
                  status: 403
"#
        )
    };
    format!(
        r#"  - name: {name}
    address:
      socket_address:
        address: {listen_ip}
        port_value: {listen_port}
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          stat_prefix: {name}
          access_log:
          - name: envoy.access_loggers.file
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog
              path: {access_log_path}
              log_format:
                json_format:
                  timestamp: "%START_TIME(%Y-%m-%dT%H:%M:%S.%3fZ)%"
                  vm_id: "{vm_id}"
                  nym_id: "{nym_id}"
                  listener: "{name}"
                  authority: "%REQ(:AUTHORITY)%"
                  method: "%REQ(:METHOD)%"
                  path: "%REQ(X-ENVOY-ORIGINAL-PATH?:PATH)%"
                  response_code: "%RESPONSE_CODE%"
                  response_code_details: "%RESPONSE_CODE_DETAILS%"
                  bytes_received: "%BYTES_RECEIVED%"
                  bytes_sent: "%BYTES_SENT%"
                  requested_server_name: "%REQUESTED_SERVER_NAME%"
                  upstream_host: "%UPSTREAM_HOST%"
          route_config:
            name: {name}_route
            virtual_hosts:
            - name: {name}_proxy
              domains: ["*"]
              routes:
{system_block_routes}{blocklist_routes}{allow_routes}          http_filters:
          - name: envoy.filters.http.dynamic_forward_proxy
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.dynamic_forward_proxy.v3.FilterConfig
              dns_cache_config:
                name: {dns_cache_name}
                dns_lookup_family: V4_ONLY
                typed_dns_resolver_config:
                  name: envoy.network.dns_resolver.cares
                  typed_config:
                    "@type": type.googleapis.com/envoy.extensions.network.dns_resolver.cares.v3.CaresDnsResolverConfig
                    resolvers:
                    - socket_address:
                        address: {dns_resolver_ip}
                        port_value: {dns_resolver_port}
                    dns_resolver_options:
                      use_tcp_for_dns_lookups: true
                      no_default_search_domain: true
          - name: envoy.filters.http.router
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
"#,
        name = name,
        listen_ip = listen_addr.ip(),
        listen_port = listen_addr.port(),
        vm_id = vm_id.unwrap_or_default(),
        nym_id = policy
            .and_then(|value| value.nym_id.as_deref())
            .unwrap_or_default(),
        access_log_path = access_log_path,
        system_block_routes = system_block_routes,
        blocklist_routes = blocklist_routes,
        allow_routes = allow_routes,
        dns_cache_name = ENVOY_DNS_CACHE_NAME,
        dns_resolver_ip = dns_resolver_addr.ip(),
        dns_resolver_port = dns_resolver_addr.port(),
    )
}

fn authority_allow_regex(domain_allowlist: Option<&[String]>, allowed_ports: &[u16]) -> String {
    let host_regex = match domain_allowlist {
        Some(domains) => {
            if domains.is_empty() {
                String::from("__nym_no_hosts_allowed__")
            } else {
                domains
                    .iter()
                    .map(|domain| format!(r"(.*\.)?{}", escape_regex_literal(domain)))
                    .collect::<Vec<_>>()
                    .join("|")
            }
        }
        _ => String::from(r"[^:]+"),
    };
    let ports_regex = allowed_ports
        .iter()
        .map(|port| port.to_string())
        .collect::<Vec<_>>()
        .join("|");
    format!(r"(?i)^({host_regex})(?::({ports_regex}))?$")
}

fn domain_list_regex(domains: &[String], allowed_ports: Option<&[u16]>) -> String {
    let hosts = domains
        .iter()
        .map(|domain| format!(r"(.*\.)?{}", escape_regex_literal(domain)))
        .collect::<Vec<_>>()
        .join("|");
    match allowed_ports {
        Some(ports) if !ports.is_empty() => {
            let ports = ports
                .iter()
                .map(|port| port.to_string())
                .collect::<Vec<_>>()
                .join("|");
            format!(r"(?i)^({hosts})(?::({ports}))?$")
        }
        _ => format!(r"(?i)^({hosts})(?::\d+)?$"),
    }
}

fn escape_regex_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '^' | '$' | '|' | '\\' => {
                escaped.push('\\');
                escaped.push(ch);
            }
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn escape_yaml_double_quoted(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn render_envoy_system_block_routes() -> String {
    [
        r"(?i)^10\..*",
        r"(?i)^172\.(1[6-9]|2[0-9]|3[0-1])\..*",
        r"(?i)^192\.168\..*",
        r"(?i)^100\.(6[4-9]|[7-9][0-9]|1[0-1][0-9]|12[0-7])\..*",
        r"(?i)^(.*\.)?svc\.cluster\.local(?::\d+)?$",
        r"(?i)^(dns\.google|cloudflare-dns\.com|dns\.quad9\.net|one\.one\.one\.one)(?::\d+)?$",
        r"(?i)^.*:(25|465|587|6667|6697)$",
    ]
    .into_iter()
    .map(render_envoy_regex_deny_routes)
    .collect()
}

fn render_envoy_regex_deny_routes(regex: &str) -> String {
    let regex = escape_yaml_double_quoted(regex);
    format!(
        r#"              - match:
                  connect_matcher: {{}}
                  headers:
                  - name: ":authority"
                    string_match:
                      safe_regex:
                        regex: "{regex}"
                direct_response:
                  status: 403
              - match:
                  prefix: "/"
                  headers:
                  - name: ":authority"
                    string_match:
                      safe_regex:
                        regex: "{regex}"
                direct_response:
                  status: 403
"#
    )
}

fn sanitize_listener_name(vm_id: &str) -> String {
    vm_id
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn spawn_limit_enforcer(controller: Arc<tokio::sync::Mutex<Option<NetworkController>>>) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(LIMIT_ENFORCER_POLL_INTERVAL);
        loop {
            ticker.tick().await;
            let mut guard = controller.lock().await;
            let Some(state) = guard.as_mut() else {
                break;
            };
            if let Err(err) = enforce_vm_guardrails(state).await {
                warn!(error = %err, "failed enforcing vm proxy guardrails");
            }
        }
    });
}

async fn enforce_vm_guardrails(state: &mut NetworkController) -> Result<()> {
    state.vm_counters.refresh()?;
    let now = Utc::now();
    let vm_ids = state.vm_guardrails.keys().cloned().collect::<Vec<_>>();
    for vm_id in vm_ids {
        let limit_snapshot = state.vm_counters.limit_snapshot(&vm_id, now)?;
        let Some(guardrail) = state.vm_guardrails.get_mut(&vm_id) else {
            continue;
        };
        let Some(pid) = guardrail.pid else {
            continue;
        };
        let Some(limit_snapshot) = limit_snapshot else {
            if guardrail.blocked_until.is_some() {
                firewall::unblock_vm_process(&vm_id, pid, guardrail.cgroup_path.as_deref())?;
                guardrail.blocked_until = None;
            }
            continue;
        };
        let combined_snapshot =
            if let Some(kernel_snapshot) = firewall::vm_counter_snapshot(&vm_id)? {
                guardrail.update_kernel_counters(now, kernel_snapshot);
                combine_limit_snapshots(&limit_snapshot, &guardrail.kernel_limit_snapshot(now))
            } else {
                guardrail.prune_counter_windows(now);
                limit_snapshot
            };

        let next_block_deadline = next_guardrail_block_deadline(guardrail, &combined_snapshot);
        match next_block_deadline {
            Some(until) if until > now => {
                let was_blocked = guardrail.blocked_until.is_some();
                firewall::block_vm_process(&vm_id, pid, guardrail.cgroup_path.as_deref())?;
                guardrail.blocked_until = Some(until);
                if !was_blocked {
                    state.vm_counters.record_guardrail_event(
                        &vm_id,
                        "blocked",
                        guardrail_reason(guardrail, &combined_snapshot).as_str(),
                        now,
                    );
                }
            }
            _ => {
                if guardrail.blocked_until.is_some() {
                    firewall::unblock_vm_process(&vm_id, pid, guardrail.cgroup_path.as_deref())?;
                    state.vm_counters.record_guardrail_event(
                        &vm_id,
                        "unblocked",
                        "guardrail window reset",
                        now,
                    );
                }
                guardrail.blocked_until = None;
            }
        }
    }
    Ok(())
}

fn next_guardrail_block_deadline(
    guardrail: &VmGuardrail,
    snapshot: &VmProxyLimitSnapshot,
) -> Option<DateTime<Utc>> {
    let connection_exceeded =
        snapshot.connection_attempts_last_minute > guardrail.max_connections_per_minute as u64;
    let bandwidth_exceeded =
        snapshot.allowed_bytes_last_hour > guardrail.bandwidth_cap_mb_per_hour as u64 * 1024 * 1024;

    match (connection_exceeded, bandwidth_exceeded) {
        (true, true) => match (
            snapshot.connection_window_resets_at,
            snapshot.bandwidth_window_resets_at,
        ) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        },
        (true, false) => snapshot.connection_window_resets_at,
        (false, true) => snapshot.bandwidth_window_resets_at,
        (false, false) => None,
    }
}

fn combine_limit_snapshots(
    proxy_snapshot: &VmProxyLimitSnapshot,
    kernel_snapshot: &VmProxyLimitSnapshot,
) -> VmProxyLimitSnapshot {
    VmProxyLimitSnapshot {
        connection_attempts_last_minute: proxy_snapshot
            .connection_attempts_last_minute
            .max(kernel_snapshot.connection_attempts_last_minute),
        allowed_bytes_last_hour: proxy_snapshot
            .allowed_bytes_last_hour
            .max(kernel_snapshot.allowed_bytes_last_hour),
        connection_window_resets_at: max_optional_datetime(
            proxy_snapshot.connection_window_resets_at,
            kernel_snapshot.connection_window_resets_at,
        ),
        bandwidth_window_resets_at: max_optional_datetime(
            proxy_snapshot.bandwidth_window_resets_at,
            kernel_snapshot.bandwidth_window_resets_at,
        ),
    }
}

fn apply_limit_snapshot(
    activity_snapshot: &mut VmProxyActivitySnapshot,
    limit_snapshot: &VmProxyLimitSnapshot,
    blocked_until: Option<DateTime<Utc>>,
    includes_kernel_counters: bool,
) {
    activity_snapshot.connection_attempts_last_minute =
        limit_snapshot.connection_attempts_last_minute;
    activity_snapshot.allowed_bytes_last_hour = limit_snapshot.allowed_bytes_last_hour;
    activity_snapshot.connection_window_resets_at = limit_snapshot.connection_window_resets_at;
    activity_snapshot.bandwidth_window_resets_at = limit_snapshot.bandwidth_window_resets_at;
    activity_snapshot.blocked_until = blocked_until;
    activity_snapshot.includes_kernel_counters = includes_kernel_counters;
}

fn limit_snapshot_has_activity(limit_snapshot: &VmProxyLimitSnapshot) -> bool {
    limit_snapshot.connection_attempts_last_minute > 0
        || limit_snapshot.allowed_bytes_last_hour > 0
        || limit_snapshot.connection_window_resets_at.is_some()
        || limit_snapshot.bandwidth_window_resets_at.is_some()
}

fn empty_activity_snapshot() -> VmProxyActivitySnapshot {
    VmProxyActivitySnapshot {
        connection_attempts: 0,
        proxied_requests: 0,
        denied_requests: 0,
        allowed_bytes: 0,
        denied_bytes: 0,
        connection_attempts_last_minute: 0,
        allowed_bytes_last_hour: 0,
        connection_window_resets_at: None,
        bandwidth_window_resets_at: None,
        blocked_until: None,
        includes_kernel_counters: false,
        updated_at: None,
        recent_events: Vec::new(),
    }
}

fn max_optional_datetime(
    left: Option<DateTime<Utc>>,
    right: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn guardrail_reason(guardrail: &VmGuardrail, snapshot: &VmProxyLimitSnapshot) -> String {
    let connection_exceeded =
        snapshot.connection_attempts_last_minute > guardrail.max_connections_per_minute as u64;
    let bandwidth_exceeded =
        snapshot.allowed_bytes_last_hour > guardrail.bandwidth_cap_mb_per_hour as u64 * 1024 * 1024;
    match (connection_exceeded, bandwidth_exceeded) {
        (true, true) => format!(
            "connections/min={} > {} and bytes/hour={} > {}MB",
            snapshot.connection_attempts_last_minute,
            guardrail.max_connections_per_minute,
            snapshot.allowed_bytes_last_hour,
            guardrail.bandwidth_cap_mb_per_hour
        ),
        (true, false) => format!(
            "connections/min={} > {}",
            snapshot.connection_attempts_last_minute, guardrail.max_connections_per_minute
        ),
        (false, true) => format!(
            "bytes/hour={} > {}MB",
            snapshot.allowed_bytes_last_hour, guardrail.bandwidth_cap_mb_per_hour
        ),
        (false, false) => "within limits".to_string(),
    }
}

impl VmGuardrail {
    fn update_kernel_counters(
        &mut self,
        now: DateTime<Utc>,
        snapshot: firewall::VmIptablesCounterSnapshot,
    ) {
        record_counter_delta(
            now,
            snapshot.new_connections,
            &mut self.last_conn_counter,
            &mut self.conn_deltas,
        );
        record_counter_delta(
            now,
            snapshot.total_bytes,
            &mut self.last_byte_counter,
            &mut self.byte_deltas,
        );
        self.prune_counter_windows(now);
    }

    fn kernel_limit_snapshot(&self, now: DateTime<Utc>) -> VmProxyLimitSnapshot {
        let connection_cutoff = now - chrono::Duration::seconds(60);
        let bandwidth_cutoff = now - chrono::Duration::seconds(60 * 60);
        let connection_attempts_last_minute = self
            .conn_deltas
            .iter()
            .filter(|event| event.occurred_at >= connection_cutoff)
            .map(|event| event.value)
            .sum();
        let allowed_bytes_last_hour = self
            .byte_deltas
            .iter()
            .filter(|event| event.occurred_at >= bandwidth_cutoff)
            .map(|event| event.value)
            .sum();
        let connection_window_resets_at = self
            .conn_deltas
            .iter()
            .find(|event| event.occurred_at >= connection_cutoff)
            .map(|event| event.occurred_at + chrono::Duration::seconds(60));
        let bandwidth_window_resets_at = self
            .byte_deltas
            .iter()
            .find(|event| event.occurred_at >= bandwidth_cutoff)
            .map(|event| event.occurred_at + chrono::Duration::seconds(60 * 60));
        VmProxyLimitSnapshot {
            connection_attempts_last_minute,
            allowed_bytes_last_hour,
            connection_window_resets_at,
            bandwidth_window_resets_at,
        }
    }

    fn prune_counter_windows(&mut self, now: DateTime<Utc>) {
        let connection_cutoff = now - chrono::Duration::seconds(60);
        while self
            .conn_deltas
            .front()
            .is_some_and(|event| event.occurred_at < connection_cutoff)
        {
            self.conn_deltas.pop_front();
        }

        let bandwidth_cutoff = now - chrono::Duration::seconds(60 * 60);
        while self
            .byte_deltas
            .front()
            .is_some_and(|event| event.occurred_at < bandwidth_cutoff)
        {
            self.byte_deltas.pop_front();
        }
    }
}

fn record_counter_delta(
    now: DateTime<Utc>,
    current: u64,
    last_counter: &mut Option<u64>,
    deltas: &mut VecDeque<CounterDeltaEvent>,
) {
    let delta = match *last_counter {
        Some(previous) if current >= previous => current - previous,
        Some(_) => current,
        None => 0,
    };
    *last_counter = Some(current);
    if delta > 0 {
        deltas.push_back(CounterDeltaEvent {
            occurred_at: now,
            value: delta,
        });
    }
}

pub async fn register_vm_proxy_policy(
    vm_id: &str,
    listen_addr: SocketAddr,
    policy: VmProxyPolicyConfig,
) -> Result<()> {
    let Some(controller) = NETWORK_CONTROLLER.get() else {
        return Ok(());
    };
    let mut guard = controller.lock().await;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    let max_connections_per_minute = policy.max_connections_per_minute;
    let bandwidth_cap_mb_per_hour = policy.bandwidth_cap_mb_per_hour;
    let mut next_policies = state.vm_proxy_policies.clone();
    next_policies.insert(
        vm_id.to_string(),
        VmProxyListener {
            listen_addr,
            policy,
        },
    );
    restart_envoy_locked(state, &next_policies).await?;
    state.vm_proxy_policies = next_policies;
    state
        .vm_guardrails
        .entry(vm_id.to_string())
        .and_modify(|guardrail| {
            guardrail.max_connections_per_minute = max_connections_per_minute;
            guardrail.bandwidth_cap_mb_per_hour = bandwidth_cap_mb_per_hour;
        })
        .or_insert(VmGuardrail {
            pid: None,
            cgroup_path: None,
            max_connections_per_minute,
            bandwidth_cap_mb_per_hour,
            blocked_until: None,
            last_conn_counter: None,
            last_byte_counter: None,
            conn_deltas: VecDeque::new(),
            byte_deltas: VecDeque::new(),
        });
    firewall::allow_proxy_listener(listen_addr)?;
    Ok(())
}

pub async fn register_vm_process(vm_id: &str, pid: u32) -> Result<()> {
    let Some(controller) = NETWORK_CONTROLLER.get() else {
        return Ok(());
    };
    let mut guard = controller.lock().await;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    if state.config.ha_mode
        && !state
            .firewall
            .as_ref()
            .is_some_and(firewall::FirewallHandle::is_installed)
    {
        bail!("network guardrails are required in ha mode but qemu firewall is not installed");
    }
    let cgroup_path = firewall::cgroup_path_for_pid(pid);
    if state.config.ha_mode && cgroup_path.is_none() {
        bail!("network guardrails are required in ha mode but qemu cgroup path is unavailable");
    }
    let entry = state
        .vm_guardrails
        .entry(vm_id.to_string())
        .or_insert(VmGuardrail {
            pid: Some(pid),
            cgroup_path: cgroup_path.clone(),
            max_connections_per_minute: 1000,
            bandwidth_cap_mb_per_hour: 1024,
            blocked_until: None,
            last_conn_counter: None,
            last_byte_counter: None,
            conn_deltas: VecDeque::new(),
            byte_deltas: VecDeque::new(),
        });
    if let Some(previous_pid) = entry.pid.replace(pid) {
        if previous_pid != pid {
            let _ = firewall::unblock_vm_process(vm_id, previous_pid, entry.cgroup_path.as_deref());
            let _ = firewall::remove_vm_counter_rules(
                vm_id,
                previous_pid,
                entry.cgroup_path.as_deref(),
            );
            entry.last_conn_counter = None;
            entry.last_byte_counter = None;
            entry.conn_deltas.clear();
            entry.byte_deltas.clear();
        }
    }
    entry.cgroup_path = cgroup_path;
    firewall::install_vm_counter_rules(vm_id, pid, entry.cgroup_path.as_deref())?;
    Ok(())
}

pub async fn unregister_vm_proxy_policy(vm_id: &str) -> Result<()> {
    let Some(controller) = NETWORK_CONTROLLER.get() else {
        return Ok(());
    };
    let mut guard = controller.lock().await;
    let Some(state) = guard.as_mut() else {
        return Ok(());
    };
    let removed_listener = state.vm_proxy_policies.get(vm_id).cloned();
    if removed_listener.is_some() {
        let mut next_policies = state.vm_proxy_policies.clone();
        next_policies.remove(vm_id);
        restart_envoy_locked(state, &next_policies).await?;
        state.vm_proxy_policies = next_policies;
    }
    let removed_guardrail = state.vm_guardrails.remove(vm_id);
    if let Some(guardrail) = &removed_guardrail {
        if let Some(pid) = guardrail.pid {
            firewall::unblock_vm_process(vm_id, pid, guardrail.cgroup_path.as_deref())?;
            firewall::remove_vm_counter_rules(vm_id, pid, guardrail.cgroup_path.as_deref())?;
        }
    }
    if let Some(listener) = removed_listener {
        firewall::revoke_proxy_listener(listener.listen_addr)?;
    }
    Ok(())
}

pub async fn vm_proxy_activity_snapshot(
    vm_id: &str,
    recent_limit: usize,
) -> Result<Option<VmProxyActivitySnapshot>> {
    let Some(controller) = NETWORK_CONTROLLER.get() else {
        return Ok(None);
    };
    let mut guard = controller.lock().await;
    let Some(state) = guard.as_mut() else {
        return Ok(None);
    };
    let now = Utc::now();
    let proxy_activity_snapshot = state.vm_counters.snapshot(vm_id, recent_limit)?;
    let proxy_limit_snapshot = state.vm_counters.limit_snapshot(vm_id, now)?;

    let guardrail = state.vm_guardrails.get_mut(vm_id);
    let (combined_limit_snapshot, blocked_until, includes_kernel_counters) =
        if let Some(guardrail) = guardrail {
            let combined_limit_snapshot =
                if let Some(kernel_snapshot) = firewall::vm_counter_snapshot(vm_id)? {
                    guardrail.update_kernel_counters(now, kernel_snapshot);
                    proxy_limit_snapshot
                        .as_ref()
                        .map(|proxy_snapshot| {
                            combine_limit_snapshots(
                                proxy_snapshot,
                                &guardrail.kernel_limit_snapshot(now),
                            )
                        })
                        .unwrap_or_else(|| guardrail.kernel_limit_snapshot(now))
                } else {
                    guardrail.prune_counter_windows(now);
                    proxy_limit_snapshot
                        .clone()
                        .unwrap_or_else(|| guardrail.kernel_limit_snapshot(now))
                };
            (
                Some(combined_limit_snapshot),
                guardrail.blocked_until,
                !guardrail.conn_deltas.is_empty() || !guardrail.byte_deltas.is_empty(),
            )
        } else {
            (proxy_limit_snapshot.clone(), None, false)
        };

    let mut activity_snapshot = match (proxy_activity_snapshot, combined_limit_snapshot.as_ref()) {
        (Some(snapshot), _) => snapshot,
        (None, Some(limit_snapshot))
            if limit_snapshot_has_activity(limit_snapshot) || blocked_until.is_some() =>
        {
            empty_activity_snapshot()
        }
        (None, None) => return Ok(None),
        (None, Some(_)) => return Ok(None),
    };

    if let Some(limit_snapshot) = combined_limit_snapshot.as_ref() {
        apply_limit_snapshot(
            &mut activity_snapshot,
            limit_snapshot,
            blocked_until,
            includes_kernel_counters,
        );
        if activity_snapshot.updated_at.is_none() {
            activity_snapshot.updated_at = Some(now);
        }
    }

    Ok(Some(activity_snapshot))
}

async fn restart_envoy_locked(
    state: &mut NetworkController,
    vm_proxy_policies: &BTreeMap<String, VmProxyListener>,
) -> Result<()> {
    let should_run = should_run_envoy(&state.config, vm_proxy_policies);
    if should_run && state.envoy.is_some() {
        validate_envoy_candidate_config(&state.config, vm_proxy_policies).await?;
    }
    if let Some(handle) = state.envoy.take() {
        handle.shutdown().await;
    }
    state.envoy = if should_run {
        Some(start_envoy(&state.config, vm_proxy_policies).await?)
    } else {
        None
    };
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vm_proxy_listener(listen_addr: &str) -> VmProxyListener {
        VmProxyListener {
            listen_addr: listen_addr.parse().expect("listen addr"),
            policy: VmProxyPolicyConfig {
                nym_id: Some("nym-123".to_string()),
                domain_allowlist: Some(vec!["api.github.com".to_string()]),
                domain_blocklist: vec!["bad.example".to_string()],
                custom_port_allowlist: vec![8080],
                bandwidth_cap_mb_per_hour: 1024,
                max_connections_per_minute: 1000,
            },
        }
    }

    #[test]
    fn should_run_envoy_when_default_proxy_or_vm_policy_exists() {
        let mut config = Config::default();
        config.guest_network.http_proxy_upstream_addr = None;
        let empty_policies = BTreeMap::new();
        assert!(!should_run_envoy(&config, &empty_policies));

        config.guest_network.http_proxy_upstream_addr = Some("127.0.0.1:3128".to_string());
        assert!(should_run_envoy(&config, &empty_policies));

        config.guest_network.http_proxy_upstream_addr = None;
        let mut policies = BTreeMap::new();
        policies.insert(
            "vm-123".to_string(),
            test_vm_proxy_listener("127.0.0.1:43128"),
        );
        assert!(should_run_envoy(&config, &policies));
    }

    #[test]
    fn envoy_runtime_config_uses_vm_listener_as_probe_without_default_proxy() {
        let mut config = Config::default();
        config.guest_network.http_proxy_upstream_addr = None;
        config.network_services.coredns_bind_addr = "127.0.0.53:53".to_string();
        config.network_services.envoy_admin_addr = "127.0.0.1:9901".to_string();
        let mut policies = BTreeMap::new();
        policies.insert(
            "vm-123".to_string(),
            test_vm_proxy_listener("127.0.0.1:43128"),
        );

        let runtime = envoy_runtime_config(&config, &policies).expect("runtime config");

        assert_eq!(runtime.default_listen_addr, None);
        assert_eq!(runtime.probe_addr, "127.0.0.1:43128".parse().unwrap());
        assert_eq!(runtime.dns_resolver_addr, "127.0.0.53:53".parse().unwrap());
        assert_eq!(runtime.admin_addr, "127.0.0.1:9901".parse().unwrap());
    }

    #[test]
    fn render_coredns_config_includes_bind_and_forwarders() {
        let rendered = render_coredns_config(
            "127.0.0.53:53".parse().expect("bind addr"),
            Path::new("/tmp/threat-domains.hosts"),
        );
        assert!(rendered.contains(".:53"));
        assert!(rendered.contains("bind 127.0.0.53"));
        assert!(rendered.contains("hosts /tmp/threat-domains.hosts"));
        assert!(rendered.contains("forward . 1.1.1.1 8.8.8.8"));
        assert!(rendered.contains("cache 30"));
    }

    #[test]
    fn render_envoy_config_includes_listener_admin_and_dns_cache() {
        let rendered = render_envoy_config(
            Some("127.0.0.1:3128".parse().expect("listen addr")),
            &BTreeMap::new(),
            "127.0.0.1:9901".parse().expect("admin addr"),
            "127.0.0.53:53".parse().expect("dns resolver addr"),
            Path::new("/tmp/envoy-access.log"),
        );
        assert!(rendered.contains("address: 127.0.0.1"));
        assert!(rendered.contains("port_value: 3128"));
        assert!(rendered.contains("port_value: 9901"));
        assert!(rendered.contains("address: 127.0.0.53"));
        assert!(rendered.contains("port_value: 53"));
        assert!(rendered.contains("connect_matcher"));
        assert!(rendered.contains("envoy.filters.http.dynamic_forward_proxy"));
        assert!(rendered.contains("envoy.clusters.dynamic_forward_proxy"));
        assert!(rendered.contains("/tmp/envoy-access.log"));
        assert!(rendered.contains(r#"svc\\.cluster\\.local"#));
        assert!(rendered.contains(r#"cloudflare-dns\\.com"#));
        assert!(rendered.contains("json_format"));
        assert!(rendered.contains("authority: \"%REQ(:AUTHORITY)%\""));
    }

    #[test]
    fn authority_allow_regex_includes_custom_ports_and_domains() {
        let regex = authority_allow_regex(Some(&["github.com".to_string()]), &[80, 443, 8080]);
        assert!(regex.contains("github\\.com"));
        assert!(regex.contains("8080"));
    }

    #[test]
    fn render_envoy_config_embeds_vm_listener_identity() {
        let mut listeners = BTreeMap::new();
        listeners.insert(
            "vm-123".to_string(),
            VmProxyListener {
                listen_addr: "127.0.0.1:43128".parse().expect("listen addr"),
                policy: VmProxyPolicyConfig {
                    nym_id: Some("nym-123".to_string()),
                    domain_allowlist: Some(vec!["api.github.com".to_string()]),
                    domain_blocklist: vec!["bad.example".to_string()],
                    custom_port_allowlist: vec![8080],
                    bandwidth_cap_mb_per_hour: 1024,
                    max_connections_per_minute: 1000,
                },
            },
        );
        let rendered = render_envoy_config(
            None,
            &listeners,
            "127.0.0.1:9901".parse().expect("admin addr"),
            "127.0.0.53:53".parse().expect("dns resolver addr"),
            Path::new("/tmp/envoy-access.log"),
        );
        assert!(rendered.contains("vm_id: \"vm-123\""));
        assert!(rendered.contains("nym_id: \"nym-123\""));
        assert!(rendered.contains(r#"bad\\.example"#));
        assert!(rendered.contains(r#"api\\.github\\.com"#));
        assert!(rendered.contains("8080"));
    }

    #[test]
    fn render_envoy_config_uses_custom_dns_resolver_for_vm_listener() {
        let mut listeners = BTreeMap::new();
        listeners.insert(
            "vm-123".to_string(),
            VmProxyListener {
                listen_addr: "127.0.0.1:43128".parse().expect("listen addr"),
                policy: VmProxyPolicyConfig {
                    nym_id: Some("nym-123".to_string()),
                    domain_allowlist: None,
                    domain_blocklist: Vec::new(),
                    custom_port_allowlist: Vec::new(),
                    bandwidth_cap_mb_per_hour: 1024,
                    max_connections_per_minute: 1000,
                },
            },
        );
        let rendered = render_envoy_config(
            None,
            &listeners,
            "127.0.0.1:9901".parse().expect("admin addr"),
            "127.0.0.77:5301".parse().expect("dns resolver addr"),
            Path::new("/tmp/envoy-access.log"),
        );
        assert!(rendered.contains("address: 127.0.0.77"));
        assert!(rendered.contains("port_value: 5301"));
        assert!(!rendered.contains("address: 127.0.0.53"));
    }

    #[test]
    fn render_envoy_config_escapes_regexes_for_yaml() {
        let rendered = render_envoy_config(
            Some("127.0.0.1:3128".parse().expect("listen addr")),
            &BTreeMap::new(),
            "127.0.0.1:9901".parse().expect("admin addr"),
            "127.0.0.53:53".parse().expect("dns resolver addr"),
            Path::new("/tmp/envoy-access.log"),
        );
        assert!(rendered.contains(r#"svc\\.cluster\\.local"#));
        assert!(rendered.contains(r#"cloudflare-dns\\.com"#));
        assert!(rendered.contains(r#"10\\..*"#));
    }

    #[test]
    fn next_guardrail_block_deadline_prefers_latest_reset_when_both_caps_are_exceeded() {
        let guardrail = VmGuardrail {
            pid: Some(1234),
            cgroup_path: Some("/kubepods.slice/pod123/vm.scope".to_string()),
            max_connections_per_minute: 1,
            bandwidth_cap_mb_per_hour: 1,
            blocked_until: None,
            last_conn_counter: None,
            last_byte_counter: None,
            conn_deltas: VecDeque::new(),
            byte_deltas: VecDeque::new(),
        };
        let snapshot = VmProxyLimitSnapshot {
            connection_attempts_last_minute: 5,
            allowed_bytes_last_hour: 4 * 1024 * 1024,
            connection_window_resets_at: Some(
                "2026-04-13T11:00:30Z".parse().expect("connection reset"),
            ),
            bandwidth_window_resets_at: Some(
                "2026-04-13T11:30:00Z".parse().expect("bandwidth reset"),
            ),
        };

        let blocked_until =
            next_guardrail_block_deadline(&guardrail, &snapshot).expect("blocked until");

        assert_eq!(blocked_until.to_rfc3339(), "2026-04-13T11:30:00+00:00");
    }

    #[test]
    fn apply_limit_snapshot_embeds_effective_guardrail_state() {
        let mut activity_snapshot = empty_activity_snapshot();
        let limit_snapshot = VmProxyLimitSnapshot {
            connection_attempts_last_minute: 7,
            allowed_bytes_last_hour: 9 * 1024,
            connection_window_resets_at: Some(
                "2026-04-13T11:00:30Z".parse().expect("connection reset"),
            ),
            bandwidth_window_resets_at: Some(
                "2026-04-13T11:30:00Z".parse().expect("bandwidth reset"),
            ),
        };
        let blocked_until = Some("2026-04-13T11:30:00Z".parse().expect("blocked until"));

        apply_limit_snapshot(&mut activity_snapshot, &limit_snapshot, blocked_until, true);

        assert_eq!(activity_snapshot.connection_attempts_last_minute, 7);
        assert_eq!(activity_snapshot.allowed_bytes_last_hour, 9 * 1024);
        assert_eq!(
            activity_snapshot
                .connection_window_resets_at
                .expect("connection reset")
                .to_rfc3339(),
            "2026-04-13T11:00:30+00:00"
        );
        assert_eq!(
            activity_snapshot
                .bandwidth_window_resets_at
                .expect("bandwidth reset")
                .to_rfc3339(),
            "2026-04-13T11:30:00+00:00"
        );
        assert_eq!(
            activity_snapshot
                .blocked_until
                .expect("blocked until")
                .to_rfc3339(),
            "2026-04-13T11:30:00+00:00"
        );
        assert!(activity_snapshot.includes_kernel_counters);
    }

    #[test]
    fn limit_snapshot_has_activity_rejects_idle_windows() {
        let idle_snapshot = VmProxyLimitSnapshot {
            connection_attempts_last_minute: 0,
            allowed_bytes_last_hour: 0,
            connection_window_resets_at: None,
            bandwidth_window_resets_at: None,
        };

        assert!(!limit_snapshot_has_activity(&idle_snapshot));
    }
}
