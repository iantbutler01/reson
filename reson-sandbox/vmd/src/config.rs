use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use uuid::Uuid;

pub const BASE_IMAGES_DIR_NAME: &str = "base_images";
const DEFAULT_NODE_REGISTRY_PREFIX: &str = "/reson-sandbox";
const DEFAULT_NODE_REGISTRY_TTL_SECS: i64 = 15;
const DEFAULT_CONTROL_SUBJECT_PREFIX: &str = "reson.sandbox.control";
const DEFAULT_CONTROL_DEDUPE_PREFIX: &str = "/reson-sandbox/command-dedupe";
const DEFAULT_CONTROL_STREAM_NAME: &str = "RESON_SANDBOX_CONTROL";
const DEFAULT_CONTROL_STREAM_MAX_AGE_SECS: u64 = 60 * 60 * 24 * 7;
const DEFAULT_CONTROL_STREAM_REPLICAS: usize = 1;
const DEFAULT_CONTROL_COMMAND_MAX_DELIVER: i64 = 5;
const DEFAULT_CONTROL_COMMAND_ACK_WAIT_MS: u64 = 30_000;
const DEFAULT_REQUIRE_CLIENT_CERT: bool = true;

#[derive(Clone, Debug)]
pub struct NodeRegistryConfig {
    pub etcd_endpoints: Vec<String>,
    pub key_prefix: String,
    pub node_id: String,
    pub advertise_endpoint: String,
    pub ttl_secs: i64,
}

impl NodeRegistryConfig {
    pub fn defaults_for_listen(listen_address: &str) -> Self {
        Self {
            etcd_endpoints: vec!["http://127.0.0.1:2379".to_string()],
            key_prefix: DEFAULT_NODE_REGISTRY_PREFIX.to_string(),
            node_id: default_node_id(),
            advertise_endpoint: format!("http://{}", listen_address.trim()),
            ttl_secs: DEFAULT_NODE_REGISTRY_TTL_SECS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControlBusConfig {
    pub nats_url: String,
    pub subject_prefix: String,
    pub node_id: String,
    pub dedupe_etcd_endpoints: Vec<String>,
    pub dedupe_prefix: String,
    pub stream_name: String,
    pub stream_max_age_secs: u64,
    pub stream_replicas: usize,
    pub command_consumer_durable: String,
    pub command_max_deliver: i64,
    pub command_ack_wait_ms: u64,
    pub dead_letter_subject: String,
    pub replay_subject: String,
}

#[derive(Clone, Debug)]
pub struct AuthConfig {
    pub admin_token: String,
    pub readonly_token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TlsServerConfig {
    pub cert_path: String,
    pub key_path: String,
    pub client_ca_path: Option<String>,
    pub require_client_cert: bool,
}

#[derive(Clone, Debug, Default)]
pub struct SecurityConfig {
    pub auth: Option<AuthConfig>,
    pub tls: Option<TlsServerConfig>,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub listen_address: String,
    pub data_dir: String,
    pub qemu_bin: String,
    pub qemu_arm64_bin: String,
    pub qemu_img_bin: String,
    pub docker_bin: String,
    pub log_level: String,
    pub force_local_build: bool,
    pub max_active_vms: Option<usize>,
    pub node_registry: Option<NodeRegistryConfig>,
    pub control_bus: Option<ControlBusConfig>,
    pub security: SecurityConfig,
}

impl Default for Config {
    fn default() -> Self {
        let data_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".bracket")
            .join("vms");

        Self {
            listen_address: "127.0.0.1:8052".to_string(),
            data_dir: data_dir.to_string_lossy().to_string(),
            qemu_bin: "qemu-system-x86_64".to_string(),
            qemu_arm64_bin: "qemu-system-aarch64".to_string(),
            qemu_img_bin: "qemu-img".to_string(),
            docker_bin: "docker".to_string(),
            log_level: "info".to_string(),
            force_local_build: false,
            max_active_vms: default_max_active_vms_from_env(),
            node_registry: default_node_registry_from_env("127.0.0.1:8052"),
            control_bus: default_control_bus_from_env(),
            security: default_security_from_env(),
        }
    }
}

impl Config {
    pub fn base_images_dir(&self) -> PathBuf {
        Path::new(&self.data_dir).join(BASE_IMAGES_DIR_NAME)
    }

    pub fn normalize(&mut self) -> Result<()> {
        if self.listen_address.trim().is_empty() {
            bail!("listen address must be provided");
        }
        if self.data_dir.trim().is_empty() {
            bail!("data dir must be provided");
        }

        let data_dir = expand_home(&self.data_dir)?;
        fs::create_dir_all(&data_dir)
            .with_context(|| format!("create data dir {}", data_dir.to_string_lossy()))?;
        let base_dir = data_dir.join(BASE_IMAGES_DIR_NAME);
        fs::create_dir_all(&base_dir)
            .with_context(|| format!("create base image dir {}", base_dir.to_string_lossy()))?;
        self.data_dir = canonical_string(&data_dir)?;

        self.qemu_bin = resolve_binary(&self.qemu_bin, "qemu-system-x86_64")?;
        self.qemu_arm64_bin = resolve_binary(&self.qemu_arm64_bin, "qemu-system-aarch64")?;
        self.qemu_img_bin = resolve_binary(&self.qemu_img_bin, "qemu-img")?;
        self.docker_bin = resolve_binary(&self.docker_bin, "docker")?;
        if self.max_active_vms == Some(0) {
            self.max_active_vms = None;
        }
        self.normalize_node_registry();
        self.normalize_control_bus();
        self.normalize_security()?;

        Ok(())
    }

    fn normalize_node_registry(&mut self) {
        let Some(registry) = self.node_registry.as_mut() else {
            return;
        };

        registry.etcd_endpoints = registry
            .etcd_endpoints
            .iter()
            .map(String::as_str)
            .map(normalize_endpoint_like)
            .collect();
        registry.etcd_endpoints.retain(|value| !value.is_empty());

        if registry.etcd_endpoints.is_empty() {
            self.node_registry = None;
            return;
        }

        let prefix = registry.key_prefix.trim().trim_matches('/');
        registry.key_prefix = if prefix.is_empty() {
            DEFAULT_NODE_REGISTRY_PREFIX.to_string()
        } else {
            format!("/{prefix}")
        };

        if registry.node_id.trim().is_empty() {
            registry.node_id = default_node_id();
        }

        if registry.advertise_endpoint.trim().is_empty() {
            registry.advertise_endpoint = format!("http://{}", self.listen_address.trim());
        } else {
            registry.advertise_endpoint = normalize_endpoint_like(&registry.advertise_endpoint);
        }

        if registry.ttl_secs <= 0 {
            registry.ttl_secs = DEFAULT_NODE_REGISTRY_TTL_SECS;
        }
    }

    fn normalize_control_bus(&mut self) {
        let Some(control) = self.control_bus.as_mut() else {
            return;
        };

        control.nats_url = normalize_nats_url(&control.nats_url);
        if control.nats_url.is_empty() {
            self.control_bus = None;
            return;
        }

        if control.subject_prefix.trim().is_empty() {
            control.subject_prefix = DEFAULT_CONTROL_SUBJECT_PREFIX.to_string();
        }

        if control.node_id.trim().is_empty() {
            control.node_id = default_node_id();
        }

        control.dedupe_etcd_endpoints = control
            .dedupe_etcd_endpoints
            .iter()
            .map(String::as_str)
            .map(normalize_endpoint_like)
            .collect();
        control
            .dedupe_etcd_endpoints
            .retain(|value| !value.trim().is_empty());

        if control.dedupe_etcd_endpoints.is_empty() {
            if let Some(node_registry) = &self.node_registry {
                control.dedupe_etcd_endpoints = node_registry.etcd_endpoints.clone();
            }
        }

        let dedupe_prefix = control.dedupe_prefix.trim().trim_matches('/');
        control.dedupe_prefix = if dedupe_prefix.is_empty() {
            DEFAULT_CONTROL_DEDUPE_PREFIX.to_string()
        } else {
            format!("/{dedupe_prefix}")
        };

        if control.stream_name.trim().is_empty() {
            control.stream_name = DEFAULT_CONTROL_STREAM_NAME.to_string();
        }
        if control.stream_max_age_secs == 0 {
            control.stream_max_age_secs = DEFAULT_CONTROL_STREAM_MAX_AGE_SECS;
        }
        if control.stream_replicas == 0 {
            control.stream_replicas = DEFAULT_CONTROL_STREAM_REPLICAS;
        }
        if control.command_max_deliver <= 0 {
            control.command_max_deliver = DEFAULT_CONTROL_COMMAND_MAX_DELIVER;
        }
        if control.command_ack_wait_ms == 0 {
            control.command_ack_wait_ms = DEFAULT_CONTROL_COMMAND_ACK_WAIT_MS;
        }
        if control.dead_letter_subject.trim().is_empty() {
            control.dead_letter_subject = format!("{}.dlq.commands", control.subject_prefix);
        }
        if control.replay_subject.trim().is_empty() {
            control.replay_subject = format!("{}.replay.commands", control.subject_prefix);
        }
        if control.command_consumer_durable.trim().is_empty() {
            control.command_consumer_durable =
                format!("{}-cmd", sanitize_jetstream_name(&control.node_id));
        } else {
            control.command_consumer_durable =
                sanitize_jetstream_name(&control.command_consumer_durable);
        }
    }

    fn normalize_security(&mut self) -> Result<()> {
        if let Some(auth) = self.security.auth.as_mut() {
            auth.admin_token = auth.admin_token.trim().to_string();
            auth.readonly_token = auth
                .readonly_token
                .as_ref()
                .map(|value| value.trim().to_string());
            if auth.admin_token.is_empty() {
                self.security.auth = None;
            } else if auth
                .readonly_token
                .as_ref()
                .is_some_and(|value| value.is_empty())
            {
                auth.readonly_token = None;
            }
        }

        let Some(tls) = self.security.tls.as_mut() else {
            return Ok(());
        };

        if tls.cert_path.trim().is_empty() && tls.key_path.trim().is_empty() {
            self.security.tls = None;
            return Ok(());
        }
        if tls.cert_path.trim().is_empty() || tls.key_path.trim().is_empty() {
            bail!("both TLS cert path and key path must be configured");
        }

        let cert_path = expand_home(&tls.cert_path)?;
        fs::metadata(&cert_path)
            .with_context(|| format!("stat tls cert path {}", cert_path.to_string_lossy()))?;
        tls.cert_path = canonical_string(&cert_path)?;

        let key_path = expand_home(&tls.key_path)?;
        fs::metadata(&key_path)
            .with_context(|| format!("stat tls key path {}", key_path.to_string_lossy()))?;
        tls.key_path = canonical_string(&key_path)?;

        tls.client_ca_path = tls
            .client_ca_path
            .as_ref()
            .map(|value| value.trim().to_string());
        if let Some(client_ca_path) = tls.client_ca_path.as_ref() {
            if client_ca_path.is_empty() {
                tls.client_ca_path = None;
            } else {
                let expanded = expand_home(client_ca_path)?;
                fs::metadata(&expanded).with_context(|| {
                    format!("stat tls client ca path {}", expanded.to_string_lossy())
                })?;
                tls.client_ca_path = Some(canonical_string(&expanded)?);
            }
        }

        if tls.client_ca_path.is_none() {
            tls.require_client_cert = false;
        }

        Ok(())
    }
}

fn expand_home(path: &str) -> Result<PathBuf> {
    if path.is_empty() {
        bail!("empty path");
    }
    if let Some(stripped) = path.strip_prefix("~/") {
        let home = dirs::home_dir().context("resolve home directory")?;
        return Ok(home.join(stripped));
    }
    if path == "~" {
        let home = dirs::home_dir().context("resolve home directory")?;
        return Ok(home);
    }
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        Ok(candidate)
    } else {
        Ok(env::current_dir()?.join(candidate))
    }
}

fn resolve_binary(value: &str, fallback: &str) -> Result<String> {
    let candidate = {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            fallback
        } else {
            trimmed
        }
    };

    let path =
        lookup_executable(candidate).with_context(|| format!("locate executable {candidate}"))?;
    canonical_string(&path)
}

fn lookup_executable(candidate: &str) -> Result<PathBuf> {
    let path = PathBuf::from(candidate);
    if path.is_absolute() {
        fs::metadata(&path).with_context(|| format!("stat {}", path.to_string_lossy()))?;
        return Ok(path);
    }

    let paths = env::var_os("PATH").context("PATH environment variable unset")?;
    for dir in env::split_paths(&paths) {
        let full = dir.join(candidate);
        if full.is_file() {
            return Ok(full);
        }
    }
    bail!("{} not found in PATH", candidate);
}

fn canonical_string(path: &Path) -> Result<String> {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Ok(canonical.to_string_lossy().to_string())
}

fn default_node_registry_from_env(listen_address: &str) -> Option<NodeRegistryConfig> {
    let endpoints = parse_csv_env("RESON_SANDBOX_ETCD_ENDPOINTS")
        .or_else(|| parse_csv_env("BRACKET_SANDBOX_ETCD_ENDPOINTS"))?;

    let mut config = NodeRegistryConfig::defaults_for_listen(listen_address);
    config.etcd_endpoints = endpoints
        .into_iter()
        .map(|value| normalize_endpoint_like(&value))
        .collect();
    config.key_prefix = env::var("RESON_SANDBOX_ETCD_PREFIX")
        .or_else(|_| env::var("BRACKET_SANDBOX_ETCD_PREFIX"))
        .unwrap_or_else(|_| DEFAULT_NODE_REGISTRY_PREFIX.to_string());
    config.node_id = env::var("RESON_SANDBOX_NODE_ID")
        .or_else(|_| env::var("BRACKET_SANDBOX_NODE_ID"))
        .unwrap_or_else(|_| default_node_id());
    config.advertise_endpoint = env::var("RESON_SANDBOX_NODE_ENDPOINT")
        .or_else(|_| env::var("BRACKET_SANDBOX_NODE_ENDPOINT"))
        .map(|value| normalize_endpoint_like(&value))
        .unwrap_or_default();
    config.ttl_secs = env::var("RESON_SANDBOX_NODE_TTL_SECS")
        .or_else(|_| env::var("BRACKET_SANDBOX_NODE_TTL_SECS"))
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_NODE_REGISTRY_TTL_SECS);

    Some(config)
}

fn default_control_bus_from_env() -> Option<ControlBusConfig> {
    let nats_url = env::var("RESON_SANDBOX_NATS_URL")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_URL"))
        .ok()
        .map(|value| normalize_nats_url(&value))?;
    if nats_url.is_empty() {
        return None;
    }

    let subject_prefix = env::var("RESON_SANDBOX_NATS_SUBJECT_PREFIX")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_SUBJECT_PREFIX"))
        .unwrap_or_else(|_| DEFAULT_CONTROL_SUBJECT_PREFIX.to_string());
    let node_id = env::var("RESON_SANDBOX_NODE_ID")
        .or_else(|_| env::var("BRACKET_SANDBOX_NODE_ID"))
        .unwrap_or_else(|_| default_node_id());
    let dedupe_etcd_endpoints = parse_csv_env("RESON_SANDBOX_ETCD_ENDPOINTS")
        .or_else(|| parse_csv_env("BRACKET_SANDBOX_ETCD_ENDPOINTS"))
        .unwrap_or_default()
        .into_iter()
        .map(|value| normalize_endpoint_like(&value))
        .collect();
    let dedupe_prefix = env::var("RESON_SANDBOX_CONTROL_DEDUPE_PREFIX")
        .or_else(|_| env::var("BRACKET_SANDBOX_CONTROL_DEDUPE_PREFIX"))
        .unwrap_or_else(|_| DEFAULT_CONTROL_DEDUPE_PREFIX.to_string());
    let stream_name = env::var("RESON_SANDBOX_NATS_STREAM")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_STREAM"))
        .unwrap_or_else(|_| DEFAULT_CONTROL_STREAM_NAME.to_string());
    let stream_max_age_secs = env::var("RESON_SANDBOX_NATS_STREAM_MAX_AGE_SECS")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_STREAM_MAX_AGE_SECS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CONTROL_STREAM_MAX_AGE_SECS);
    let stream_replicas = env::var("RESON_SANDBOX_NATS_STREAM_REPLICAS")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_STREAM_REPLICAS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CONTROL_STREAM_REPLICAS);
    let command_consumer_durable = env::var("RESON_SANDBOX_CONTROL_CONSUMER_DURABLE")
        .or_else(|_| env::var("BRACKET_SANDBOX_CONTROL_CONSUMER_DURABLE"))
        .unwrap_or_else(|_| format!("{}-cmd", sanitize_jetstream_name(&node_id)));
    let command_max_deliver = env::var("RESON_SANDBOX_CONTROL_MAX_DELIVER")
        .or_else(|_| env::var("BRACKET_SANDBOX_CONTROL_MAX_DELIVER"))
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CONTROL_COMMAND_MAX_DELIVER);
    let command_ack_wait_ms = env::var("RESON_SANDBOX_CONTROL_ACK_WAIT_MS")
        .or_else(|_| env::var("BRACKET_SANDBOX_CONTROL_ACK_WAIT_MS"))
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_CONTROL_COMMAND_ACK_WAIT_MS);
    let dead_letter_subject = env::var("RESON_SANDBOX_NATS_DEAD_LETTER_SUBJECT")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_DEAD_LETTER_SUBJECT"))
        .unwrap_or_else(|_| format!("{}.dlq.commands", subject_prefix));
    let replay_subject = env::var("RESON_SANDBOX_NATS_REPLAY_SUBJECT")
        .or_else(|_| env::var("BRACKET_SANDBOX_NATS_REPLAY_SUBJECT"))
        .unwrap_or_else(|_| format!("{}.replay.commands", subject_prefix));

    Some(ControlBusConfig {
        nats_url,
        subject_prefix: subject_prefix.clone(),
        node_id,
        dedupe_etcd_endpoints,
        dedupe_prefix,
        stream_name,
        stream_max_age_secs,
        stream_replicas,
        command_consumer_durable,
        command_max_deliver,
        command_ack_wait_ms,
        dead_letter_subject,
        replay_subject,
    })
}

fn default_max_active_vms_from_env() -> Option<usize> {
    env::var("RESON_SANDBOX_MAX_ACTIVE_VMS")
        .or_else(|_| env::var("BRACKET_SANDBOX_MAX_ACTIVE_VMS"))
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn default_security_from_env() -> SecurityConfig {
    let admin_token = env::var("RESON_SANDBOX_AUTH_TOKEN")
        .or_else(|_| env::var("BRACKET_SANDBOX_AUTH_TOKEN"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let readonly_token = env::var("RESON_SANDBOX_READONLY_AUTH_TOKEN")
        .or_else(|_| env::var("BRACKET_SANDBOX_READONLY_AUTH_TOKEN"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let auth = admin_token.map(|admin_token| AuthConfig {
        admin_token,
        readonly_token,
    });

    let cert_path = env::var("RESON_SANDBOX_TLS_CERT_PATH")
        .or_else(|_| env::var("BRACKET_SANDBOX_TLS_CERT_PATH"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let key_path = env::var("RESON_SANDBOX_TLS_KEY_PATH")
        .or_else(|_| env::var("BRACKET_SANDBOX_TLS_KEY_PATH"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let client_ca_path = env::var("RESON_SANDBOX_TLS_CLIENT_CA_PATH")
        .or_else(|_| env::var("BRACKET_SANDBOX_TLS_CLIENT_CA_PATH"))
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let require_client_cert = env::var("RESON_SANDBOX_TLS_REQUIRE_CLIENT_CERT")
        .or_else(|_| env::var("BRACKET_SANDBOX_TLS_REQUIRE_CLIENT_CERT"))
        .ok()
        .and_then(|value| parse_bool_flag(&value))
        .unwrap_or(DEFAULT_REQUIRE_CLIENT_CERT);

    let tls = match (cert_path, key_path) {
        (Some(cert_path), Some(key_path)) => Some(TlsServerConfig {
            cert_path,
            key_path,
            client_ca_path,
            require_client_cert,
        }),
        _ => None,
    };

    SecurityConfig { auth, tls }
}

fn parse_csv_env(name: &str) -> Option<Vec<String>> {
    let value = env::var(name).ok()?;
    let endpoints: Vec<String> = value
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToOwned::to_owned)
        .collect();
    if endpoints.is_empty() {
        return None;
    }
    Some(endpoints)
}

fn normalize_endpoint_like(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains("://") {
        return trimmed.to_string();
    }
    format!("http://{trimmed}")
}

fn normalize_nats_url(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.contains("://") {
        return trimmed.to_string();
    }
    format!("nats://{trimmed}")
}

fn sanitize_jetstream_name(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "reson_sandbox".to_string()
    } else {
        trimmed.to_string()
    }
}

fn default_node_id() -> String {
    if let Ok(value) = env::var("HOSTNAME") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    format!("node-{}", Uuid::new_v4())
}

fn parse_bool_flag(raw: &str) -> Option<bool> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    }
}
