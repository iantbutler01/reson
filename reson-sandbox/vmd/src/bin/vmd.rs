use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

use vmd_rs::app;
use vmd_rs::config::{
    AuthConfig, Config, ControlBusConfig, NodeRegistryConfig, SecurityConfig, TlsServerConfig,
};

#[derive(Parser, Debug)]
#[command(name = "vmd", about = "Bracket VM daemon (Rust)")]
struct Args {
    #[arg(long)]
    listen: Option<String>,
    #[arg(long)]
    data_dir: Option<String>,
    #[arg(long)]
    qemu_bin: Option<String>,
    #[arg(long)]
    qemu_arm64_bin: Option<String>,
    #[arg(long)]
    qemu_img: Option<String>,
    #[arg(long)]
    docker_bin: Option<String>,
    #[arg(long)]
    log_level: Option<String>,
    /// Skip checking S3 for prebuilt VM images and always build locally
    #[arg(long)]
    force_local_build: bool,
    /// Admission limit for concurrently active VMs on this node.
    #[arg(long)]
    max_active_vms: Option<usize>,
    /// Enable node registry heartbeats by providing etcd endpoints (comma-separated).
    #[arg(long)]
    registry_etcd_endpoints: Option<String>,
    /// Override etcd key prefix used for node registry keys.
    #[arg(long)]
    registry_prefix: Option<String>,
    /// Override registry node identifier.
    #[arg(long)]
    node_id: Option<String>,
    /// Override advertised daemon endpoint for node routing.
    #[arg(long)]
    advertise_endpoint: Option<String>,
    /// Registry key lease TTL in seconds.
    #[arg(long)]
    registry_ttl_secs: Option<i64>,
    /// Disable node registry heartbeats even when env vars are present.
    #[arg(long)]
    disable_node_registry: bool,
    /// Enable control command consumer and override NATS URL.
    #[arg(long)]
    control_nats_url: Option<String>,
    /// Override control bus subject prefix.
    #[arg(long)]
    control_subject_prefix: Option<String>,
    /// Override control bus node identifier.
    #[arg(long)]
    control_node_id: Option<String>,
    /// Disable control command consumer even when env vars are present.
    #[arg(long)]
    disable_control_bus: bool,
    /// Bearer token required for control-plane RPCs.
    #[arg(long)]
    auth_token: Option<String>,
    /// Path to file containing bearer token required for control-plane RPCs.
    #[arg(long)]
    auth_token_file: Option<String>,
    /// Optional read-only bearer token that may call read RPCs only.
    #[arg(long)]
    readonly_auth_token: Option<String>,
    /// Path to file containing optional read-only bearer token.
    #[arg(long)]
    readonly_auth_token_file: Option<String>,
    /// PEM server certificate for TLS listener.
    #[arg(long)]
    tls_cert: Option<String>,
    /// PEM private key for TLS listener.
    #[arg(long)]
    tls_key: Option<String>,
    /// PEM CA bundle used to validate mTLS client certificates.
    #[arg(long)]
    tls_client_ca: Option<String>,
    /// Disable strict client-certificate requirement when TLS is enabled.
    #[arg(long)]
    tls_allow_optional_client_cert: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::default();

    if let Some(listen) = args.listen {
        cfg.listen_address = listen;
    }
    if let Some(dir) = args.data_dir {
        cfg.data_dir = dir;
    }
    if let Some(qemu) = args.qemu_bin {
        cfg.qemu_bin = qemu;
    }
    if let Some(qemu) = args.qemu_arm64_bin {
        cfg.qemu_arm64_bin = qemu;
    }
    if let Some(qemu_img) = args.qemu_img {
        cfg.qemu_img_bin = qemu_img;
    }
    if let Some(docker) = args.docker_bin {
        cfg.docker_bin = docker;
    }
    if let Some(level) = args.log_level {
        cfg.log_level = level;
    }
    if args.force_local_build {
        cfg.force_local_build = true;
    }
    if let Some(limit) = args.max_active_vms {
        cfg.max_active_vms = if limit == 0 { None } else { Some(limit) };
    }
    if args.disable_node_registry {
        cfg.node_registry = None;
    }
    if args.disable_control_bus {
        cfg.control_bus = None;
    }

    let has_registry_overrides = args.registry_etcd_endpoints.is_some()
        || args.registry_prefix.is_some()
        || args.node_id.is_some()
        || args.advertise_endpoint.is_some()
        || args.registry_ttl_secs.is_some();
    if has_registry_overrides {
        let mut registry = cfg
            .node_registry
            .clone()
            .unwrap_or_else(|| NodeRegistryConfig::defaults_for_listen(&cfg.listen_address));
        if let Some(endpoints) = args.registry_etcd_endpoints {
            registry.etcd_endpoints = endpoints
                .split(',')
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect();
        }
        if let Some(prefix) = args.registry_prefix {
            registry.key_prefix = prefix;
        }
        if let Some(node_id) = args.node_id {
            registry.node_id = node_id;
        }
        if let Some(endpoint) = args.advertise_endpoint {
            registry.advertise_endpoint = endpoint;
        }
        if let Some(ttl_secs) = args.registry_ttl_secs {
            registry.ttl_secs = ttl_secs;
        }
        cfg.node_registry = Some(registry);
    }

    let has_control_overrides = args.control_nats_url.is_some()
        || args.control_subject_prefix.is_some()
        || args.control_node_id.is_some();
    if has_control_overrides {
        let mut control = cfg.control_bus.clone().unwrap_or(ControlBusConfig {
            nats_url: "nats://127.0.0.1:4222".to_string(),
            subject_prefix: "reson.sandbox.control".to_string(),
            node_id: "vmd-node".to_string(),
            dedupe_etcd_endpoints: Vec::new(),
            dedupe_prefix: "/reson-sandbox/command-dedupe".to_string(),
            stream_name: "RESON_SANDBOX_CONTROL".to_string(),
            stream_max_age_secs: 60 * 60 * 24 * 7,
            stream_replicas: 1,
            command_consumer_durable: "vmd-node-cmd".to_string(),
            command_max_deliver: 5,
            command_ack_wait_ms: 30_000,
            dead_letter_subject: "reson.sandbox.control.dlq.commands".to_string(),
            replay_subject: "reson.sandbox.control.replay.commands".to_string(),
        });
        if let Some(url) = args.control_nats_url {
            control.nats_url = url;
        }
        if let Some(prefix) = args.control_subject_prefix {
            control.subject_prefix = prefix;
        }
        if let Some(node_id) = args.control_node_id {
            control.node_id = node_id;
        }
        cfg.control_bus = Some(control);
    }

    let mut security = cfg.security.clone();
    let auth_token = resolve_secret(args.auth_token, args.auth_token_file.as_deref())?;
    let readonly_auth_token = resolve_secret(
        args.readonly_auth_token,
        args.readonly_auth_token_file.as_deref(),
    )?;
    if auth_token.is_some() || readonly_auth_token.is_some() {
        let mut auth = security.auth.take().unwrap_or(AuthConfig {
            admin_token: String::new(),
            readonly_token: None,
        });
        if let Some(token) = auth_token {
            auth.admin_token = token;
        }
        if readonly_auth_token.is_some() {
            auth.readonly_token = readonly_auth_token;
        }
        security.auth = Some(auth);
    }

    if args.tls_cert.is_some() || args.tls_key.is_some() || args.tls_client_ca.is_some() {
        let mut tls = security.tls.take().unwrap_or(TlsServerConfig {
            cert_path: String::new(),
            key_path: String::new(),
            client_ca_path: None,
            require_client_cert: true,
        });
        if let Some(cert) = args.tls_cert {
            tls.cert_path = cert;
        }
        if let Some(key) = args.tls_key {
            tls.key_path = key;
        }
        if args.tls_client_ca.is_some() {
            tls.client_ca_path = args.tls_client_ca;
        }
        if args.tls_allow_optional_client_cert {
            tls.require_client_cert = false;
        }
        security.tls = Some(tls);
    }
    cfg.security = SecurityConfig {
        auth: security.auth,
        tls: security.tls,
    };

    init_tracing(&cfg.log_level)?;
    app::run_server(cfg).await
}

fn init_tracing(level: &str) -> Result<()> {
    let filter = EnvFilter::try_new(level)
        .or_else(|_| EnvFilter::try_new(format!("vmd_rs={level}")))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();

    Ok(())
}

fn resolve_secret(inline: Option<String>, file_path: Option<&str>) -> Result<Option<String>> {
    if let Some(value) = inline {
        let trimmed = value.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(Some(trimmed));
        }
    }
    let Some(file_path) = file_path else {
        return Ok(None);
    };
    let value = std::fs::read_to_string(file_path)?;
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() {
        return Ok(None);
    }
    Ok(Some(trimmed))
}
