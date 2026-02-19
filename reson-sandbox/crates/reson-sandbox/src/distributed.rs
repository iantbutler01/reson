use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_nats::jetstream;
use etcd_client::{Client as EtcdClient, GetOptions, PutOptions};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::{DistributedControlConfig, Result, SandboxError};

#[derive(Clone)]
pub(crate) struct DistributedControlPlane {
    cfg: DistributedControlConfig,
    etcd: Arc<Mutex<EtcdClient>>,
    jetstream: jetstream::Context,
}

#[derive(Clone, Debug)]
pub(crate) struct SessionRoute {
    pub session_id: String,
    pub vm_id: String,
    pub endpoint: String,
    pub node_id: Option<String>,
    pub fork_id: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct NodeRoute {
    pub node_id: String,
    pub endpoint: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SessionRouteRecord {
    session_id: String,
    vm_id: String,
    endpoint: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    fork_id: Option<String>,
    #[serde(default)]
    updated_at_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NodeRecord {
    #[serde(default)]
    node_id: String,
    endpoint: String,
    #[serde(default)]
    updated_at_unix_ms: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct PortAllocation {
    pub session_id: String,
    pub vm_id: String,
    pub endpoint: String,
    pub guest_port: u16,
    pub host_port: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PortAllocationRecord {
    session_id: String,
    vm_id: String,
    endpoint: String,
    guest_port: u16,
    host_port: u16,
    #[serde(default)]
    updated_at_unix_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct OutboxRecord {
    kind: String,
    command_id: String,
    subject: String,
    payload: Value,
    #[serde(default)]
    created_at_unix_ms: u64,
    #[serde(default)]
    updated_at_unix_ms: u64,
    #[serde(default)]
    attempts: u64,
}

pub(crate) struct PortAllocationLease {
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl PortAllocationLease {
    pub(crate) async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

impl DistributedControlPlane {
    pub(crate) async fn connect(cfg: DistributedControlConfig) -> Result<Self> {
        if cfg.etcd_endpoints.is_empty() {
            return Err(SandboxError::InvalidEndpoint(
                "distributed control requires at least one etcd endpoint".to_string(),
            ));
        }
        let etcd = EtcdClient::connect(cfg.etcd_endpoints.clone(), None)
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!("failed connecting to etcd: {err}"))
            })?;
        let nats = async_nats::connect(cfg.nats_url.clone())
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!("failed connecting to nats: {err}"))
            })?;
        let normalized = normalize_cfg(cfg);
        let jetstream = jetstream::new(nats.clone());
        ensure_control_stream(&jetstream, &normalized).await?;
        let plane = Self {
            cfg: normalized,
            etcd: Arc::new(Mutex::new(etcd)),
            jetstream,
        };
        spawn_outbox_replay_worker(plane.clone());
        Ok(plane)
    }

    pub(crate) async fn get_session_route(&self, session_id: &str) -> Result<Option<SessionRoute>> {
        let key = self.session_key(session_id);
        let mut client = self.etcd.lock().await;
        let response = client.get(key, None).await.map_err(|err| {
            SandboxError::DaemonUnavailable(format!("etcd get session route failed: {err}"))
        })?;
        let Some(kv) = response.kvs().first() else {
            return Ok(None);
        };
        let route = decode_session_route(kv.value())?;
        Ok(Some(route))
    }

    pub(crate) async fn put_session_route(&self, route: SessionRoute) -> Result<()> {
        let key = self.session_key(&route.session_id);
        let payload = serde_json::to_vec(&SessionRouteRecord {
            session_id: route.session_id.clone(),
            vm_id: route.vm_id.clone(),
            endpoint: route.endpoint.clone(),
            node_id: route.node_id.clone(),
            fork_id: route.fork_id.clone(),
            updated_at_unix_ms: unix_millis(),
        })
        .map_err(|err| SandboxError::InvalidResponse(format!("serialize session route: {err}")))?;
        let mut client = self.etcd.lock().await;
        client.put(key, payload, None).await.map_err(|err| {
            SandboxError::DaemonUnavailable(format!("etcd put session route failed: {err}"))
        })?;
        Ok(())
    }

    pub(crate) async fn delete_session_route(&self, session_id: &str) -> Result<()> {
        let key = self.session_key(session_id);
        let mut client = self.etcd.lock().await;
        client.delete(key, None).await.map_err(|err| {
            SandboxError::DaemonUnavailable(format!("etcd delete session route failed: {err}"))
        })?;
        Ok(())
    }

    pub(crate) async fn list_node_routes(&self) -> Result<Vec<NodeRoute>> {
        let key_prefix = self.nodes_prefix();
        let mut client = self.etcd.lock().await;
        let response = client
            .get(key_prefix, Some(GetOptions::new().with_prefix()))
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!("etcd list nodes failed: {err}"))
            })?;

        let mut routes = Vec::new();
        for kv in response.kvs() {
            let key = String::from_utf8_lossy(kv.key()).to_string();
            if let Ok(route) = decode_node_route(&key, kv.value()) {
                routes.push(route);
            }
        }
        routes.sort_by(|a, b| a.node_id.cmp(&b.node_id).then(a.endpoint.cmp(&b.endpoint)));
        routes.dedup_by(|a, b| a.node_id == b.node_id && a.endpoint == b.endpoint);
        Ok(routes)
    }

    pub(crate) async fn select_node_for_session(&self, session_id: &str) -> Result<NodeRoute> {
        let routes = self.list_node_routes().await?;
        if routes.is_empty() {
            return Err(SandboxError::DaemonUnavailable(
                "distributed control has no registered nodes in etcd".to_string(),
            ));
        }

        let mut hasher = DefaultHasher::new();
        session_id.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % routes.len();
        Ok(routes[idx].clone())
    }

    pub(crate) async fn publish_event(&self, event_name: &str, payload: Value) -> Result<()> {
        let subject = format!("{}.{}", self.cfg.nats_subject_prefix, event_name);
        let envelope = json!({
            "event": event_name,
            "timestamp_unix_ms": unix_millis(),
            "payload": payload,
        });
        let bytes = serde_json::to_vec(&envelope).map_err(|err| {
            SandboxError::InvalidResponse(format!("serialize control event: {err}"))
        })?;
        self.jetstream
            .publish(subject, bytes.into())
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!("jetstream publish failed: {err}"))
            })?
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!("jetstream publish ack failed: {err}"))
            })?;
        Ok(())
    }

    pub(crate) async fn publish_command(
        &self,
        command_type: &str,
        ordering_key: &str,
        payload: Value,
    ) -> Result<String> {
        let command_id = Uuid::new_v4().to_string();
        let idempotency_key = payload
            .get("idempotency_key")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| derive_idempotency_key(command_type, ordering_key, &payload));
        let envelope = json!({
            "schema_version": "v1",
            "command_id": command_id,
            "command_type": command_type,
            "tenant_id": payload.get("tenant_id").and_then(Value::as_str).unwrap_or("default"),
            "workspace_id": payload.get("workspace_id").and_then(Value::as_str).unwrap_or("default"),
            "session_id": payload.get("session_id").and_then(Value::as_str),
            "vm_id": payload.get("vm_id").and_then(Value::as_str),
            "target_node_id": payload.get("target_node_id").and_then(Value::as_str),
            "ordering_key": ordering_key,
            "issued_at_unix_ms": unix_millis(),
            "timeout_ms": payload.get("timeout_ms").and_then(Value::as_u64).unwrap_or(30_000),
            "idempotency_key": idempotency_key,
            "trace_id": payload
                .get("trace_id")
                .and_then(Value::as_str)
                .unwrap_or(command_id.as_str()),
            "causation_id": payload.get("causation_id").and_then(Value::as_str),
            "expected_fence": payload.get("expected_fence"),
            "expected_versions": payload.get("expected_versions"),
            "payload": payload,
        });
        let subject = format!("{}.cmd.{command_type}", self.cfg.nats_subject_prefix);
        let outbox = OutboxRecord {
            kind: "command".to_string(),
            command_id: command_id.clone(),
            subject,
            payload: envelope,
            created_at_unix_ms: unix_millis(),
            updated_at_unix_ms: unix_millis(),
            attempts: 0,
        };
        self.put_outbox_record(&outbox).await?;
        self.publish_outbox_record(&outbox).await?;
        self.delete_outbox_record(&outbox.command_id).await?;
        Ok(command_id)
    }

    pub(crate) async fn acquire_port_lease(
        &self,
        allocation: PortAllocation,
    ) -> Result<PortAllocationLease> {
        const DEFAULT_LEASE_TTL_SECS: i64 = 30;
        let key = self.port_key(&allocation.vm_id, allocation.host_port);
        write_port_allocation(
            &self.cfg.etcd_endpoints,
            &key,
            &allocation,
            DEFAULT_LEASE_TTL_SECS,
        )
        .await?;

        let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
        let endpoints = self.cfg.etcd_endpoints.clone();
        let key_for_task = key.clone();
        let allocation_for_task = allocation.clone();
        let interval = lease_interval(DEFAULT_LEASE_TTL_SECS);
        let join = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut stop_rx => break,
                    _ = tokio::time::sleep(interval) => {
                        let _ = write_port_allocation(
                            &endpoints,
                            &key_for_task,
                            &allocation_for_task,
                            DEFAULT_LEASE_TTL_SECS,
                        ).await;
                    }
                }
            }

            let _ = delete_etcd_key(&endpoints, &key_for_task).await;
        });

        Ok(PortAllocationLease {
            stop_tx: Some(stop_tx),
            join: Some(join),
        })
    }

    fn session_key(&self, session_id: &str) -> String {
        format!("{}{session_id}", self.sessions_prefix())
    }

    fn nodes_prefix(&self) -> String {
        format!("{}/nodes/", self.cfg.etcd_prefix)
    }

    fn sessions_prefix(&self) -> String {
        format!("{}/sessions/", self.cfg.etcd_prefix)
    }

    fn ports_prefix(&self) -> String {
        format!("{}/ports/", self.cfg.etcd_prefix)
    }

    fn outbox_prefix(&self) -> String {
        format!("{}/outbox/", self.cfg.etcd_prefix)
    }

    fn outbox_pending_prefix(&self) -> String {
        format!("{}pending/", self.outbox_prefix())
    }

    fn outbox_pending_key(&self, command_id: &str) -> String {
        format!("{}{}", self.outbox_pending_prefix(), command_id)
    }

    fn port_key(&self, vm_id: &str, host_port: u16) -> String {
        format!("{}{}/{}", self.ports_prefix(), vm_id, host_port)
    }

    async fn put_outbox_record(&self, record: &OutboxRecord) -> Result<()> {
        let key = self.outbox_pending_key(&record.command_id);
        let payload = serde_json::to_vec(record).map_err(|err| {
            SandboxError::InvalidResponse(format!("serialize outbox record: {err}"))
        })?;
        let mut client = self.etcd.lock().await;
        client.put(key, payload, None).await.map_err(|err| {
            SandboxError::DaemonUnavailable(format!("etcd put outbox record failed: {err}"))
        })?;
        Ok(())
    }

    async fn delete_outbox_record(&self, command_id: &str) -> Result<()> {
        let key = self.outbox_pending_key(command_id);
        let mut client = self.etcd.lock().await;
        client.delete(key, None).await.map_err(|err| {
            SandboxError::DaemonUnavailable(format!("etcd delete outbox record failed: {err}"))
        })?;
        Ok(())
    }

    async fn publish_outbox_record(&self, record: &OutboxRecord) -> Result<()> {
        let bytes = serde_json::to_vec(&record.payload).map_err(|err| {
            SandboxError::InvalidResponse(format!("serialize outbox publish payload: {err}"))
        })?;
        self.jetstream
            .send_publish(
                record.subject.clone(),
                jetstream::context::Publish::build()
                    .message_id(record.command_id.clone())
                    .payload(bytes.into()),
            )
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!(
                    "jetstream publish outbox record failed (command_id={}): {err}",
                    record.command_id
                ))
            })?
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!(
                    "jetstream publish outbox ack failed (command_id={}): {err}",
                    record.command_id
                ))
            })?;
        Ok(())
    }

    pub(crate) async fn replay_outbox_once(&self) -> Result<()> {
        let prefix = self.outbox_pending_prefix();
        let mut client = self.etcd.lock().await;
        let response = client
            .get(prefix, Some(GetOptions::new().with_prefix()))
            .await
            .map_err(|err| {
                SandboxError::DaemonUnavailable(format!("etcd list outbox records failed: {err}"))
            })?;
        let records: Vec<OutboxRecord> = response
            .kvs()
            .iter()
            .filter_map(|kv| serde_json::from_slice::<OutboxRecord>(kv.value()).ok())
            .collect();
        drop(client);

        for mut record in records {
            if self.publish_outbox_record(&record).await.is_ok() {
                let _ = self.delete_outbox_record(&record.command_id).await;
            } else {
                record.attempts += 1;
                record.updated_at_unix_ms = unix_millis();
                let _ = self.put_outbox_record(&record).await;
            }
        }

        Ok(())
    }
}

fn spawn_outbox_replay_worker(control: DistributedControlPlane) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(5));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            interval.tick().await;
            let _ = control.replay_outbox_once().await;
        }
    });
}

fn normalize_cfg(mut cfg: DistributedControlConfig) -> DistributedControlConfig {
    cfg.etcd_prefix = normalize_prefix(&cfg.etcd_prefix);
    if cfg.nats_subject_prefix.trim().is_empty() {
        cfg.nats_subject_prefix = "reson.sandbox.control".to_string();
    }
    if cfg.nats_stream_name.trim().is_empty() {
        cfg.nats_stream_name = "RESON_SANDBOX_CONTROL".to_string();
    }
    if cfg.nats_stream_max_age_secs == 0 {
        cfg.nats_stream_max_age_secs = 60 * 60 * 24 * 7;
    }
    if cfg.nats_stream_replicas == 0 {
        cfg.nats_stream_replicas = 1;
    }
    if cfg.nats_dead_letter_subject.trim().is_empty() {
        cfg.nats_dead_letter_subject = format!("{}.dlq.commands", cfg.nats_subject_prefix);
    }
    cfg
}

async fn ensure_control_stream(
    jetstream: &jetstream::Context,
    cfg: &DistributedControlConfig,
) -> Result<()> {
    let mut subjects = vec![
        format!("{}.cmd.>", cfg.nats_subject_prefix),
        format!("{}.evt.>", cfg.nats_subject_prefix),
        cfg.nats_dead_letter_subject.clone(),
        format!("{}.replay.>", cfg.nats_subject_prefix),
    ];
    subjects.sort();
    subjects.dedup();

    jetstream
        .get_or_create_stream(jetstream::stream::Config {
            name: cfg.nats_stream_name.clone(),
            subjects,
            max_age: Duration::from_secs(cfg.nats_stream_max_age_secs),
            storage: jetstream::stream::StorageType::File,
            num_replicas: cfg.nats_stream_replicas,
            ..Default::default()
        })
        .await
        .map_err(|err| {
            SandboxError::DaemonUnavailable(format!("ensure control jetstream stream failed: {err}"))
        })?;
    Ok(())
}

fn normalize_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim().trim_matches('/');
    if trimmed.is_empty() {
        return "/reson-sandbox".to_string();
    }
    format!("/{trimmed}")
}

fn decode_session_route(raw: &[u8]) -> Result<SessionRoute> {
    if let Ok(record) = serde_json::from_slice::<SessionRouteRecord>(raw) {
        return Ok(SessionRoute {
            session_id: record.session_id,
            vm_id: record.vm_id,
            endpoint: record.endpoint,
            node_id: record.node_id,
            fork_id: record.fork_id,
        });
    }

    let endpoint = String::from_utf8(raw.to_vec()).map_err(|err| {
        SandboxError::InvalidResponse(format!("invalid session route value: {err}"))
    })?;
    Ok(SessionRoute {
        session_id: String::new(),
        vm_id: String::new(),
        endpoint,
        node_id: None,
        fork_id: None,
    })
}

fn decode_node_route(key: &str, raw: &[u8]) -> Result<NodeRoute> {
    if let Ok(record) = serde_json::from_slice::<NodeRecord>(raw) {
        let node_id = if record.node_id.trim().is_empty() {
            key.rsplit('/').next().unwrap_or_default().to_string()
        } else {
            record.node_id
        };
        if record.endpoint.trim().is_empty() {
            return Err(SandboxError::InvalidResponse(
                "node record missing endpoint".to_string(),
            ));
        }
        return Ok(NodeRoute {
            node_id,
            endpoint: record.endpoint,
        });
    }

    let endpoint = String::from_utf8(raw.to_vec())
        .map_err(|err| SandboxError::InvalidResponse(format!("invalid node route value: {err}")))?;
    let node_id = key.rsplit('/').next().unwrap_or_default().to_string();
    Ok(NodeRoute { node_id, endpoint })
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn derive_idempotency_key(command_type: &str, ordering_key: &str, payload: &Value) -> String {
    let mut hasher = DefaultHasher::new();
    command_type.hash(&mut hasher);
    ordering_key.hash(&mut hasher);
    serde_json::to_string(payload)
        .unwrap_or_default()
        .hash(&mut hasher);
    format!("idem-{:016x}", hasher.finish())
}

fn lease_interval(ttl_secs: i64) -> std::time::Duration {
    let ttl = ttl_secs.max(2) as u64;
    let interval = (ttl / 2).max(1);
    std::time::Duration::from_secs(interval)
}

async fn write_port_allocation(
    endpoints: &[String],
    key: &str,
    allocation: &PortAllocation,
    ttl_secs: i64,
) -> Result<()> {
    let mut client = EtcdClient::connect(endpoints.to_vec(), None)
        .await
        .map_err(|err| SandboxError::DaemonUnavailable(format!("etcd connect failed: {err}")))?;
    let lease = client.lease_grant(ttl_secs, None).await.map_err(|err| {
        SandboxError::DaemonUnavailable(format!("etcd lease grant failed: {err}"))
    })?;
    let payload = serde_json::to_vec(&PortAllocationRecord {
        session_id: allocation.session_id.clone(),
        vm_id: allocation.vm_id.clone(),
        endpoint: allocation.endpoint.clone(),
        guest_port: allocation.guest_port,
        host_port: allocation.host_port,
        updated_at_unix_ms: unix_millis(),
    })
    .map_err(|err| SandboxError::InvalidResponse(format!("serialize port allocation: {err}")))?;
    client
        .put(key, payload, Some(PutOptions::new().with_lease(lease.id())))
        .await
        .map_err(|err| SandboxError::DaemonUnavailable(format!("etcd put failed: {err}")))?;
    Ok(())
}

async fn delete_etcd_key(endpoints: &[String], key: &str) -> Result<()> {
    let mut client = EtcdClient::connect(endpoints.to_vec(), None)
        .await
        .map_err(|err| SandboxError::DaemonUnavailable(format!("etcd connect failed: {err}")))?;
    client
        .delete(key, None)
        .await
        .map_err(|err| SandboxError::DaemonUnavailable(format!("etcd delete failed: {err}")))?;
    Ok(())
}
