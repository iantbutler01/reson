use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use etcd_client::{Client as EtcdClient, GetOptions};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::{debug, info, warn};

use crate::config::{ControlBusConfig, NodeRegistryConfig};
use crate::state::Manager;

const META_SESSION_ID: &str = "reson.session_id";

#[derive(Clone)]
pub struct ReconcileConfig {
    pub etcd_endpoints: Vec<String>,
    pub etcd_prefix: String,
    pub node_id: String,
    pub node_endpoint: String,
    pub interval: Duration,
    pub nats_url: Option<String>,
    pub nats_subject_prefix: String,
}

pub struct ReconcileHandle {
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl ReconcileHandle {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SessionRoute {
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    vm_id: String,
    endpoint: String,
    #[serde(default)]
    node_id: Option<String>,
    #[serde(default)]
    fork_id: Option<String>,
}

#[derive(Clone, Debug)]
struct SessionRouteEntry {
    key: String,
    route: SessionRoute,
}

#[derive(Default)]
struct ReconcilePlan {
    delete_keys: Vec<String>,
    upserts: Vec<SessionRoute>,
}

pub async fn start(
    manager: Arc<Manager>,
    node_registry: Option<NodeRegistryConfig>,
    control_bus: Option<ControlBusConfig>,
    mut trigger_rx: mpsc::UnboundedReceiver<()>,
) -> Result<Option<ReconcileHandle>> {
    let Some(config) = derive_config(node_registry, control_bus.clone()) else {
        return Ok(None);
    };

    let mut nats = None;
    if let Some(url) = config.nats_url.as_ref() {
        match async_nats::connect(url.clone()).await {
            Ok(client) => nats = Some(client),
            Err(err) => {
                warn!(url = %url, err = %err, "reconcile nats connect failed; continuing without event publish")
            }
        }
    }

    let interval_secs = config.interval.as_secs();
    let log_node_id = config.node_id.clone();
    let log_endpoint = config.node_endpoint.clone();
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(config.interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);

        if let Err(err) = reconcile_once(&manager, &config, nats.as_ref()).await {
            warn!(node_id = %config.node_id, err = %err, "initial reconciliation run failed");
        }

        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                maybe_trigger = trigger_rx.recv() => {
                    if maybe_trigger.is_none() {
                        break;
                    }
                    if let Err(err) = reconcile_once(&manager, &config, nats.as_ref()).await {
                        warn!(node_id = %config.node_id, err = %err, "triggered reconciliation run failed");
                    }
                }
                _ = ticker.tick() => {
                    if let Err(err) = reconcile_once(&manager, &config, nats.as_ref()).await {
                        warn!(node_id = %config.node_id, err = %err, "periodic reconciliation run failed");
                    }
                }
            }
        }
    });

    info!(
        node_id = %log_node_id,
        endpoint = %log_endpoint,
        interval_secs = interval_secs,
        "reconciliation worker enabled"
    );

    Ok(Some(ReconcileHandle {
        stop_tx: Some(stop_tx),
        join: Some(join),
    }))
}

fn derive_config(
    node_registry: Option<NodeRegistryConfig>,
    control_bus: Option<ControlBusConfig>,
) -> Option<ReconcileConfig> {
    let node_registry = node_registry?;
    if node_registry.etcd_endpoints.is_empty() {
        return None;
    }

    let (nats_url, nats_subject_prefix, node_id) = if let Some(control) = control_bus {
        (
            Some(control.nats_url),
            control.subject_prefix,
            if control.node_id.trim().is_empty() {
                node_registry.node_id.clone()
            } else {
                control.node_id
            },
        )
    } else {
        (
            None,
            "reson.sandbox.control".to_string(),
            node_registry.node_id.clone(),
        )
    };

    Some(ReconcileConfig {
        etcd_endpoints: node_registry.etcd_endpoints,
        etcd_prefix: node_registry.key_prefix,
        node_id,
        node_endpoint: node_registry.advertise_endpoint,
        interval: Duration::from_secs(30),
        nats_url,
        nats_subject_prefix,
    })
}

async fn reconcile_once(
    manager: &Manager,
    config: &ReconcileConfig,
    nats: Option<&async_nats::Client>,
) -> Result<()> {
    let local_sessions = collect_local_sessions(manager).await;
    let mut client = EtcdClient::connect(config.etcd_endpoints.clone(), None)
        .await
        .context("connect etcd for reconciliation")?;

    let sessions_prefix = format!("{}/sessions/", config.etcd_prefix.trim_end_matches('/'));
    let response = client
        .get(sessions_prefix, Some(GetOptions::new().with_prefix()))
        .await
        .context("read session routes for reconciliation")?;

    let routes = response
        .kvs()
        .iter()
        .filter_map(|kv| {
            let key = String::from_utf8(kv.key().to_vec()).ok()?;
            decode_route_entry(key, kv.value()).ok()
        })
        .collect::<Vec<_>>();

    let plan = plan_reconcile(
        &config.node_endpoint,
        &config.node_id,
        &local_sessions,
        routes,
    );

    for key in &plan.delete_keys {
        let _ = client.delete(key.clone(), None).await;
    }
    for route in &plan.upserts {
        let key = format!(
            "{}/sessions/{}",
            config.etcd_prefix.trim_end_matches('/'),
            route.session_id
        );
        let payload = serde_json::to_vec(&json!({
            "session_id": route.session_id,
            "vm_id": route.vm_id,
            "endpoint": route.endpoint,
            "node_id": route.node_id,
            "fork_id": route.fork_id,
            "updated_at_unix_ms": unix_millis(),
        }))
        .context("serialize route upsert payload")?;
        client
            .put(key, payload, None)
            .await
            .context("upsert reconciled session route")?;
    }

    let summary = json!({
        "node_id": config.node_id,
        "endpoint": config.node_endpoint,
        "deleted": plan.delete_keys.len(),
        "upserted": plan.upserts.len(),
        "updated_at_unix_ms": unix_millis(),
    });
    let reconcile_key = format!(
        "{}/reconcile/{}/{}",
        config.etcd_prefix.trim_end_matches('/'),
        config.node_id,
        unix_millis()
    );
    let _ = client
        .put(reconcile_key, summary.to_string(), None)
        .await
        .context("write reconcile summary")?;

    if let Some(nats) = nats {
        let subject = format!("{}.evt.reconcile.completed", config.nats_subject_prefix);
        let _ = nats.publish(subject, summary.to_string().into()).await;
    }

    debug!(
        node_id = %config.node_id,
        endpoint = %config.node_endpoint,
        upserted = plan.upserts.len(),
        deleted = plan.delete_keys.len(),
        "reconciliation run complete"
    );

    Ok(())
}

async fn collect_local_sessions(manager: &Manager) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for vm in manager.list().await {
        if let Some(session_id) = vm.metadata.get(META_SESSION_ID) {
            out.insert(session_id.clone(), vm.id.clone());
        }
    }
    out
}

fn decode_route_entry(key: String, raw: &[u8]) -> Result<SessionRouteEntry> {
    if let Ok(mut route) = serde_json::from_slice::<SessionRoute>(raw) {
        if route.session_id.trim().is_empty() {
            route.session_id = key.rsplit('/').next().unwrap_or_default().to_string();
        }
        return Ok(SessionRouteEntry { key, route });
    }

    let endpoint = String::from_utf8(raw.to_vec()).context("decode legacy route endpoint")?;
    let session_id = key.rsplit('/').next().unwrap_or_default().to_string();
    Ok(SessionRouteEntry {
        key,
        route: SessionRoute {
            session_id,
            vm_id: String::new(),
            endpoint,
            node_id: None,
            fork_id: None,
        },
    })
}

fn plan_reconcile(
    node_endpoint: &str,
    node_id: &str,
    local_sessions: &HashMap<String, String>,
    remote_routes: Vec<SessionRouteEntry>,
) -> ReconcilePlan {
    let mut plan = ReconcilePlan::default();
    let mut seen_valid_sessions = HashMap::new();

    for entry in remote_routes {
        if entry.route.endpoint != node_endpoint {
            continue;
        }
        match local_sessions.get(&entry.route.session_id) {
            Some(vm_id) if *vm_id == entry.route.vm_id => {
                seen_valid_sessions.insert(entry.route.session_id, true);
            }
            _ => {
                plan.delete_keys.push(entry.key);
            }
        }
    }

    for (session_id, vm_id) in local_sessions {
        if seen_valid_sessions.contains_key(session_id) {
            continue;
        }
        plan.upserts.push(SessionRoute {
            session_id: session_id.clone(),
            vm_id: vm_id.clone(),
            endpoint: node_endpoint.to_string(),
            node_id: Some(node_id.to_string()),
            fork_id: None,
        });
    }

    plan
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconcile_plan_upserts_missing_and_deletes_stale() {
        let mut local = HashMap::new();
        local.insert("s1".to_string(), "vm1".to_string());
        local.insert("s2".to_string(), "vm2".to_string());

        let routes = vec![
            SessionRouteEntry {
                key: "/reson-sandbox/sessions/s1".to_string(),
                route: SessionRoute {
                    session_id: "s1".to_string(),
                    vm_id: "vm1".to_string(),
                    endpoint: "http://node-a".to_string(),
                    node_id: Some("node-a".to_string()),
                    fork_id: None,
                },
            },
            SessionRouteEntry {
                key: "/reson-sandbox/sessions/s_stale".to_string(),
                route: SessionRoute {
                    session_id: "s_stale".to_string(),
                    vm_id: "vm-stale".to_string(),
                    endpoint: "http://node-a".to_string(),
                    node_id: Some("node-a".to_string()),
                    fork_id: None,
                },
            },
        ];

        let plan = plan_reconcile("http://node-a", "node-a", &local, routes);
        assert_eq!(plan.delete_keys, vec!["/reson-sandbox/sessions/s_stale"]);
        assert_eq!(plan.upserts.len(), 1);
        assert_eq!(plan.upserts[0].session_id, "s2");
        assert_eq!(plan.upserts[0].vm_id, "vm2");
        assert_eq!(plan.upserts[0].endpoint, "http://node-a");
    }
}
