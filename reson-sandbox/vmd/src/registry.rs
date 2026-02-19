use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use etcd_client::{Client as EtcdClient, PutOptions};
use serde_json::json;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tracing::{info, warn};

use crate::config::NodeRegistryConfig;

pub struct NodeRegistryHandle {
    key: String,
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl NodeRegistryHandle {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }

    pub fn key(&self) -> &str {
        &self.key
    }
}

pub async fn start(config: Option<NodeRegistryConfig>) -> Result<Option<NodeRegistryHandle>> {
    let Some(config) = config else {
        return Ok(None);
    };

    let key = registry_key(&config);
    register_once(&config, &key)
        .await
        .with_context(|| format!("initial node registry write for key {key}"))?;

    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let task_config = config.clone();
    let task_key = key.clone();
    let interval = heartbeat_interval(task_config.ttl_secs);
    let join = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = &mut stop_rx => {
                    break;
                }
                _ = tokio::time::sleep(interval) => {
                    if let Err(err) = register_once(&task_config, &task_key).await {
                        warn!(
                            key = %task_key,
                            err = %err,
                            "node registry heartbeat write failed"
                        );
                    }
                }
            }
        }

        if let Err(err) = delete_key(&task_config, &task_key).await {
            warn!(
                key = %task_key,
                err = %err,
                "failed deleting node registry key on shutdown"
            );
        }
    });

    info!(
        key = %key,
        endpoint = %config.advertise_endpoint,
        ttl_secs = config.ttl_secs,
        "node registry heartbeat enabled"
    );

    Ok(Some(NodeRegistryHandle {
        key,
        stop_tx: Some(stop_tx),
        join: Some(join),
    }))
}

async fn register_once(config: &NodeRegistryConfig, key: &str) -> Result<()> {
    let mut client = EtcdClient::connect(config.etcd_endpoints.clone(), None)
        .await
        .context("connect etcd for node heartbeat")?;
    let lease = client
        .lease_grant(config.ttl_secs, None)
        .await
        .context("grant lease for node heartbeat")?;
    let lease_id = lease.id();
    let payload = json!({
        "node_id": config.node_id,
        "endpoint": config.advertise_endpoint,
        "updated_at_unix_ms": unix_millis(),
    })
    .to_string();

    client
        .put(key, payload, Some(PutOptions::new().with_lease(lease_id)))
        .await
        .context("put node heartbeat key")?;
    Ok(())
}

async fn delete_key(config: &NodeRegistryConfig, key: &str) -> Result<()> {
    let mut client = EtcdClient::connect(config.etcd_endpoints.clone(), None)
        .await
        .context("connect etcd for node key delete")?;
    client.delete(key, None).await.context("delete node key")?;
    Ok(())
}

fn registry_key(config: &NodeRegistryConfig) -> String {
    format!(
        "{}/nodes/{}",
        config.key_prefix.trim_end_matches('/'),
        config.node_id
    )
}

fn heartbeat_interval(ttl_secs: i64) -> Duration {
    let ttl = ttl_secs.max(2) as u64;
    let interval = (ttl / 2).max(1);
    Duration::from_secs(interval)
}

fn unix_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
