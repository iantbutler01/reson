// @dive-file: Runs distributed control-bus command consumption with idempotency, ownership fencing, and failure handling.
// @dive-rel: Consumes ControlBusConfig from vmd/src/config.rs and enforces bounded in-flight command behavior.
// @dive-rel: Publishes replay/dead-letter/overload signals consumed by operational gates in scripts/verify_reson_sandbox.sh.
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use async_nats::jetstream;
use async_nats::jetstream::AckKind;
use async_nats::jetstream::consumer::{AckPolicy, pull};
use etcd_client::{Client as EtcdClient, Compare, CompareOp, Txn, TxnOp};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::{Mutex, Semaphore, mpsc, oneshot};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::ControlBusConfig;
use crate::partition::PartitionGate;

pub struct CommandConsumerHandle {
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl CommandConsumerHandle {
    pub async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

#[derive(Debug, Deserialize)]
struct CommandEnvelope {
    #[serde(default)]
    command_id: String,
    #[serde(default)]
    idempotency_key: String,
    #[serde(default)]
    command_type: String,
    #[serde(default)]
    ordering_key: String,
    #[serde(default)]
    expected_fence: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DeadLetterEnvelope {
    dead_letter_id: String,
    #[serde(default)]
    command_id: String,
    #[serde(default)]
    original_subject: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    details: String,
    #[serde(default)]
    delivered: i64,
    #[serde(default)]
    node_id: String,
    #[serde(default)]
    captured_at_unix_ms: u64,
    payload: Value,
}

#[derive(Clone)]
struct EtcdDedupeStore {
    etcd: std::sync::Arc<Mutex<EtcdClient>>,
    key_prefix: String,
}

#[derive(Clone)]
struct EtcdOwnershipFenceStore {
    etcd: std::sync::Arc<Mutex<EtcdClient>>,
    key_prefix: String,
}

impl EtcdDedupeStore {
    async fn connect(config: &ControlBusConfig) -> Result<Option<Self>> {
        if config.dedupe_etcd_endpoints.is_empty() {
            return Ok(None);
        }
        let client = EtcdClient::connect(config.dedupe_etcd_endpoints.clone(), None)
            .await
            .context("connect etcd for command dedupe")?;
        Ok(Some(Self {
            etcd: std::sync::Arc::new(Mutex::new(client)),
            key_prefix: config.dedupe_prefix.trim_end_matches('/').to_string(),
        }))
    }

    async fn mark_or_duplicate(&self, idempotency_key: &str) -> Result<bool> {
        let key = format!("{}/{}", self.key_prefix, idempotency_key);
        let mut client = self.etcd.lock().await;
        let txn = Txn::new()
            .when(vec![Compare::version(key.clone(), CompareOp::Equal, 0)])
            .and_then(vec![TxnOp::put(key, b"1", None)]);
        let response = client.txn(txn).await.context("dedupe txn")?;
        Ok(!response.succeeded())
    }
}

impl EtcdOwnershipFenceStore {
    async fn connect(config: &ControlBusConfig) -> Result<Option<Self>> {
        if config.dedupe_etcd_endpoints.is_empty() {
            return Ok(None);
        }
        let client = EtcdClient::connect(config.dedupe_etcd_endpoints.clone(), None)
            .await
            .context("connect etcd for ownership fences")?;
        Ok(Some(Self {
            etcd: std::sync::Arc::new(Mutex::new(client)),
            key_prefix: format!(
                "{}/ownership-fences",
                config.dedupe_prefix.trim_end_matches('/')
            ),
        }))
    }

    async fn check_and_rotate(
        &self,
        ordering_key: &str,
        expected_fence: Option<&str>,
    ) -> Result<String> {
        let key = format!(
            "{}/{}",
            self.key_prefix,
            sanitize_key_component(ordering_key)
        );
        let mut client = self.etcd.lock().await;
        let current = read_fence_value(&mut client, &key).await?;
        if !ownership_fence_allows_transition(current.as_deref(), expected_fence) {
            let current_display = current.as_deref().unwrap_or("<none>");
            let expected_display = expected_fence.unwrap_or("<none>");
            return Err(anyhow!(
                "ownership fence mismatch for ordering_key={ordering_key}: expected={expected_display} current={current_display}"
            ));
        }

        let next_fence = Uuid::new_v4().to_string();
        let compare = match expected_fence {
            Some(expected) => Compare::value(key.clone(), CompareOp::Equal, expected),
            None => Compare::version(key.clone(), CompareOp::Equal, 0),
        };
        let txn = Txn::new().when(vec![compare]).and_then(vec![TxnOp::put(
            key.clone(),
            next_fence.clone(),
            None,
        )]);
        let response = client.txn(txn).await.context("ownership fence txn")?;
        if !response.succeeded() {
            let latest = read_fence_value(&mut client, &key).await?;
            let current_display = latest.as_deref().unwrap_or("<none>");
            let expected_display = expected_fence.unwrap_or("<none>");
            return Err(anyhow!(
                "ownership fence compare-and-swap failed: expected={expected_display} current={current_display}"
            ));
        }
        Ok(next_fence)
    }
}

pub async fn start(config: Option<ControlBusConfig>) -> Result<Option<CommandConsumerHandle>> {
    start_with_trigger(config, None, None).await
}

pub async fn start_with_trigger(
    config: Option<ControlBusConfig>,
    reconcile_trigger: Option<mpsc::UnboundedSender<()>>,
    partition_gate: Option<PartitionGate>,
) -> Result<Option<CommandConsumerHandle>> {
    let Some(config) = config else {
        return Ok(None);
    };

    let nats = async_nats::connect(config.nats_url.clone())
        .await
        .context("connect nats for command consumer")?;
    let jetstream = jetstream::new(nats);
    ensure_control_stream(&jetstream, &config).await?;

    let command_subject = format!("{}.cmd.>", config.subject_prefix);
    let control_stream = jetstream
        .get_stream(config.stream_name.clone())
        .await
        .context("get control stream")?;
    let consumer = control_stream
        .get_or_create_consumer(
            &config.command_consumer_durable,
            pull::Config {
                durable_name: Some(config.command_consumer_durable.clone()),
                ack_policy: AckPolicy::Explicit,
                ack_wait: Duration::from_millis(config.command_ack_wait_ms),
                max_deliver: config.command_max_deliver,
                filter_subject: command_subject.clone(),
                max_ack_pending: 1_024,
                ..Default::default()
            },
        )
        .await
        .context("get or create command consumer")?;
    let mut messages = consumer
        .messages()
        .await
        .context("start command consumer stream")?;

    let dedupe_store = EtcdDedupeStore::connect(&config).await?;
    let fence_store = EtcdOwnershipFenceStore::connect(&config).await?;
    let node_id = config.node_id.clone();
    let log_node_id = config.node_id.clone();
    let log_nats_url = config.nats_url.clone();
    let log_stream_name = config.stream_name.clone();
    let log_durable = config.command_consumer_durable.clone();
    let log_max_deliver = config.command_max_deliver;
    let log_dead_letter_subject = config.dead_letter_subject.clone();
    let log_dedupe_endpoint_count = config.dedupe_etcd_endpoints.len();
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    // @dive: Dedupe state is shared across spawned command workers so duplicate suppression remains cluster-node local and bounded by TTL.
    let seen_commands: std::sync::Arc<Mutex<HashMap<String, Instant>>> =
        std::sync::Arc::new(Mutex::new(HashMap::new()));
    // @dive: Enforces a hard bound on in-flight control commands; overload is signaled deterministically via NAK+retry hint.
    let inflight_limit = std::sync::Arc::new(Semaphore::new(config.max_inflight_commands));

    let join = tokio::spawn(async move {
        const DEDUPE_TTL: Duration = Duration::from_secs(600);
        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                maybe_msg = messages.next() => {
                    let Some(maybe_msg) = maybe_msg else {
                        break;
                    };
                    match maybe_msg {
                        Ok(message) => {
                            let permit = match inflight_limit.clone().try_acquire_owned() {
                                Ok(permit) => permit,
                                Err(_) => {
                                    handle_overloaded_command_message(
                                        &message,
                                        &config,
                                        &node_id,
                                        &jetstream,
                                    )
                                    .await;
                                    continue;
                                }
                            };
                            let config = config.clone();
                            let node_id = node_id.clone();
                            let dedupe_store = dedupe_store.clone();
                            let fence_store = fence_store.clone();
                            let partition_gate = partition_gate.clone();
                            let seen_commands = seen_commands.clone();
                            let reconcile_trigger = reconcile_trigger.clone();
                            let jetstream = jetstream.clone();
                            tokio::spawn(async move {
                                let _permit = permit;
                                process_command_message(
                                    message,
                                    &config,
                                    &node_id,
                                    dedupe_store.as_ref(),
                                    fence_store.as_ref(),
                                    partition_gate.as_ref(),
                                    &seen_commands,
                                    reconcile_trigger.as_ref(),
                                    &jetstream,
                                    DEDUPE_TTL,
                                )
                                .await;
                            });
                        }
                        Err(err) => {
                            warn!(
                                node_id = %node_id,
                                err = %err,
                                "command consumer stream yielded an error"
                            );
                        }
                    }
                }
            }
        }
    });

    info!(
        node_id = %log_node_id,
        nats_url = %log_nats_url,
        stream_name = %log_stream_name,
        subject = %command_subject,
        durable = %log_durable,
        max_deliver = log_max_deliver,
        dead_letter_subject = %log_dead_letter_subject,
        dedupe_etcd_endpoints = log_dedupe_endpoint_count,
        "control command consumer enabled"
    );

    Ok(Some(CommandConsumerHandle {
        stop_tx: Some(stop_tx),
        join: Some(join),
    }))
}

pub async fn replay_dead_letters(config: ControlBusConfig, limit: usize) -> Result<usize> {
    if limit == 0 {
        return Ok(0);
    }

    let nats = async_nats::connect(config.nats_url.clone())
        .await
        .context("connect nats for dead-letter replay")?;
    let jetstream = jetstream::new(nats);
    ensure_control_stream(&jetstream, &config).await?;

    let control_stream = jetstream
        .get_stream(config.stream_name.clone())
        .await
        .context("get control stream for replay")?;
    let replay_consumer_name = format!("{}-dlq-replay", config.command_consumer_durable);
    let consumer = control_stream
        .get_or_create_consumer(
            &replay_consumer_name,
            pull::Config {
                durable_name: Some(replay_consumer_name.clone()),
                ack_policy: AckPolicy::Explicit,
                ack_wait: Duration::from_millis(config.command_ack_wait_ms),
                max_deliver: config.command_max_deliver,
                filter_subject: config.dead_letter_subject.clone(),
                max_ack_pending: 256,
                ..Default::default()
            },
        )
        .await
        .context("get or create dead-letter replay consumer")?;

    let mut messages = consumer
        .messages()
        .await
        .context("start dead-letter replay stream")?;

    let mut replayed = 0usize;
    let mut idle_timeouts = 0usize;
    while replayed < limit {
        let next = tokio::time::timeout(Duration::from_millis(500), messages.next()).await;
        let maybe_msg = match next {
            Ok(value) => {
                idle_timeouts = 0;
                value
            }
            Err(_) => {
                idle_timeouts += 1;
                if idle_timeouts >= 3 {
                    break;
                }
                continue;
            }
        };

        let Some(maybe_msg) = maybe_msg else {
            break;
        };

        let message = match maybe_msg {
            Ok(message) => message,
            Err(err) => {
                warn!(err = %err, "dead-letter replay stream yielded error");
                continue;
            }
        };

        match serde_json::from_slice::<DeadLetterEnvelope>(&message.payload) {
            Ok(envelope) => {
                if envelope.original_subject.trim().is_empty() {
                    warn!("dead-letter envelope missing original subject; dropping");
                    let _ = message.ack_with(AckKind::Term).await;
                    continue;
                }

                let payload_bytes =
                    serde_json::to_vec(&envelope.payload).context("serialize replay payload")?;
                let replay_id = if envelope.command_id.trim().is_empty() {
                    format!("replay-{}", Uuid::new_v4())
                } else {
                    format!("replay-{}-{}", envelope.command_id, unix_millis())
                };

                let publish_ack = jetstream
                    .send_publish(
                        envelope.original_subject.clone(),
                        jetstream::context::Publish::build()
                            .message_id(replay_id)
                            .payload(payload_bytes.into()),
                    )
                    .await
                    .context("publish replay command")?;
                publish_ack.await.context("await replay publish ack")?;

                let replay_event = json!({
                    "replayed_at_unix_ms": unix_millis(),
                    "dead_letter_id": envelope.dead_letter_id,
                    "command_id": envelope.command_id,
                    "original_subject": envelope.original_subject,
                });
                let replay_event_payload =
                    serde_json::to_vec(&replay_event).context("serialize replay event payload")?;
                let event_ack = jetstream
                    .publish(config.replay_subject.clone(), replay_event_payload.into())
                    .await
                    .context("publish replay event")?;
                event_ack.await.context("await replay event ack")?;

                message
                    .ack()
                    .await
                    .map_err(|err| anyhow!("ack dead-letter message: {err}"))?;
                replayed += 1;
            }
            Err(err) => {
                warn!(err = %err, "invalid dead-letter payload; terminating message");
                let _ = message.ack_with(AckKind::Term).await;
            }
        }
    }

    Ok(replayed)
}

#[allow(clippy::too_many_arguments)]
async fn process_command_message(
    message: jetstream::Message,
    config: &ControlBusConfig,
    node_id: &str,
    dedupe_store: Option<&EtcdDedupeStore>,
    fence_store: Option<&EtcdOwnershipFenceStore>,
    partition_gate: Option<&PartitionGate>,
    seen_commands: &std::sync::Arc<Mutex<HashMap<String, Instant>>>,
    reconcile_trigger: Option<&mpsc::UnboundedSender<()>>,
    jetstream: &jetstream::Context,
    dedupe_ttl: Duration,
) {
    let parsed = serde_json::from_slice::<CommandEnvelope>(&message.payload);
    let envelope = match parsed {
        Ok(envelope) => envelope,
        Err(err) => {
            handle_failed_command_message(
                &message,
                config,
                node_id,
                jetstream,
                "invalid_envelope",
                &err.to_string(),
            )
            .await;
            return;
        }
    };

    if let Some(gate) = partition_gate {
        if !gate.mutation_allowed().await {
            let reason = gate.mutation_rejection_reason().await.unwrap_or_else(|| {
                "network partition fail-closed: rejecting mutating commands".to_string()
            });
            warn!(
                node_id = %node_id,
                subject = %message.subject,
                command_id = %envelope.command_id,
                command_type = %envelope.command_type,
                reason = %reason,
                "rejecting control command while quorum visibility is lost"
            );
            if let Err(err) = message
                .ack_with(AckKind::Nak(Some(gate.command_retry_delay())))
                .await
            {
                warn!(
                    node_id = %node_id,
                    command_id = %envelope.command_id,
                    err = %err,
                    "failed to nack control command during partition fail-closed enforcement"
                );
            }
            return;
        }
    }

    let now = Instant::now();

    let dedupe_key = dedupe_key(&envelope);
    if let Some(store) = dedupe_store {
        match store.mark_or_duplicate(&dedupe_key).await {
            Ok(true) => {
                debug!(
                    node_id = %node_id,
                    subject = %message.subject,
                    command_id = %envelope.command_id,
                    idempotency_key = %dedupe_key,
                    "dropping duplicate control command (etcd dedupe)"
                );
                let _ = message.ack().await;
                return;
            }
            Ok(false) => {}
            Err(err) => {
                warn!(
                    node_id = %node_id,
                    subject = %message.subject,
                    idempotency_key = %dedupe_key,
                    err = %err,
                    "etcd dedupe check failed; falling back to local dedupe"
                );
            }
        }
    }

    {
        let mut seen = seen_commands.lock().await;
        seen.retain(|_, ts| now.duration_since(*ts) < dedupe_ttl);
        if seen.contains_key(&dedupe_key) {
            debug!(
                node_id = %node_id,
                subject = %message.subject,
                command_id = %envelope.command_id,
                idempotency_key = %dedupe_key,
                "dropping duplicate control command"
            );
            let _ = message.ack().await;
            return;
        }
        seen.insert(dedupe_key.clone(), now);
    }

    if let Some(store) = fence_store {
        let ordering_key = ownership_scope_key(&envelope);
        if let Err(err) = store
            .check_and_rotate(&ordering_key, envelope.expected_fence.as_deref())
            .await
        {
            handle_failed_command_message(
                &message,
                config,
                node_id,
                jetstream,
                "ownership_fence_conflict",
                &err.to_string(),
            )
            .await;
            return;
        }
    }

    debug!(
        node_id = %node_id,
        subject = %message.subject,
        command_id = %envelope.command_id,
        idempotency_key = %dedupe_key,
        command_type = %envelope.command_type,
        ordering_key = %envelope.ordering_key,
        "received control command"
    );

    if envelope.command_type == "reconcile.run" {
        if let Some(tx) = reconcile_trigger {
            if tx.send(()).is_err() {
                warn!(
                    node_id = %node_id,
                    command_id = %envelope.command_id,
                    "reconcile trigger receiver dropped"
                );
            }
        }
    }

    if let Err(err) = message.ack().await {
        warn!(
            node_id = %node_id,
            command_id = %envelope.command_id,
            err = %err,
            "failed to ack control command"
        );
    }
}

async fn handle_overloaded_command_message(
    message: &jetstream::Message,
    config: &ControlBusConfig,
    node_id: &str,
    jetstream: &jetstream::Context,
) {
    let retry_after_ms = config.overload_retry_after_ms.max(1);
    let details = format!("ResourceExhausted retry_after_ms={retry_after_ms}");
    let payload = json!({
        "node_id": node_id,
        "subject": message.subject.to_string(),
        "reason": "resource_exhausted",
        "retry_after_ms": retry_after_ms,
        "captured_at_unix_ms": unix_millis(),
    });
    let subject = format!("{}.evt.command.overloaded", config.subject_prefix);
    if let Ok(bytes) = serde_json::to_vec(&payload) {
        if let Ok(ack) = jetstream.publish(subject, bytes.into()).await {
            let _ = ack.await;
        }
    }
    if let Err(err) = message
        .ack_with(AckKind::Nak(Some(Duration::from_millis(retry_after_ms))))
        .await
    {
        warn!(
            node_id = %node_id,
            err = %err,
            details = %details,
            "failed to nack overloaded control command"
        );
    }
}

async fn handle_failed_command_message(
    message: &jetstream::Message,
    config: &ControlBusConfig,
    node_id: &str,
    jetstream: &jetstream::Context,
    reason: &str,
    details: &str,
) {
    let delivered = message
        .info()
        .map(|info| info.delivered)
        .unwrap_or(1)
        .max(1);
    let max_deliver = config.command_max_deliver.max(1);

    if delivered >= max_deliver {
        if let Err(err) =
            publish_dead_letter(message, config, node_id, jetstream, reason, details).await
        {
            warn!(
                node_id = %node_id,
                reason = %reason,
                err = %err,
                "failed publishing control command dead-letter"
            );
        }
        if let Err(err) = message.ack_with(AckKind::Term).await {
            warn!(
                node_id = %node_id,
                reason = %reason,
                err = %err,
                "failed to terminate poison control command"
            );
        }
        return;
    }

    if let Err(err) = message.ack_with(AckKind::Nak(None)).await {
        warn!(
            node_id = %node_id,
            reason = %reason,
            err = %err,
            "failed to nack control command"
        );
    }
}

async fn publish_dead_letter(
    message: &jetstream::Message,
    config: &ControlBusConfig,
    node_id: &str,
    jetstream: &jetstream::Context,
    reason: &str,
    details: &str,
) -> Result<()> {
    let delivered = message
        .info()
        .map(|info| info.delivered)
        .unwrap_or(1)
        .max(1);
    let payload = serde_json::from_slice::<Value>(&message.payload).unwrap_or_else(|_| {
        json!({
            "raw_payload": String::from_utf8_lossy(&message.payload).to_string(),
        })
    });

    let dead_letter = DeadLetterEnvelope {
        dead_letter_id: Uuid::new_v4().to_string(),
        command_id: serde_json::from_slice::<CommandEnvelope>(&message.payload)
            .map(|envelope| envelope.command_id)
            .unwrap_or_default(),
        original_subject: message.subject.to_string(),
        reason: reason.to_string(),
        details: details.to_string(),
        delivered,
        node_id: node_id.to_string(),
        captured_at_unix_ms: unix_millis(),
        payload,
    };
    let bytes = serde_json::to_vec(&dead_letter).context("serialize dead-letter envelope")?;

    let publish_ack = jetstream
        .send_publish(
            config.dead_letter_subject.clone(),
            jetstream::context::Publish::build()
                .message_id(dead_letter.dead_letter_id.clone())
                .payload(bytes.into()),
        )
        .await
        .context("publish dead-letter envelope")?;
    publish_ack
        .await
        .context("await dead-letter publish acknowledgement")?;

    Ok(())
}

async fn ensure_control_stream(
    jetstream: &jetstream::Context,
    config: &ControlBusConfig,
) -> Result<()> {
    let mut subjects = vec![
        format!("{}.cmd.>", config.subject_prefix),
        format!("{}.evt.>", config.subject_prefix),
        config.dead_letter_subject.clone(),
        config.replay_subject.clone(),
    ];
    subjects.sort();
    subjects.dedup();

    jetstream
        .get_or_create_stream(jetstream::stream::Config {
            name: config.stream_name.clone(),
            subjects,
            max_age: Duration::from_secs(config.stream_max_age_secs.max(60)),
            storage: jetstream::stream::StorageType::File,
            num_replicas: config.stream_replicas.max(1),
            ..Default::default()
        })
        .await
        .context("ensure control stream")?;

    Ok(())
}

fn dedupe_key(envelope: &CommandEnvelope) -> String {
    if !envelope.idempotency_key.trim().is_empty() {
        return envelope.idempotency_key.trim().to_string();
    }
    if !envelope.command_id.trim().is_empty() {
        return envelope.command_id.trim().to_string();
    }
    format!("anon-{}", Uuid::new_v4())
}

fn ownership_scope_key(envelope: &CommandEnvelope) -> String {
    if !envelope.ordering_key.trim().is_empty() {
        return envelope.ordering_key.trim().to_string();
    }
    if !envelope.command_id.trim().is_empty() {
        return envelope.command_id.trim().to_string();
    }
    format!("anon-owner-{}", Uuid::new_v4())
}

fn ownership_fence_allows_transition(current: Option<&str>, expected: Option<&str>) -> bool {
    match expected {
        Some(expected) => current.is_some_and(|value| value == expected),
        None => current.is_none(),
    }
}

fn sanitize_key_component(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "key".to_string()
    } else {
        trimmed.to_string()
    }
}

async fn read_fence_value(client: &mut EtcdClient, key: &str) -> Result<Option<String>> {
    let response = client
        .get(key.to_string(), None)
        .await
        .context("read ownership fence key")?;
    let Some(kv) = response.kvs().first() else {
        return Ok(None);
    };
    let value = String::from_utf8(kv.value().to_vec()).context("decode ownership fence value")?;
    if value.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(value))
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
    fn ownership_scope_key_prefers_ordering_key() {
        let envelope = CommandEnvelope {
            command_id: "command-1".to_string(),
            idempotency_key: String::new(),
            command_type: "session.attach".to_string(),
            ordering_key: "session-1".to_string(),
            expected_fence: None,
        };
        assert_eq!(ownership_scope_key(&envelope), "session-1");
    }

    #[test]
    fn ownership_fence_transition_rejects_stale_expectation() {
        assert!(ownership_fence_allows_transition(None, None));
        assert!(!ownership_fence_allows_transition(Some("fence-1"), None));
        assert!(ownership_fence_allows_transition(
            Some("fence-1"),
            Some("fence-1")
        ));
        assert!(!ownership_fence_allows_transition(
            Some("fence-2"),
            Some("fence-1")
        ));
    }
}
