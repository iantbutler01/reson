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
use tokio_stream::wrappers::ReceiverStream;
use tonic::Request;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::ControlBusConfig;
use crate::partition::PartitionGate;
use crate::proto::bracket::portproxy::v1::daemon_manager_client::DaemonManagerClient;
use crate::proto::bracket::portproxy::v1::shell_exec_client::ShellExecClient;
use crate::proto::bracket::portproxy::v1::{
    AttachDaemonRequest, AttachDaemonResponse, AttachDaemonStart, ExecDaemonRequest, ExecRequest,
    ExecResponse, ExecStart, attach_daemon_request, attach_daemon_response, exec_request,
    exec_response,
};
use crate::state::{Manager, SnapshotParams, UpdateVmParams};

const META_EXEC_RESTORE_SNAPSHOT_ID: &str = "reson.execution_restore_snapshot_id";
const META_EXEC_RESTORE_SNAPSHOT_NAME: &str = "reson.execution_restore_snapshot_name";
const META_TIER_B_ELIGIBLE: &str = "reson.tier_b_eligible";

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
    #[serde(default)]
    target_node_id: Option<String>,
    #[serde(default)]
    payload: Value,
}

#[derive(Debug, Deserialize)]
struct ExecRunPayload {
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    vm_id: String,
    #[serde(default)]
    command: String,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    timeout_secs: Option<i32>,
    #[serde(default)]
    detach: bool,
    #[serde(default)]
    shell: Option<String>,
}

#[derive(Clone)]
struct ActiveExecStream {
    request_tx: mpsc::Sender<AttachDaemonRequest>,
}

#[derive(Debug, Deserialize)]
struct ExecStreamStartPayload {
    #[serde(default)]
    stream_id: String,
    #[serde(default)]
    logical_stream_id: String,
    #[serde(default)]
    cluster_id: String,
    #[serde(default)]
    producer_epoch: u64,
    #[serde(default)]
    resume_after_event_seq: u64,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    vm_id: String,
    #[serde(default)]
    command: String,
    #[serde(default)]
    env: HashMap<String, String>,
    #[serde(default)]
    timeout_secs: Option<i32>,
    #[serde(default)]
    detach: bool,
    #[serde(default)]
    shell: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExecStreamInputPayload {
    #[serde(default)]
    stream_id: String,
    #[serde(default)]
    input_seq: u64,
    #[serde(default)]
    input_kind: String,
    #[serde(default)]
    data: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
struct ExecRunResult {
    command_id: String,
    session_id: String,
    vm_id: String,
    stdout: String,
    stderr: String,
    exit_code: Option<i32>,
    timed_out: bool,
    error: Option<String>,
    executed_by_node_id: String,
    completed_at_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExecStreamEvent {
    cluster_id: String,
    logical_stream_id: String,
    stream_id: String,
    event_seq: u64,
    event_id: String,
    producer_epoch: u64,
    command_id: String,
    session_id: String,
    vm_id: String,
    kind: String,
    #[serde(default)]
    data: Vec<u8>,
    #[serde(default)]
    exit_code: Option<i32>,
    #[serde(default)]
    timed_out: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    sequence: u64,
    emitted_by_node_id: String,
    emitted_at_unix_ms: u64,
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

pub async fn start(
    config: Option<ControlBusConfig>,
    manager: std::sync::Arc<Manager>,
) -> Result<Option<CommandConsumerHandle>> {
    start_with_trigger(config, manager, None, None).await
}

pub async fn start_with_trigger(
    config: Option<ControlBusConfig>,
    manager: std::sync::Arc<Manager>,
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
    let active_exec_streams: std::sync::Arc<Mutex<HashMap<String, ActiveExecStream>>> =
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
                            let active_exec_streams = active_exec_streams.clone();
                            let reconcile_trigger = reconcile_trigger.clone();
                            let manager = std::sync::Arc::clone(&manager);
                            let jetstream = jetstream.clone();
                            tokio::spawn(async move {
                                let _permit = permit;
                                process_command_message(
                                    message,
                                    &config,
                                    &node_id,
                                    manager.as_ref(),
                                    dedupe_store.as_ref(),
                                    fence_store.as_ref(),
                                    partition_gate.as_ref(),
                                    &seen_commands,
                                    &active_exec_streams,
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
    manager: &Manager,
    dedupe_store: Option<&EtcdDedupeStore>,
    fence_store: Option<&EtcdOwnershipFenceStore>,
    partition_gate: Option<&PartitionGate>,
    seen_commands: &std::sync::Arc<Mutex<HashMap<String, Instant>>>,
    active_exec_streams: &std::sync::Arc<Mutex<HashMap<String, ActiveExecStream>>>,
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

    if let Some(target_node_id) = envelope
        .target_node_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        if target_node_id != node_id {
            // @dive: Node-target filtering must happen before dedupe/fence CAS so non-owner consumers never steal ownership transitions.
            debug!(
                node_id = %node_id,
                target_node_id = %target_node_id,
                command_id = %envelope.command_id,
                command_type = %envelope.command_type,
                "skipping control command targeted at a different node"
            );
            if let Err(err) = message.ack().await {
                warn!(
                    node_id = %node_id,
                    command_id = %envelope.command_id,
                    err = %err,
                    "failed to ack command targeted at different node"
                );
            }
            return;
        }
    }

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
    let delivery_attempt = message
        .info()
        .map(|info| info.delivered)
        .unwrap_or(1)
        .max(1);
    let is_redelivery = delivery_attempt > 1;

    let dedupe_key = dedupe_key(&envelope);
    if !is_redelivery {
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
                    if let Err(err) = message.ack().await {
                        warn!(
                            node_id = %node_id,
                            command_id = %envelope.command_id,
                            idempotency_key = %dedupe_key,
                            err = %err,
                            "failed to ack duplicate command after etcd dedupe"
                        );
                    }
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
                if let Err(err) = message.ack().await {
                    warn!(
                        node_id = %node_id,
                        command_id = %envelope.command_id,
                        idempotency_key = %dedupe_key,
                        err = %err,
                        "failed to ack duplicate command after local dedupe"
                    );
                }
                return;
            }
            seen.insert(dedupe_key.clone(), now);
        }
    } else {
        // @dive: Broker redeliveries must bypass duplicate suppression so failed commands can exhaust retry budget and reach DLQ.
        debug!(
            node_id = %node_id,
            subject = %message.subject,
            command_id = %envelope.command_id,
            idempotency_key = %dedupe_key,
            delivery_attempt = delivery_attempt,
            "processing control command redelivery without dedupe short-circuit"
        );
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
    } else if envelope.command_type == "exec.run" {
        if let Err(err) =
            handle_exec_run_command(&envelope, config, node_id, manager, jetstream).await
        {
            handle_failed_command_message(
                &message,
                config,
                node_id,
                jetstream,
                "exec_run_failed",
                &err.to_string(),
            )
            .await;
            return;
        }
    } else if envelope.command_type == "exec.stream.start" {
        if let Err(err) = handle_exec_stream_start_command(
            &envelope,
            config,
            node_id,
            manager,
            active_exec_streams,
            jetstream,
        )
        .await
        {
            handle_failed_command_message(
                &message,
                config,
                node_id,
                jetstream,
                "exec_stream_start_failed",
                &err.to_string(),
            )
            .await;
            return;
        }
    } else if envelope.command_type == "exec.stream.input" {
        if let Err(err) = handle_exec_stream_input_command(&envelope, active_exec_streams).await {
            handle_failed_command_message(
                &message,
                config,
                node_id,
                jetstream,
                "exec_stream_input_failed",
                &err.to_string(),
            )
            .await;
            return;
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

async fn handle_exec_run_command(
    envelope: &CommandEnvelope,
    config: &ControlBusConfig,
    node_id: &str,
    manager: &Manager,
    jetstream: &jetstream::Context,
) -> Result<()> {
    let payload: ExecRunPayload = serde_json::from_value(envelope.payload.clone())
        .context("decode exec.run command payload")?;
    if payload.vm_id.trim().is_empty() {
        return Err(anyhow!("exec.run payload missing vm_id"));
    }
    if payload.command.trim().is_empty() {
        return Err(anyhow!("exec.run payload missing command"));
    }

    let vm = manager
        .start_vm(payload.vm_id.as_str())
        .await
        .map_err(|err| anyhow!("ensure vm running for exec.run: {err}"))?;
    let rpc_port = vm.network.rpc_port;
    if rpc_port <= 0 {
        return Err(anyhow!(
            "exec.run vm {} missing rpc_port after start",
            payload.vm_id
        ));
    }
    let endpoint = format!("http://127.0.0.1:{rpc_port}");
    let mut client = ShellExecClient::connect(endpoint)
        .await
        .context("connect shell exec client for exec.run")?;

    let shell = payload.shell.unwrap_or_else(|| "bash".to_string());
    let (req_tx, req_rx) = mpsc::channel(8);
    req_tx
        .send(ExecRequest {
            request: Some(exec_request::Request::Start(ExecStart {
                args: vec![shell, "-lc".to_string(), payload.command.clone()],
                env: payload.env,
                detach: payload.detach,
                timeout: payload.timeout_secs,
            })),
        })
        .await
        .map_err(|_| anyhow!("enqueue exec.run start request"))?;
    drop(req_tx);

    let response = client
        .exec(Request::new(ReceiverStream::new(req_rx)))
        .await
        .context("invoke shell exec stream for exec.run")?;
    let mut stream = response.into_inner();

    let mut stdout = String::new();
    let mut stderr = String::new();
    let mut exit_code = None;
    let mut timed_out = false;

    while let Some(frame) = stream
        .message()
        .await
        .context("read exec.run stream frame")?
    {
        match frame {
            ExecResponse {
                response: Some(exec_response::Response::StdoutData(bytes)),
            } => {
                stdout.push_str(&String::from_utf8_lossy(&bytes));
            }
            ExecResponse {
                response: Some(exec_response::Response::StderrData(bytes)),
            } => {
                stderr.push_str(&String::from_utf8_lossy(&bytes));
            }
            ExecResponse {
                response: Some(exec_response::Response::ExitCode(code)),
            } => {
                if code == 124 {
                    timed_out = true;
                }
                exit_code = Some(code);
                break;
            }
            ExecResponse { response: None } => {}
        }
    }

    let command_id = if envelope.command_id.trim().is_empty() {
        Uuid::new_v4().to_string()
    } else {
        envelope.command_id.clone()
    };
    let event_subject = format!(
        "{}.evt.exec.result.{}",
        config.subject_prefix,
        sanitize_subject_token(command_id.as_str())
    );
    // @dive: Command-scoped result subjects let facade waits filter exact exec completions without scanning shared event traffic.
    let result = ExecRunResult {
        command_id,
        session_id: payload.session_id,
        vm_id: payload.vm_id,
        stdout,
        stderr,
        exit_code,
        timed_out,
        error: None,
        executed_by_node_id: node_id.to_string(),
        completed_at_unix_ms: unix_millis(),
    };
    let bytes = serde_json::to_vec(&result).context("serialize exec.run result event")?;
    let publish_ack = jetstream
        .publish(event_subject, bytes.into())
        .await
        .context("publish exec.run result event")?;
    publish_ack
        .await
        .context("await exec.run result event publish ack")?;
    Ok(())
}

async fn handle_exec_stream_start_command(
    envelope: &CommandEnvelope,
    config: &ControlBusConfig,
    node_id: &str,
    manager: &Manager,
    active_exec_streams: &std::sync::Arc<Mutex<HashMap<String, ActiveExecStream>>>,
    jetstream: &jetstream::Context,
) -> Result<()> {
    let payload: ExecStreamStartPayload = serde_json::from_value(envelope.payload.clone())
        .context("decode exec.stream.start command payload")?;
    let logical_stream_id = if payload.logical_stream_id.trim().is_empty() {
        payload.stream_id.trim().to_string()
    } else {
        payload.logical_stream_id.trim().to_string()
    };
    if logical_stream_id.is_empty() {
        return Err(anyhow!(
            "exec.stream.start payload missing logical_stream_id/stream_id"
        ));
    }
    if payload.vm_id.trim().is_empty() {
        return Err(anyhow!("exec.stream.start payload missing vm_id"));
    }
    let resume_only = payload.resume_after_event_seq > 0;
    if payload.command.trim().is_empty() && !resume_only {
        return Err(anyhow!("exec.stream.start payload missing command"));
    }
    // @dive: timeout/detach are carried for compatibility with the control payload; daemon attach mode currently enforces lifecycle via stream semantics.
    let _ignored_timeout_secs = payload.timeout_secs;
    let _ignored_detach = payload.detach;

    let stream_id = logical_stream_id.clone();
    let cluster_id = if payload.cluster_id.trim().is_empty() {
        config.cluster_id.clone()
    } else {
        payload.cluster_id.trim().to_string()
    };
    let producer_epoch = payload.producer_epoch;
    let command_id = if envelope.command_id.trim().is_empty() {
        Uuid::new_v4().to_string()
    } else {
        envelope.command_id.clone()
    };
    let session_id = payload.session_id.clone();
    let vm_id = payload.vm_id.clone();
    let mut sequence = payload.resume_after_event_seq;

    {
        let guard = active_exec_streams.lock().await;
        if guard.contains_key(&stream_id) {
            // @dive: Resume commands are idempotent by logical stream id; if producer is already active on this node we keep it and avoid rerun.
            info!(
                stream_id = %stream_id,
                command_id = %command_id,
                producer_epoch = producer_epoch,
                "exec.stream.start ignored because stream is already active"
            );
            return Ok(());
        }
    }

    let mut vm = manager
        .start_vm(payload.vm_id.as_str())
        .await
        .map_err(|err| anyhow!("ensure vm running for exec.stream.start: {err}"))?;
    let mut vm_metadata = vm.metadata.clone();
    let tier_b_eligible = metadata_tier_b_eligible(&vm_metadata);
    if resume_only && tier_b_eligible {
        let Some(snapshot_id) =
            resolve_execution_restore_snapshot_id(manager, payload.vm_id.as_str(), &vm_metadata)
                .await?
        else {
            // @dive: Tier-B resume cannot rerun commands; missing restore marker is emitted as a terminal stream error.
            sequence = sequence.saturating_add(1);
            publish_exec_stream_event(
                config,
                jetstream,
                ExecStreamEvent {
                    cluster_id: cluster_id.clone(),
                    logical_stream_id: logical_stream_id.clone(),
                    stream_id: stream_id.clone(),
                    event_seq: sequence,
                    event_id: next_stream_event_id(cluster_id.as_str()),
                    producer_epoch,
                    command_id: command_id.clone(),
                    session_id: session_id.clone(),
                    vm_id: vm_id.clone(),
                    kind: "error".to_string(),
                    data: Vec::new(),
                    exit_code: None,
                    timed_out: false,
                    error: Some(
                        "tier_b_eligible exec stream resume requires execution restore snapshot marker"
                            .to_string(),
                    ),
                    sequence,
                    emitted_by_node_id: node_id.to_string(),
                    emitted_at_unix_ms: unix_millis(),
                },
            )
            .await?;
            return Ok(());
        };
        match manager
            .restore_snapshot(payload.vm_id.as_str(), snapshot_id.as_str())
            .await
        {
            Ok(restored_vm) => {
                vm = restored_vm;
                vm_metadata = vm.metadata.clone();
                info!(
                    vm_id = %vm_id,
                    stream_id = %stream_id,
                    snapshot_id = %snapshot_id,
                    "restored tier-b execution snapshot for exec stream resume"
                );
            }
            Err(err) => {
                // @dive: Resume attach remains no-rerun; restore failure must terminate stream instead of replaying command.
                sequence = sequence.saturating_add(1);
                publish_exec_stream_event(
                    config,
                    jetstream,
                    ExecStreamEvent {
                        cluster_id: cluster_id.clone(),
                        logical_stream_id: logical_stream_id.clone(),
                        stream_id: stream_id.clone(),
                        event_seq: sequence,
                        event_id: next_stream_event_id(cluster_id.as_str()),
                        producer_epoch,
                        command_id: command_id.clone(),
                        session_id: session_id.clone(),
                        vm_id: vm_id.clone(),
                        kind: "error".to_string(),
                        data: Vec::new(),
                        exit_code: None,
                        timed_out: false,
                        error: Some(format!(
                            "tier-b execution snapshot restore failed for exec stream resume: {err}"
                        )),
                        sequence,
                        emitted_by_node_id: node_id.to_string(),
                        emitted_at_unix_ms: unix_millis(),
                    },
                )
                .await?;
                return Ok(());
            }
        }
    }
    let rpc_port = vm.network.rpc_port;
    if rpc_port <= 0 {
        return Err(anyhow!(
            "exec.stream.start vm {} missing rpc_port after start",
            payload.vm_id
        ));
    }
    let endpoint = format!("http://127.0.0.1:{rpc_port}");
    let mut daemon_client = DaemonManagerClient::connect(endpoint.clone())
        .await
        .context("connect daemon manager client for exec.stream.start")?;
    let daemon_name = format!("reson-exec-stream-{}", sanitize_subject_token(&stream_id));
    if !resume_only {
        let shell = payload.shell.clone().unwrap_or_else(|| "bash".to_string());
        daemon_client
            .exec_daemon(Request::new(ExecDaemonRequest {
                name: daemon_name.clone(),
                args: vec![shell, "-lc".to_string(), payload.command.clone()],
                env: payload.env.clone(),
            }))
            .await
            .context("invoke daemon manager for exec.stream.start")?;
    }

    let attach_retry_deadline = if resume_only {
        Some(tokio::time::Instant::now() + Duration::from_secs(20))
    } else {
        None
    };
    let (response, req_tx) = loop {
        let mut attach_client = DaemonManagerClient::connect(endpoint.clone())
            .await
            .context("connect daemon manager client for exec.stream attach")?;
        let (req_tx, req_rx) = mpsc::channel(64);
        req_tx
            .send(AttachDaemonRequest {
                request: Some(attach_daemon_request::Request::Start(AttachDaemonStart {
                    name: daemon_name.clone(),
                })),
            })
            .await
            .map_err(|_| anyhow!("enqueue exec.stream attach start request"))?;
        match attach_client
            .attach_daemon(Request::new(ReceiverStream::new(req_rx)))
            .await
        {
            Ok(response) => break (response, req_tx),
            Err(status) => {
                if resume_only && status.code() == tonic::Code::NotFound {
                    if let Some(deadline) = attach_retry_deadline {
                        if tokio::time::Instant::now() < deadline {
                            tokio::time::sleep(Duration::from_millis(500)).await;
                            continue;
                        }
                    }
                    // @dive: Resume path is attach-only to prevent rerunning non-idempotent commands when producer is already gone.
                    sequence = sequence.saturating_add(1);
                    publish_exec_stream_event(
                        config,
                        jetstream,
                        ExecStreamEvent {
                            cluster_id: cluster_id.clone(),
                            logical_stream_id: logical_stream_id.clone(),
                            stream_id: stream_id.clone(),
                            event_seq: sequence,
                            event_id: next_stream_event_id(cluster_id.as_str()),
                            producer_epoch,
                            command_id: command_id.clone(),
                            session_id: session_id.clone(),
                            vm_id: vm_id.clone(),
                            kind: "error".to_string(),
                            data: Vec::new(),
                            exit_code: None,
                            timed_out: false,
                            error: Some(format!(
                                "exec stream resume attach target missing for {} (producer not rerun)",
                                daemon_name
                            )),
                            sequence,
                            emitted_by_node_id: node_id.to_string(),
                            emitted_at_unix_ms: unix_millis(),
                        },
                    )
                    .await?;
                    return Ok(());
                }
                return Err(anyhow!(
                    "invoke daemon attach for exec.stream.start: {status}"
                ));
            }
        }
    };
    let mut stream = response.into_inner();

    {
        let mut guard = active_exec_streams.lock().await;
        guard.insert(
            stream_id.clone(),
            ActiveExecStream {
                request_tx: req_tx.clone(),
            },
        );
    }
    if tier_b_eligible {
        match refresh_execution_restore_snapshot_marker(manager, vm_id.as_str(), &vm_metadata).await
        {
            Ok((snapshot_id, snapshot_name)) => {
                info!(
                    vm_id = %vm_id,
                    stream_id = %stream_id,
                    snapshot_id = %snapshot_id,
                    snapshot_name = %snapshot_name,
                    "refreshed execution restore snapshot marker for tier-b exec stream"
                );
            }
            Err(err) => {
                warn!(
                    vm_id = %vm_id,
                    stream_id = %stream_id,
                    error = %err,
                    "failed refreshing execution restore snapshot marker for tier-b exec stream"
                );
            }
        }
    }
    info!(
        stream_id = %stream_id,
        command_id = %command_id,
        vm_id = %vm_id,
        producer_epoch = producer_epoch,
        resume_after_event_seq = payload.resume_after_event_seq,
        "exec.stream.start established"
    );

    sequence = sequence.saturating_add(1);
    publish_exec_stream_event(
        config,
        jetstream,
        ExecStreamEvent {
            cluster_id: cluster_id.clone(),
            logical_stream_id: logical_stream_id.clone(),
            stream_id: stream_id.clone(),
            event_seq: sequence,
            event_id: next_stream_event_id(cluster_id.as_str()),
            producer_epoch,
            command_id: command_id.clone(),
            session_id: session_id.clone(),
            vm_id: vm_id.clone(),
            kind: "started".to_string(),
            data: Vec::new(),
            exit_code: None,
            timed_out: false,
            error: None,
            sequence,
            emitted_by_node_id: node_id.to_string(),
            emitted_at_unix_ms: unix_millis(),
        },
    )
    .await?;

    let config = config.clone();
    let node_id = node_id.to_string();
    let jetstream = jetstream.clone();
    let active_exec_streams = active_exec_streams.clone();
    tokio::spawn(async move {
        let mut terminal_emitted = false;
        let mut pending_exit_code: Option<i32> = None;
        const EXIT_FLUSH_GRACE: Duration = Duration::from_millis(150);
        loop {
            let next_message = if let Some(code) = pending_exit_code {
                match tokio::time::timeout(EXIT_FLUSH_GRACE, stream.message()).await {
                    Ok(message) => message,
                    Err(_) => {
                        sequence = sequence.saturating_add(1);
                        debug!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            exit_code = code,
                            "exec.stream deferred exit frame"
                        );
                        if let Err(err) = publish_exec_stream_event(
                            &config,
                            &jetstream,
                            ExecStreamEvent {
                                cluster_id: cluster_id.clone(),
                                logical_stream_id: logical_stream_id.clone(),
                                stream_id: stream_id.clone(),
                                event_seq: sequence,
                                event_id: next_stream_event_id(cluster_id.as_str()),
                                producer_epoch,
                                command_id: command_id.clone(),
                                session_id: session_id.clone(),
                                vm_id: vm_id.clone(),
                                kind: "exit".to_string(),
                                data: Vec::new(),
                                exit_code: Some(code),
                                timed_out: code == 124,
                                error: None,
                                sequence,
                                emitted_by_node_id: node_id.clone(),
                                emitted_at_unix_ms: unix_millis(),
                            },
                        )
                        .await
                        {
                            warn!(
                                stream_id = %stream_id,
                                sequence = sequence,
                                err = %err,
                                "failed publishing deferred exec.stream exit event"
                            );
                        }
                        terminal_emitted = true;
                        break;
                    }
                }
            } else {
                stream.message().await
            };
            match next_message {
                Ok(Some(AttachDaemonResponse {
                    response: Some(attach_daemon_response::Response::StdoutData(bytes)),
                })) => {
                    sequence = sequence.saturating_add(1);
                    debug!(
                        stream_id = %stream_id,
                        sequence = sequence,
                        bytes = bytes.len(),
                        "exec.stream stdout frame"
                    );
                    if let Err(err) = publish_exec_stream_event(
                        &config,
                        &jetstream,
                        ExecStreamEvent {
                            cluster_id: cluster_id.clone(),
                            logical_stream_id: logical_stream_id.clone(),
                            stream_id: stream_id.clone(),
                            event_seq: sequence,
                            event_id: next_stream_event_id(cluster_id.as_str()),
                            producer_epoch,
                            command_id: command_id.clone(),
                            session_id: session_id.clone(),
                            vm_id: vm_id.clone(),
                            kind: "stdout".to_string(),
                            data: bytes,
                            exit_code: None,
                            timed_out: false,
                            error: None,
                            sequence,
                            emitted_by_node_id: node_id.clone(),
                            emitted_at_unix_ms: unix_millis(),
                        },
                    )
                    .await
                    {
                        warn!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            err = %err,
                            "failed publishing exec.stream stdout event"
                        );
                    }
                }
                Ok(Some(AttachDaemonResponse {
                    response: Some(attach_daemon_response::Response::StderrData(bytes)),
                })) => {
                    sequence = sequence.saturating_add(1);
                    debug!(
                        stream_id = %stream_id,
                        sequence = sequence,
                        bytes = bytes.len(),
                        "exec.stream stderr frame"
                    );
                    if let Err(err) = publish_exec_stream_event(
                        &config,
                        &jetstream,
                        ExecStreamEvent {
                            cluster_id: cluster_id.clone(),
                            logical_stream_id: logical_stream_id.clone(),
                            stream_id: stream_id.clone(),
                            event_seq: sequence,
                            event_id: next_stream_event_id(cluster_id.as_str()),
                            producer_epoch,
                            command_id: command_id.clone(),
                            session_id: session_id.clone(),
                            vm_id: vm_id.clone(),
                            kind: "stderr".to_string(),
                            data: bytes,
                            exit_code: None,
                            timed_out: false,
                            error: None,
                            sequence,
                            emitted_by_node_id: node_id.clone(),
                            emitted_at_unix_ms: unix_millis(),
                        },
                    )
                    .await
                    {
                        warn!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            err = %err,
                            "failed publishing exec.stream stderr event"
                        );
                    }
                }
                Ok(Some(AttachDaemonResponse {
                    response: Some(attach_daemon_response::Response::ExitCode(code)),
                })) => {
                    if code == 124 {
                        sequence = sequence.saturating_add(1);
                        debug!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            "exec.stream timeout frame"
                        );
                        if let Err(err) = publish_exec_stream_event(
                            &config,
                            &jetstream,
                            ExecStreamEvent {
                                cluster_id: cluster_id.clone(),
                                logical_stream_id: logical_stream_id.clone(),
                                stream_id: stream_id.clone(),
                                event_seq: sequence,
                                event_id: next_stream_event_id(cluster_id.as_str()),
                                producer_epoch,
                                command_id: command_id.clone(),
                                session_id: session_id.clone(),
                                vm_id: vm_id.clone(),
                                kind: "timeout".to_string(),
                                data: Vec::new(),
                                exit_code: None,
                                timed_out: true,
                                error: None,
                                sequence,
                                emitted_by_node_id: node_id.clone(),
                                emitted_at_unix_ms: unix_millis(),
                            },
                        )
                        .await
                        {
                            warn!(
                                stream_id = %stream_id,
                                sequence = sequence,
                                err = %err,
                                "failed publishing exec.stream timeout event"
                            );
                        }
                    }
                    pending_exit_code = Some(code);
                    continue;
                }
                Ok(Some(_)) => {}
                Ok(None) => {
                    if let Some(code) = pending_exit_code {
                        sequence = sequence.saturating_add(1);
                        debug!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            exit_code = code,
                            "exec.stream exit frame"
                        );
                        if let Err(err) = publish_exec_stream_event(
                            &config,
                            &jetstream,
                            ExecStreamEvent {
                                cluster_id: cluster_id.clone(),
                                logical_stream_id: logical_stream_id.clone(),
                                stream_id: stream_id.clone(),
                                event_seq: sequence,
                                event_id: next_stream_event_id(cluster_id.as_str()),
                                producer_epoch,
                                command_id: command_id.clone(),
                                session_id: session_id.clone(),
                                vm_id: vm_id.clone(),
                                kind: "exit".to_string(),
                                data: Vec::new(),
                                exit_code: Some(code),
                                timed_out: code == 124,
                                error: None,
                                sequence,
                                emitted_by_node_id: node_id.clone(),
                                emitted_at_unix_ms: unix_millis(),
                            },
                        )
                        .await
                        {
                            warn!(
                                stream_id = %stream_id,
                                sequence = sequence,
                                err = %err,
                                "failed publishing exec.stream exit event"
                            );
                        }
                        terminal_emitted = true;
                    }
                    debug!(
                        stream_id = %stream_id,
                        sequence = sequence,
                        "exec.stream response stream closed"
                    );
                    break;
                }
                Err(err) => {
                    if let Some(code) = pending_exit_code {
                        sequence = sequence.saturating_add(1);
                        debug!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            exit_code = code,
                            "exec.stream exit frame after transport close"
                        );
                        if let Err(err) = publish_exec_stream_event(
                            &config,
                            &jetstream,
                            ExecStreamEvent {
                                cluster_id: cluster_id.clone(),
                                logical_stream_id: logical_stream_id.clone(),
                                stream_id: stream_id.clone(),
                                event_seq: sequence,
                                event_id: next_stream_event_id(cluster_id.as_str()),
                                producer_epoch,
                                command_id: command_id.clone(),
                                session_id: session_id.clone(),
                                vm_id: vm_id.clone(),
                                kind: "exit".to_string(),
                                data: Vec::new(),
                                exit_code: Some(code),
                                timed_out: code == 124,
                                error: None,
                                sequence,
                                emitted_by_node_id: node_id.clone(),
                                emitted_at_unix_ms: unix_millis(),
                            },
                        )
                        .await
                        {
                            warn!(
                                stream_id = %stream_id,
                                sequence = sequence,
                                err = %err,
                                "failed publishing exec.stream exit event"
                            );
                        }
                        terminal_emitted = true;
                        break;
                    }
                    sequence = sequence.saturating_add(1);
                    warn!(
                        stream_id = %stream_id,
                        sequence = sequence,
                        err = %err,
                        "exec.stream transport error"
                    );
                    if let Err(err) = publish_exec_stream_event(
                        &config,
                        &jetstream,
                        ExecStreamEvent {
                            cluster_id: cluster_id.clone(),
                            logical_stream_id: logical_stream_id.clone(),
                            stream_id: stream_id.clone(),
                            event_seq: sequence,
                            event_id: next_stream_event_id(cluster_id.as_str()),
                            producer_epoch,
                            command_id: command_id.clone(),
                            session_id: session_id.clone(),
                            vm_id: vm_id.clone(),
                            kind: "error".to_string(),
                            data: Vec::new(),
                            exit_code: None,
                            timed_out: false,
                            error: Some(format!("exec stream transport error: {err}")),
                            sequence,
                            emitted_by_node_id: node_id.clone(),
                            emitted_at_unix_ms: unix_millis(),
                        },
                    )
                    .await
                    {
                        warn!(
                            stream_id = %stream_id,
                            sequence = sequence,
                            err = %err,
                            "failed publishing exec.stream error event"
                        );
                    }
                    terminal_emitted = true;
                    break;
                }
            }
        }

        if !terminal_emitted {
            sequence = sequence.saturating_add(1);
            if let Err(err) = publish_exec_stream_event(
                &config,
                &jetstream,
                ExecStreamEvent {
                    cluster_id: cluster_id.clone(),
                    logical_stream_id: logical_stream_id.clone(),
                    stream_id: stream_id.clone(),
                    event_seq: sequence,
                    event_id: next_stream_event_id(cluster_id.as_str()),
                    producer_epoch,
                    command_id: command_id.clone(),
                    session_id: session_id.clone(),
                    vm_id: vm_id.clone(),
                    kind: "error".to_string(),
                    data: Vec::new(),
                    exit_code: None,
                    timed_out: false,
                    error: Some("exec stream ended before terminal event".to_string()),
                    sequence,
                    emitted_by_node_id: node_id.clone(),
                    emitted_at_unix_ms: unix_millis(),
                },
            )
            .await
            {
                warn!(
                    stream_id = %stream_id,
                    sequence = sequence,
                    err = %err,
                    "failed publishing terminal exec.stream error event"
                );
            }
        }

        let mut guard = active_exec_streams.lock().await;
        guard.remove(&stream_id);
    });

    Ok(())
}

async fn handle_exec_stream_input_command(
    envelope: &CommandEnvelope,
    active_exec_streams: &std::sync::Arc<Mutex<HashMap<String, ActiveExecStream>>>,
) -> Result<()> {
    let payload: ExecStreamInputPayload = serde_json::from_value(envelope.payload.clone())
        .context("decode exec.stream.input command payload")?;
    if payload.stream_id.trim().is_empty() {
        return Err(anyhow!("exec.stream.input payload missing stream_id"));
    }

    let input_kind = payload.input_kind.trim().to_ascii_lowercase();
    match input_kind.as_str() {
        "stdin" => {
            debug!(
                stream_id = %payload.stream_id,
                input_seq = payload.input_seq,
                "exec.stream.input stdin chunk"
            );
            let active = {
                let guard = active_exec_streams.lock().await;
                guard.get(&payload.stream_id).cloned()
            };
            let active = active.ok_or_else(|| {
                anyhow!(
                    "exec.stream.input stream_id {} not found for stdin seq={}",
                    payload.stream_id,
                    payload.input_seq
                )
            })?;
            active
                .request_tx
                .send(AttachDaemonRequest {
                    request: Some(attach_daemon_request::Request::StdinData(
                        payload.data.unwrap_or_default(),
                    )),
                })
                .await
                .map_err(|_| {
                    anyhow!(
                        "exec.stream.input stream_id {} stdin channel closed",
                        payload.stream_id
                    )
                })?;
        }
        "eof" => {
            debug!(
                stream_id = %payload.stream_id,
                input_seq = payload.input_seq,
                "exec.stream.input eof"
            );
            let removed = {
                let mut guard = active_exec_streams.lock().await;
                guard.remove(&payload.stream_id)
            };
            if let Some(active) = removed {
                drop(active.request_tx);
            }
        }
        other => {
            return Err(anyhow!("unsupported exec.stream.input kind: {other}"));
        }
    }

    Ok(())
}

async fn publish_exec_stream_event(
    config: &ControlBusConfig,
    jetstream: &jetstream::Context,
    event: ExecStreamEvent,
) -> Result<()> {
    let logical_stream_id = if event.logical_stream_id.trim().is_empty() {
        event.stream_id.as_str()
    } else {
        event.logical_stream_id.as_str()
    };
    let subject = format!(
        "{}.evt.exec.stream.{}",
        config.subject_prefix,
        sanitize_subject_token(logical_stream_id)
    );
    let bytes = serde_json::to_vec(&event).context("serialize exec stream event")?;
    let publish_ack = jetstream
        .publish(subject, bytes.into())
        .await
        .context("publish exec stream event")?;
    publish_ack
        .await
        .context("await exec stream event publish ack")?;
    Ok(())
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

fn metadata_tier_b_eligible(metadata: &HashMap<String, String>) -> bool {
    let raw = metadata
        .get(META_TIER_B_ELIGIBLE)
        .or_else(|| metadata.get("tier_b_eligible"))
        .map(String::as_str)
        .unwrap_or("true");
    !matches!(
        raw.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

async fn resolve_execution_restore_snapshot_id(
    manager: &Manager,
    vm_id: &str,
    metadata: &HashMap<String, String>,
) -> Result<Option<String>> {
    if let Some(snapshot_id) = metadata
        .get(META_EXEC_RESTORE_SNAPSHOT_ID)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
    {
        return Ok(Some(snapshot_id));
    }
    let Some(snapshot_name) = metadata
        .get(META_EXEC_RESTORE_SNAPSHOT_NAME)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
    else {
        return Ok(None);
    };
    let snapshots = manager.list_snapshots(vm_id).await.map_err(|err| {
        anyhow!("list snapshots for execution restore marker lookup failed: {err}")
    })?;
    Ok(snapshots
        .into_iter()
        .find(|snapshot| snapshot.name == snapshot_name)
        .map(|snapshot| snapshot.id))
}

async fn refresh_execution_restore_snapshot_marker(
    manager: &Manager,
    vm_id: &str,
    vm_metadata: &HashMap<String, String>,
) -> Result<(String, String)> {
    if let Some(previous_snapshot_id) = vm_metadata
        .get(META_EXEC_RESTORE_SNAPSHOT_ID)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
    {
        let _ = manager
            .delete_snapshot(vm_id, previous_snapshot_id.as_str())
            .await;
    }

    let snapshot = manager
        .create_snapshot(
            vm_id,
            SnapshotParams {
                label: "execution-restore".to_string(),
                description: "control-bus exec stream restore marker".to_string(),
            },
        )
        .await
        .map_err(|err| anyhow!("create execution restore snapshot marker failed: {err}"))?;

    let mut updated_metadata = vm_metadata.clone();
    updated_metadata.insert(
        META_EXEC_RESTORE_SNAPSHOT_ID.to_string(),
        snapshot.id.clone(),
    );
    updated_metadata.insert(
        META_EXEC_RESTORE_SNAPSHOT_NAME.to_string(),
        snapshot.name.clone(),
    );
    manager
        .update_vm(
            vm_id,
            UpdateVmParams {
                name: None,
                metadata: Some(updated_metadata),
            },
        )
        .await
        .map_err(|err| anyhow!("persist execution restore snapshot marker failed: {err}"))?;

    Ok((snapshot.id, snapshot.name))
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

fn sanitize_subject_token(raw: &str) -> String {
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
        "command".to_string()
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

fn next_stream_event_id(cluster_id: &str) -> String {
    format!(
        "{}-{}",
        sanitize_key_component(cluster_id),
        Uuid::now_v7().as_simple()
    )
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
            target_node_id: None,
            payload: Value::Null,
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

    #[test]
    fn stream_event_id_is_cluster_scoped_and_unique() {
        let first = next_stream_event_id("cluster-alpha");
        let second = next_stream_event_id("cluster-alpha");
        assert!(first.starts_with("cluster-alpha-"));
        assert!(second.starts_with("cluster-alpha-"));
        assert_ne!(first, second);
    }

    #[test]
    fn exec_stream_event_serializes_identity_envelope_fields() {
        let event = ExecStreamEvent {
            cluster_id: "cluster-a".to_string(),
            logical_stream_id: "logical-stream-1".to_string(),
            stream_id: "logical-stream-1".to_string(),
            event_seq: 4,
            event_id: "cluster-a-event-123".to_string(),
            producer_epoch: 2,
            command_id: "cmd-1".to_string(),
            session_id: "session-1".to_string(),
            vm_id: "vm-1".to_string(),
            kind: "stdout".to_string(),
            data: b"ok".to_vec(),
            exit_code: None,
            timed_out: false,
            error: None,
            sequence: 4,
            emitted_by_node_id: "node-a".to_string(),
            emitted_at_unix_ms: 42,
        };

        let value = serde_json::to_value(event).expect("serialize event");
        assert_eq!(
            value.get("cluster_id").and_then(Value::as_str),
            Some("cluster-a")
        );
        assert_eq!(
            value.get("logical_stream_id").and_then(Value::as_str),
            Some("logical-stream-1")
        );
        assert_eq!(value.get("event_seq").and_then(Value::as_u64), Some(4));
        assert_eq!(
            value.get("event_id").and_then(Value::as_str),
            Some("cluster-a-event-123")
        );
        assert_eq!(value.get("producer_epoch").and_then(Value::as_u64), Some(2));
        assert_eq!(value.get("sequence").and_then(Value::as_u64), Some(4));
    }

    #[test]
    fn exec_stream_start_payload_supports_legacy_stream_id_field() {
        let raw = json!({
            "stream_id": "legacy-stream",
            "session_id": "session-1",
            "vm_id": "vm-1",
            "command": "echo ok"
        });
        let payload: ExecStreamStartPayload = serde_json::from_value(raw).expect("decode payload");
        assert_eq!(payload.stream_id, "legacy-stream");
        assert!(payload.logical_stream_id.is_empty());
        assert_eq!(payload.resume_after_event_seq, 0);
        assert_eq!(payload.producer_epoch, 0);
    }
}
