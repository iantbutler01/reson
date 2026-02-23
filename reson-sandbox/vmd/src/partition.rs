// @dive-file: Network-partition policy monitor and fail-closed gating for mutating control paths.
// @dive-rel: Integrated by daemon orchestration in vmd/src/app.rs and checked by verifier gates.
// @dive-rel: Enforces bounded grace for local streams while blocking new mutations during quorum loss.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use etcd_client::{Client as EtcdClient, GetOptions};
use tokio::sync::{RwLock, oneshot};
use tokio::task::JoinHandle;
use tokio::time::MissedTickBehavior;
use tracing::{debug, info, warn};

const DEFAULT_PROBE_INTERVAL: Duration = Duration::from_secs(2);
const DEFAULT_FAILURE_THRESHOLD: u32 = 3;
const DEFAULT_LOCAL_STREAM_GRACE: Duration = Duration::from_secs(30);
const DEFAULT_COMMAND_RETRY_DELAY: Duration = Duration::from_secs(2);

#[derive(Clone, Debug)]
pub struct PartitionPolicyConfig {
    pub etcd_endpoints: Vec<String>,
    pub key_prefix: String,
    pub probe_interval: Duration,
    pub failure_threshold: u32,
    pub local_stream_grace: Duration,
    pub command_retry_delay: Duration,
}

impl PartitionPolicyConfig {
    pub fn normalize(mut self) -> Self {
        if self.probe_interval.is_zero() {
            self.probe_interval = DEFAULT_PROBE_INTERVAL;
        }
        if self.failure_threshold == 0 {
            self.failure_threshold = DEFAULT_FAILURE_THRESHOLD;
        }
        if self.local_stream_grace.is_zero() {
            self.local_stream_grace = DEFAULT_LOCAL_STREAM_GRACE;
        }
        if self.command_retry_delay.is_zero() {
            self.command_retry_delay = DEFAULT_COMMAND_RETRY_DELAY;
        }
        self
    }
}

#[derive(Clone)]
pub struct PartitionGate {
    state: std::sync::Arc<RwLock<PartitionState>>,
    failure_threshold: u32,
    local_stream_grace: Duration,
    command_retry_delay: Duration,
}

#[derive(Clone, Debug, Default)]
struct PartitionState {
    consecutive_failures: u32,
    partitioned_since: Option<Instant>,
}

pub struct PartitionMonitorHandle {
    gate: PartitionGate,
    stop_tx: Option<oneshot::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl PartitionMonitorHandle {
    pub fn gate(&self) -> PartitionGate {
        self.gate.clone()
    }

    pub async fn shutdown(mut self) {
        if let Some(tx) = self.stop_tx.take() {
            let _ = tx.send(());
        }
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

impl PartitionGate {
    fn new(
        failure_threshold: u32,
        local_stream_grace: Duration,
        command_retry_delay: Duration,
    ) -> Self {
        Self {
            state: std::sync::Arc::new(RwLock::new(PartitionState::default())),
            failure_threshold,
            local_stream_grace,
            command_retry_delay,
        }
    }

    pub async fn mutation_allowed(&self) -> bool {
        self.state.read().await.partitioned_since.is_none()
    }

    pub async fn mutation_rejection_reason(&self) -> Option<String> {
        let state = self.state.read().await;
        let partitioned_since = state.partitioned_since?;
        let elapsed_ms = partitioned_since.elapsed().as_millis();
        Some(format!(
            "network partition fail-closed: control-plane quorum visibility lost {}ms ago; rejecting mutating commands",
            elapsed_ms
        ))
    }

    pub fn command_retry_delay(&self) -> Duration {
        self.command_retry_delay
    }

    pub async fn local_stream_allowed(&self, stream_started_at: Instant) -> bool {
        let state = self.state.read().await;
        local_stream_allowed_with_now(
            state.partitioned_since,
            stream_started_at,
            self.local_stream_grace,
            Instant::now(),
        )
    }

    async fn record_probe_success(&self) {
        let mut state = self.state.write().await;
        if state.partitioned_since.is_some() {
            info!("control-plane quorum visibility restored");
        }
        state.consecutive_failures = 0;
        state.partitioned_since = None;
    }

    async fn record_probe_failure(&self) {
        let mut state = self.state.write().await;
        state.consecutive_failures = state.consecutive_failures.saturating_add(1);
        // @dive: Fail-closed flips only once at threshold crossing so logs and state transitions stay stable during prolonged outages.
        if state.consecutive_failures >= self.failure_threshold && state.partitioned_since.is_none()
        {
            state.partitioned_since = Some(Instant::now());
            warn!(
                consecutive_failures = state.consecutive_failures,
                failure_threshold = self.failure_threshold,
                local_stream_grace_secs = self.local_stream_grace.as_secs(),
                "network partition detected; mutating commands are now fail-closed"
            );
        }
    }

    #[cfg(test)]
    async fn set_partitioned_since_for_test(&self, value: Option<Instant>) {
        let mut state = self.state.write().await;
        state.partitioned_since = value;
        if value.is_none() {
            state.consecutive_failures = 0;
        }
    }
}

pub async fn start(
    config: Option<PartitionPolicyConfig>,
) -> Result<Option<PartitionMonitorHandle>> {
    let Some(config) = config.map(PartitionPolicyConfig::normalize) else {
        return Ok(None);
    };
    if config.etcd_endpoints.is_empty() {
        return Ok(None);
    }

    let gate = PartitionGate::new(
        config.failure_threshold,
        config.local_stream_grace,
        config.command_retry_delay,
    );
    let loop_gate = gate.clone();
    let loop_config = config.clone();
    let (stop_tx, mut stop_rx) = oneshot::channel::<()>();
    let join = tokio::spawn(async move {
        let mut interval = tokio::time::interval(loop_config.probe_interval);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut client: Option<EtcdClient> = None;

        loop {
            tokio::select! {
                _ = &mut stop_rx => break,
                _ = interval.tick() => {
                    match probe_once(&loop_config, &mut client).await {
                        Ok(()) => loop_gate.record_probe_success().await,
                        Err(err) => {
                            // @dive: Probe failures are intentionally treated as control-plane visibility loss, not fatal task exits.
                            debug!(err = %err, "control-plane quorum probe failed");
                            loop_gate.record_probe_failure().await;
                        }
                    }
                }
            }
        }
    });

    Ok(Some(PartitionMonitorHandle {
        gate,
        stop_tx: Some(stop_tx),
        join: Some(join),
    }))
}

async fn probe_once(config: &PartitionPolicyConfig, client: &mut Option<EtcdClient>) -> Result<()> {
    if client.is_none() {
        let connected = EtcdClient::connect(config.etcd_endpoints.clone(), None)
            .await
            .context("connect etcd for partition probe")?;
        *client = Some(connected);
    }

    let key_prefix = format!("{}/nodes/", config.key_prefix.trim_end_matches('/'));
    let probe = client
        .as_mut()
        .expect("partition probe client must be initialized")
        .get(
            key_prefix,
            Some(GetOptions::new().with_prefix().with_limit(1)),
        )
        .await;
    if let Err(err) = probe {
        // @dive: On RPC failure we drop the client so next tick forces a clean reconnect instead of reusing stale transport.
        *client = None;
        return Err(err).context("etcd partition probe get");
    }
    Ok(())
}

fn local_stream_allowed_with_now(
    partitioned_since: Option<Instant>,
    stream_started_at: Instant,
    local_stream_grace: Duration,
    now: Instant,
) -> bool {
    let Some(partitioned_since) = partitioned_since else {
        return true;
    };
    // @dive: Streams that began after partition onset are blocked; pre-existing streams get bounded grace-only continuation.
    if stream_started_at > partitioned_since {
        return false;
    }
    now.duration_since(partitioned_since) <= local_stream_grace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn partition_gate_blocks_mutations_after_threshold() {
        let gate = PartitionGate::new(3, Duration::from_secs(30), Duration::from_secs(2));
        assert!(gate.mutation_allowed().await);

        gate.record_probe_failure().await;
        assert!(gate.mutation_allowed().await);

        gate.record_probe_failure().await;
        assert!(gate.mutation_allowed().await);

        gate.record_probe_failure().await;
        assert!(!gate.mutation_allowed().await);
        assert!(gate.mutation_rejection_reason().await.is_some());
    }

    #[tokio::test]
    async fn partition_gate_allows_only_preexisting_streams_within_grace() {
        let gate = PartitionGate::new(1, Duration::from_secs(10), Duration::from_secs(2));
        let now = Instant::now();
        gate.set_partitioned_since_for_test(Some(now - Duration::from_secs(3)))
            .await;

        assert!(
            gate.local_stream_allowed(now - Duration::from_secs(4))
                .await
        );
        assert!(
            !gate
                .local_stream_allowed(now - Duration::from_secs(1))
                .await
        );

        gate.set_partitioned_since_for_test(Some(now - Duration::from_secs(20)))
            .await;
        assert!(
            !gate
                .local_stream_allowed(now - Duration::from_secs(30))
                .await
        );
    }
}
