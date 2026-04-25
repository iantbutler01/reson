use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::SystemTime;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

const MAX_RECENT_EVENTS_PER_VM: usize = 200;
const MAX_ACCESS_LOG_READ_BYTES_PER_REFRESH: u64 = 1024 * 1024;
const MAX_VM_COUNTER_STATES: usize = 2048;
const CONNECTION_WINDOW_SECS: i64 = 60;
const BANDWIDTH_WINDOW_SECS: i64 = 60 * 60;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VmProxyAccessEvent {
    pub seq: u64,
    pub occurred_at: DateTime<Utc>,
    pub authority: String,
    pub method: String,
    pub path: String,
    pub decision: String,
    pub response_code: u16,
    pub bytes_received: u64,
    pub bytes_sent: u64,
    pub requested_server_name: Option<String>,
    pub upstream_host: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VmProxyActivitySnapshot {
    pub connection_attempts: u64,
    pub proxied_requests: u64,
    pub denied_requests: u64,
    pub allowed_bytes: u64,
    pub denied_bytes: u64,
    pub connection_attempts_last_minute: u64,
    pub allowed_bytes_last_hour: u64,
    pub connection_window_resets_at: Option<DateTime<Utc>>,
    pub bandwidth_window_resets_at: Option<DateTime<Utc>>,
    pub blocked_until: Option<DateTime<Utc>>,
    pub includes_kernel_counters: bool,
    pub updated_at: Option<DateTime<Utc>>,
    pub recent_events: Vec<VmProxyAccessEvent>,
}

#[derive(Clone, Debug)]
pub struct VmProxyLimitSnapshot {
    pub connection_attempts_last_minute: u64,
    pub allowed_bytes_last_hour: u64,
    pub connection_window_resets_at: Option<DateTime<Utc>>,
    pub bandwidth_window_resets_at: Option<DateTime<Utc>>,
}

#[derive(Default)]
pub struct VmCounters {
    access_log_path: PathBuf,
    read_offset: u64,
    last_modified_at: Option<SystemTime>,
    partial_line: String,
    next_seq: u64,
    by_vm: HashMap<String, VmCounterState>,
}

#[derive(Default)]
struct VmCounterState {
    connection_attempts: u64,
    proxied_requests: u64,
    denied_requests: u64,
    allowed_bytes: u64,
    denied_bytes: u64,
    updated_at: Option<DateTime<Utc>>,
    recent_events: VecDeque<VmProxyAccessEvent>,
    connection_events: VecDeque<DateTime<Utc>>,
    usage_events: VecDeque<VmProxyUsageEvent>,
}

#[derive(Clone, Debug)]
struct VmProxyUsageEvent {
    occurred_at: DateTime<Utc>,
    allowed_bytes: u64,
}

#[derive(Debug, Deserialize)]
struct RawEnvoyAccessLog {
    timestamp: Option<String>,
    vm_id: Option<String>,
    authority: Option<String>,
    method: Option<String>,
    path: Option<String>,
    response_code: Option<String>,
    response_code_details: Option<String>,
    bytes_received: Option<String>,
    bytes_sent: Option<String>,
    requested_server_name: Option<String>,
    upstream_host: Option<String>,
}

impl VmCounters {
    pub fn new(access_log_path: impl Into<PathBuf>) -> Self {
        Self {
            access_log_path: access_log_path.into(),
            ..Self::default()
        }
    }

    pub fn snapshot(
        &mut self,
        vm_id: &str,
        recent_limit: usize,
    ) -> Result<Option<VmProxyActivitySnapshot>> {
        self.refresh()?;
        Ok(self
            .by_vm
            .get(vm_id)
            .map(|state| state.snapshot(recent_limit)))
    }

    pub fn refresh(&mut self) -> Result<()> {
        self.sync_from_access_log()?;
        let now = Utc::now();
        for state in self.by_vm.values_mut() {
            state.prune_usage_windows(now);
        }
        self.prune_counter_states();
        Ok(())
    }

    pub fn limit_snapshot(
        &mut self,
        vm_id: &str,
        now: DateTime<Utc>,
    ) -> Result<Option<VmProxyLimitSnapshot>> {
        self.sync_from_access_log()?;
        for state in self.by_vm.values_mut() {
            state.prune_usage_windows(now);
        }
        self.prune_counter_states();
        Ok(self.by_vm.get(vm_id).map(|state| state.limit_snapshot(now)))
    }

    pub fn record_guardrail_event(
        &mut self,
        vm_id: &str,
        decision: &str,
        detail: &str,
        occurred_at: DateTime<Utc>,
    ) {
        self.next_seq = self.next_seq.saturating_add(1);
        let state = self.by_vm.entry(vm_id.to_string()).or_default();
        state.updated_at = Some(occurred_at);
        state.recent_events.push_back(VmProxyAccessEvent {
            seq: self.next_seq,
            occurred_at,
            authority: String::new(),
            method: "GUARDRAIL".to_string(),
            path: detail.to_string(),
            decision: decision.to_string(),
            response_code: 429,
            bytes_received: 0,
            bytes_sent: 0,
            requested_server_name: None,
            upstream_host: None,
        });
        while state.recent_events.len() > MAX_RECENT_EVENTS_PER_VM {
            state.recent_events.pop_front();
        }
    }

    fn sync_from_access_log(&mut self) -> Result<()> {
        if !self.access_log_path.exists() {
            return Ok(());
        }

        let mut file = File::open(&self.access_log_path)
            .with_context(|| format!("open envoy access log {}", self.access_log_path.display()))?;
        let metadata = file
            .metadata()
            .with_context(|| format!("stat envoy access log {}", self.access_log_path.display()))?;
        let file_len = metadata.len();
        let modified_at = metadata.modified().ok();

        if file_len < self.read_offset
            || (file_len <= self.read_offset && modified_at != self.last_modified_at)
        {
            self.read_offset = 0;
            self.partial_line.clear();
        }

        file.seek(SeekFrom::Start(self.read_offset))
            .with_context(|| format!("seek envoy access log {}", self.access_log_path.display()))?;
        let remaining = file_len.saturating_sub(self.read_offset);
        let read_limit = remaining.min(MAX_ACCESS_LOG_READ_BYTES_PER_REFRESH);
        let mut buf = Vec::with_capacity(read_limit.min(64 * 1024) as usize);
        file.take(read_limit)
            .read_to_end(&mut buf)
            .with_context(|| format!("read envoy access log {}", self.access_log_path.display()))?;
        self.read_offset = self
            .read_offset
            .saturating_add(buf.len() as u64)
            .min(file_len);
        if self.read_offset == file_len {
            self.last_modified_at = modified_at;
        }

        if buf.is_empty() {
            return Ok(());
        }

        let mut combined = std::mem::take(&mut self.partial_line);
        combined.push_str(&String::from_utf8_lossy(&buf));
        let ends_with_newline = combined.ends_with('\n');
        let mut lines = combined.lines().map(str::to_owned).collect::<Vec<_>>();
        if !ends_with_newline {
            self.partial_line = lines.pop().unwrap_or_default();
        }

        for line in lines {
            self.ingest_line(&line);
        }

        Ok(())
    }

    fn prune_counter_states(&mut self) {
        if self.by_vm.len() <= MAX_VM_COUNTER_STATES {
            return;
        }
        let mut states = self
            .by_vm
            .iter()
            .map(|(vm_id, state)| (vm_id.clone(), state.updated_at))
            .collect::<Vec<_>>();
        states.sort_by_key(|(_, updated_at)| *updated_at);
        let remove_count = self.by_vm.len().saturating_sub(MAX_VM_COUNTER_STATES);
        for (vm_id, _) in states.into_iter().take(remove_count) {
            self.by_vm.remove(&vm_id);
        }
    }

    fn ingest_line(&mut self, line: &str) {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            return;
        }

        let Ok(raw) = serde_json::from_str::<RawEnvoyAccessLog>(trimmed) else {
            return;
        };

        let Some(vm_id) = raw.vm_id.map(|value| value.trim().to_string()) else {
            return;
        };
        if vm_id.is_empty() {
            return;
        }

        let occurred_at = raw
            .timestamp
            .as_deref()
            .and_then(parse_timestamp)
            .unwrap_or_else(Utc::now);
        let response_code = raw
            .response_code
            .as_deref()
            .and_then(|value| value.parse::<u16>().ok())
            .unwrap_or_default();
        let bytes_received = raw
            .bytes_received
            .as_deref()
            .and_then(parse_counter)
            .unwrap_or_default();
        let bytes_sent = raw
            .bytes_sent
            .as_deref()
            .and_then(parse_counter)
            .unwrap_or_default();
        let denied = response_code == 403
            || raw
                .response_code_details
                .as_deref()
                .is_some_and(|value| value.contains("direct_response"));

        self.next_seq = self.next_seq.saturating_add(1);
        let event = VmProxyAccessEvent {
            seq: self.next_seq,
            occurred_at,
            authority: raw.authority.unwrap_or_default(),
            method: raw.method.unwrap_or_default(),
            path: raw.path.unwrap_or_default(),
            decision: if denied {
                "denied".to_string()
            } else {
                "allowed".to_string()
            },
            response_code,
            bytes_received,
            bytes_sent,
            requested_server_name: raw
                .requested_server_name
                .filter(|value| !value.trim().is_empty()),
            upstream_host: raw.upstream_host.filter(|value| !value.trim().is_empty()),
        };

        let total_bytes = bytes_received.saturating_add(bytes_sent);
        let state = self.by_vm.entry(vm_id).or_default();
        state.connection_attempts = state.connection_attempts.saturating_add(1);
        state.connection_events.push_back(occurred_at);
        if denied {
            state.denied_requests = state.denied_requests.saturating_add(1);
            state.denied_bytes = state.denied_bytes.saturating_add(total_bytes);
        } else {
            state.proxied_requests = state.proxied_requests.saturating_add(1);
            state.allowed_bytes = state.allowed_bytes.saturating_add(total_bytes);
            state.usage_events.push_back(VmProxyUsageEvent {
                occurred_at,
                allowed_bytes: total_bytes,
            });
        }
        state.updated_at = Some(occurred_at);
        state.recent_events.push_back(event);
        while state.recent_events.len() > MAX_RECENT_EVENTS_PER_VM {
            state.recent_events.pop_front();
        }
    }
}

impl VmCounterState {
    fn snapshot(&self, recent_limit: usize) -> VmProxyActivitySnapshot {
        let recent_events = self
            .recent_events
            .iter()
            .rev()
            .take(recent_limit.max(1))
            .cloned()
            .collect::<Vec<_>>();
        VmProxyActivitySnapshot {
            connection_attempts: self.connection_attempts,
            proxied_requests: self.proxied_requests,
            denied_requests: self.denied_requests,
            allowed_bytes: self.allowed_bytes,
            denied_bytes: self.denied_bytes,
            connection_attempts_last_minute: 0,
            allowed_bytes_last_hour: 0,
            connection_window_resets_at: None,
            bandwidth_window_resets_at: None,
            blocked_until: None,
            includes_kernel_counters: false,
            updated_at: self.updated_at,
            recent_events,
        }
    }

    fn limit_snapshot(&self, now: DateTime<Utc>) -> VmProxyLimitSnapshot {
        let connection_cutoff = now - chrono::Duration::seconds(CONNECTION_WINDOW_SECS);
        let bandwidth_cutoff = now - chrono::Duration::seconds(BANDWIDTH_WINDOW_SECS);
        let connection_attempts_last_minute = self
            .connection_events
            .iter()
            .filter(|occurred_at| **occurred_at >= connection_cutoff)
            .count() as u64;
        let mut allowed_bytes_last_hour = 0_u64;
        let connection_window_oldest = self
            .connection_events
            .iter()
            .find(|occurred_at| **occurred_at >= connection_cutoff)
            .copied();
        let mut bandwidth_window_oldest = None;

        for event in &self.usage_events {
            if event.occurred_at >= bandwidth_cutoff {
                allowed_bytes_last_hour =
                    allowed_bytes_last_hour.saturating_add(event.allowed_bytes);
                if bandwidth_window_oldest.is_none() {
                    bandwidth_window_oldest = Some(event.occurred_at);
                }
            }
        }

        VmProxyLimitSnapshot {
            connection_attempts_last_minute,
            allowed_bytes_last_hour,
            connection_window_resets_at: connection_window_oldest
                .map(|ts| ts + chrono::Duration::seconds(CONNECTION_WINDOW_SECS)),
            bandwidth_window_resets_at: bandwidth_window_oldest
                .map(|ts| ts + chrono::Duration::seconds(BANDWIDTH_WINDOW_SECS)),
        }
    }

    fn prune_usage_windows(&mut self, now: DateTime<Utc>) {
        let connection_cutoff = now - chrono::Duration::seconds(CONNECTION_WINDOW_SECS);
        while self
            .connection_events
            .front()
            .is_some_and(|occurred_at| *occurred_at < connection_cutoff)
        {
            self.connection_events.pop_front();
        }

        let bandwidth_cutoff = now - chrono::Duration::seconds(BANDWIDTH_WINDOW_SECS);
        while self
            .usage_events
            .front()
            .is_some_and(|event| event.occurred_at < bandwidth_cutoff)
        {
            self.usage_events.pop_front();
        }
    }
}

fn parse_counter(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}

fn parse_timestamp(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|value| value.with_timezone(&Utc))
        .ok()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;

    #[test]
    fn vm_counters_tracks_allowed_and_denied_events() {
        let tempdir = TempDir::new().expect("tempdir");
        let log_path = tempdir.path().join("access.log");
        std::fs::write(
            &log_path,
            concat!(
                "{\"timestamp\":\"2026-04-13T10:00:00.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"api.github.com:443\",\"method\":\"CONNECT\",\"path\":\"\",\"response_code\":\"200\",\"response_code_details\":\"via_upstream\",\"bytes_received\":\"10\",\"bytes_sent\":\"20\",\"requested_server_name\":\"api.github.com\",\"upstream_host\":\"140.82.112.5:443\"}\n",
                "{\"timestamp\":\"2026-04-13T10:00:01.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"smtp.example.com:25\",\"method\":\"CONNECT\",\"path\":\"\",\"response_code\":\"403\",\"response_code_details\":\"direct_response\",\"bytes_received\":\"2\",\"bytes_sent\":\"3\"}\n"
            ),
        )
        .expect("write access log");

        let mut counters = VmCounters::new(&log_path);
        let snapshot = counters
            .snapshot("vm-1", 10)
            .expect("snapshot")
            .expect("vm snapshot");

        assert_eq!(snapshot.connection_attempts, 2);
        assert_eq!(snapshot.proxied_requests, 1);
        assert_eq!(snapshot.denied_requests, 1);
        assert_eq!(snapshot.allowed_bytes, 30);
        assert_eq!(snapshot.denied_bytes, 5);
        assert_eq!(snapshot.recent_events.len(), 2);
        assert_eq!(snapshot.recent_events[0].decision, "denied");
        assert_eq!(snapshot.recent_events[1].decision, "allowed");
    }

    #[test]
    fn vm_counters_handles_truncated_log() {
        let tempdir = TempDir::new().expect("tempdir");
        let log_path = tempdir.path().join("access.log");
        std::fs::write(
            &log_path,
            "{\"timestamp\":\"2026-04-13T10:00:00.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"api.github.com:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"1\",\"bytes_sent\":\"2\"}\n",
        )
        .expect("write access log");
        let mut counters = VmCounters::new(&log_path);
        let _ = counters.snapshot("vm-1", 10).expect("first snapshot");

        std::fs::write(
            &log_path,
            "{\"timestamp\":\"2026-04-13T10:00:01.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"api.openai.com:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"3\",\"bytes_sent\":\"4\"}\n",
        )
        .expect("rewrite access log");

        let snapshot = counters
            .snapshot("vm-1", 10)
            .expect("second snapshot")
            .expect("vm snapshot");
        assert_eq!(snapshot.connection_attempts, 2);
        assert_eq!(snapshot.recent_events[0].authority, "api.openai.com:443");
    }

    #[test]
    fn vm_counters_ignores_global_listener_entries_without_vm_id() {
        let tempdir = TempDir::new().expect("tempdir");
        let log_path = tempdir.path().join("access.log");
        std::fs::write(
            &log_path,
            "{\"timestamp\":\"2026-04-13T10:00:00.000Z\",\"vm_id\":\"\",\"authority\":\"example.com:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"1\",\"bytes_sent\":\"2\"}\n",
        )
        .expect("write access log");

        let mut counters = VmCounters::new(&log_path);
        assert!(counters.snapshot("vm-1", 10).expect("snapshot").is_none());
    }

    #[test]
    fn vm_counters_computes_limit_windows() {
        let tempdir = TempDir::new().expect("tempdir");
        let log_path = tempdir.path().join("access.log");
        std::fs::write(
            &log_path,
            concat!(
                "{\"timestamp\":\"2026-04-13T10:58:59.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"old.example:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"5\",\"bytes_sent\":\"5\"}\n",
                "{\"timestamp\":\"2026-04-13T10:59:30.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"api.github.com:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"10\",\"bytes_sent\":\"20\"}\n",
                "{\"timestamp\":\"2026-04-13T10:59:50.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"api.openai.com:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"1\",\"bytes_sent\":\"2\"}\n"
            ),
        )
        .expect("write access log");

        let mut counters = VmCounters::new(&log_path);
        let snapshot = counters
            .limit_snapshot(
                "vm-1",
                chrono::DateTime::parse_from_rfc3339("2026-04-13T11:00:00.000Z")
                    .expect("timestamp")
                    .with_timezone(&Utc),
            )
            .expect("limit snapshot")
            .expect("vm snapshot");

        assert_eq!(snapshot.connection_attempts_last_minute, 2);
        assert_eq!(snapshot.allowed_bytes_last_hour, 43);
        assert_eq!(
            snapshot
                .connection_window_resets_at
                .expect("connection reset")
                .to_rfc3339(),
            "2026-04-13T11:00:30+00:00"
        );
        assert_eq!(
            snapshot
                .bandwidth_window_resets_at
                .expect("bandwidth reset")
                .to_rfc3339(),
            "2026-04-13T11:58:59+00:00"
        );
    }

    #[test]
    fn vm_counters_connection_window_counts_denied_attempts() {
        let tempdir = TempDir::new().expect("tempdir");
        let log_path = tempdir.path().join("access.log");
        std::fs::write(
            &log_path,
            concat!(
                "{\"timestamp\":\"2026-04-13T10:59:30.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"api.github.com:443\",\"method\":\"CONNECT\",\"response_code\":\"200\",\"bytes_received\":\"10\",\"bytes_sent\":\"20\"}\n",
                "{\"timestamp\":\"2026-04-13T10:59:50.000Z\",\"vm_id\":\"vm-1\",\"authority\":\"smtp.example.com:25\",\"method\":\"CONNECT\",\"response_code\":\"403\",\"response_code_details\":\"direct_response\",\"bytes_received\":\"1\",\"bytes_sent\":\"2\"}\n"
            ),
        )
        .expect("write access log");

        let mut counters = VmCounters::new(&log_path);
        let snapshot = counters
            .limit_snapshot(
                "vm-1",
                chrono::DateTime::parse_from_rfc3339("2026-04-13T11:00:00.000Z")
                    .expect("timestamp")
                    .with_timezone(&Utc),
            )
            .expect("limit snapshot")
            .expect("vm snapshot");

        assert_eq!(snapshot.connection_attempts_last_minute, 2);
        assert_eq!(snapshot.allowed_bytes_last_hour, 30);
        assert_eq!(
            snapshot
                .connection_window_resets_at
                .expect("connection reset")
                .to_rfc3339(),
            "2026-04-13T11:00:30+00:00"
        );
    }
}
