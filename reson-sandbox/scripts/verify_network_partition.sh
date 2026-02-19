#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run network-partition gate"
    exit 1
  fi
  warn "runtime sources missing; skipping network-partition gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "network-partition gate: static fail-closed policy checks"
rg -n "start_partition_monitor|derive_partition_policy_config|network partition fail-closed|Status::unavailable" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "PartitionGate|ack_with\\(AckKind::Nak\\(Some\\(gate.command_retry_delay\\(\\)\\)\\)\\)|rejecting control command while quorum visibility is lost" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "PartitionPolicyConfig|mutation_allowed|local_stream_allowed|partitioned_since|probe_once" "$REPO_ROOT/vmd/src/partition.rs" >/dev/null

log "network-partition gate: fail-closed + bounded-grace unit coverage"
(cd "$REPO_ROOT" && cargo test -p vmd partition_gate_blocks_mutations_after_threshold -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p vmd partition_gate_allows_only_preexisting_streams_within_grace -- --nocapture)

log "network-partition gate: passed"
