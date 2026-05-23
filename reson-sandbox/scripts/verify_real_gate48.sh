#!/usr/bin/env bash
# @dive-file: Real gate 48 wrapper for distributed stream identity/checkpoint/no-replay semantics.
# @dive-rel: Executes focused selectors from scripts/integration/verify_real_failover.sh --distributed.
# @dive-rel: Covers event envelope identity fields, event_seq checkpoint resume, terminal no-rerun, and producer-epoch continuity.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

for arg in "$@"; do
  case "$arg" in
    --selector|--distributed)
      err "verify_real_gate48.sh does not accept --selector/--distributed (both are controlled by gate 48)"
      exit 2
      ;;
  esac
done

selectors=(
  "distributed_stream_resume_from_checkpoint_is_forward_only_without_replay"
  "distributed_stream_events_include_identity_envelope_and_monotonic_sequence"
  "distributed_terminal_stream_is_not_rerun_after_primary_failover"
  "distributed_failover_rebind_updates_route_and_emits_events"
)

log "real gate 48: distributed stream identity/checkpoint semantics"
for selector in "${selectors[@]}"; do
  log "real gate 48: running selector ${selector}"
  "$REPO_ROOT/scripts/integration/verify_real_failover.sh" "$@" --distributed --selector "$selector"
done
