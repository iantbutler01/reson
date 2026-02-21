#!/usr/bin/env bash
# @dive-file: Real gate 43 wrapper for exactly-once in-flight command acknowledgement under failover.
# @dive-rel: Delegates to scripts/integration/verify_real_failover.sh selector inflight_exec_is_acknowledged_exactly_once_under_primary_loss.
# @dive-rel: Used by scripts/verify_strict_real.sh as real dedupe/no-duplicate-side-effect evidence.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "--selector" ]]; then
    err "verify_real_gate43.sh does not accept --selector (selector is fixed for gate 43)"
    exit 2
  fi
done

log "real gate 43: in-flight exactly-once under failover"
exec "$REPO_ROOT/scripts/integration/verify_real_failover.sh" "$@" \
  --selector inflight_exec_is_acknowledged_exactly_once_under_primary_loss
