#!/usr/bin/env bash
# @dive-file: Real gate 42 wrapper for Tier-B restore-marker rehydration and resumed command flow.
# @dive-rel: Delegates to scripts/integration/verify_real_failover.sh selector tierb_restore_marker_rehydrates_and_resumes_under_failover_rebind.
# @dive-rel: Used by scripts/verify_strict_real.sh for real Tier-B execution-state recovery evidence.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "--selector" ]]; then
    err "verify_real_gate42.sh does not accept --selector (selector is fixed for gate 42)"
    exit 2
  fi
done

log "real gate 42: tier-b restore-marker rehydrate + resume"
exec "$REPO_ROOT/scripts/integration/verify_real_failover.sh" "$@" \
  --selector tierb_restore_marker_rehydrates_and_resumes_under_failover_rebind
