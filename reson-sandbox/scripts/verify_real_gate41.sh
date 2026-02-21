#!/usr/bin/env bash
# @dive-file: Real gate 41 wrapper for L2 unplanned node-loss continuity under active command stream.
# @dive-rel: Delegates to scripts/integration/verify_real_failover.sh selector primary_node_loss_during_active_exec_stream_rebinds_session.
# @dive-rel: Used by scripts/verify_strict_real.sh to enforce real continuity evidence.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "--selector" ]]; then
    err "verify_real_gate41.sh does not accept --selector (selector is fixed for gate 41)"
    exit 2
  fi
done

log "real gate 41: active-stream continuity on unplanned primary loss (L2)"
exec "$REPO_ROOT/scripts/integration/verify_real_failover.sh" "$@" \
  --selector primary_node_loss_during_active_exec_stream_rebinds_session
