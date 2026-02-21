#!/usr/bin/env bash
# @dive-file: Real gate 46 wrapper for distributed failover continuity suite on live control-plane machinery.
# @dive-rel: Delegates to scripts/integration/verify_real_failover.sh in --distributed mode.
# @dive-rel: Captures full L3 distributed continuity surface including stream recovery and MQ failover variants.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "--distributed" ]]; then
    err "verify_real_gate46.sh sets --distributed internally; remove explicit --distributed"
    exit 2
  fi
done

log "real gate 46: distributed failover continuity suite (L3)"
exec "$REPO_ROOT/scripts/integration/verify_real_failover.sh" "$@" --distributed
