#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run reconcile gate"
    exit 1
  fi
  warn "runtime sources missing; skipping reconcile gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "reconcile gate: static wiring checks"
rg -n "pub mod reconcile;" "$REPO_ROOT/vmd/src/lib.rs" >/dev/null
rg -n "start_reconcile_worker\\(" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "reconcile::start\\(" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "reconcile\\.run" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "fn plan_reconcile\\(" "$REPO_ROOT/vmd/src/reconcile.rs" >/dev/null

log "reconcile gate: convergence unit tests"
(cd "$REPO_ROOT" && cargo test -p vmd reconcile_plan_upserts_missing_and_deletes_stale --locked)

log "reconcile gate: passed"
