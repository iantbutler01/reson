#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_reconcile contract coverage.
# @dive-rel: Invoked by scripts/verify_reson_sandbox.sh gate orchestration and/or Makefile targets.
# @dive-rel: Uses scripts/common.sh helpers for strict-mode behavior, diagnostics, and command validation.

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
rg -n "fn plan_reconcile\\(|load_partitioned_session_routes|session_shard_count|reconcile_checkpoint_key|is_sharded_session_key" "$REPO_ROOT/vmd/src/reconcile.rs" >/dev/null

log "reconcile gate: convergence unit tests"
(cd "$REPO_ROOT" && cargo test -p vmd reconcile_plan_upserts_missing_and_deletes_stale --locked)
(cd "$REPO_ROOT" && cargo test -p vmd sharded_session_key_detection_is_precise --locked)

log "reconcile gate: passed"
