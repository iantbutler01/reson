#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_planned_drain_handoff contract coverage.
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
    err "runtime sources missing; strict mode cannot run planned drain handoff gate"
    exit 1
  fi
  warn "runtime sources missing; skipping planned drain handoff gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "planned-drain gate: static contract checks"
[[ -f "$REPO_ROOT/specs/runbooks/PLANNED_DRAIN_HANDOFF_AND_FENCING.md" ]] || {
  err "missing planned drain runbook"
  exit 1
}
rg -n "admission_frozen" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" "$REPO_ROOT/vmd/src/config.rs" "$REPO_ROOT/vmd/src/registry.rs" >/dev/null
rg -n "ownership_fence|expected_fence" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "continuity_rebinds_session_after_primary_vmd_loss" "$REPO_ROOT/crates/reson-sandbox/tests/facade_contract.rs" >/dev/null

log "planned-drain gate: frozen-node admission exclusion"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --features distributed-control distributed::tests::eligible_routes_skips_admission_frozen_nodes -- --nocapture)

log "planned-drain gate: handoff continuity attach/rebind"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract continuity_rebinds_session_after_primary_vmd_loss -- --nocapture)

log "planned-drain gate: in-flight command fencing"
(cd "$REPO_ROOT" && cargo test -p vmd ownership_fence_transition_rejects_stale_expectation -- --nocapture)

log "planned-drain gate: passed"
