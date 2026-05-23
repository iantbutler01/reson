#!/usr/bin/env bash
# @dive-file: Verifies Tier-B cross-node execution-state restore contract and continuity restore evidence.
# @dive-rel: Enforced by scripts/verify_reson_sandbox.sh as a strict gate for checklist item 12.32.
# @dive-rel: Executes facade contract tests in crates/reson-sandbox/tests/facade_contract.rs.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run tier-b execution-state restore gate"
    exit 1
  fi
  warn "runtime sources missing; skipping tier-b execution-state restore gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "tier-b execution-state restore gate: static contract checks"
[[ -f "$REPO_ROOT/specs/runbooks/TIERB_EXECUTION_STATE_RESTORE.md" ]] || {
  err "missing Tier-B execution-state restore runbook"
  exit 1
}
rg -n "META_EXEC_RESTORE_SNAPSHOT_ID|META_EXEC_RESTORE_SNAPSHOT_NAME|execution_restore_snapshot_id" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "restore_execution_state_if_needed\(|restore_snapshot\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "continuity_rebinds_session_after_primary_vmd_loss|restore_calls|restored_snapshot_ids" "$REPO_ROOT/crates/reson-sandbox/tests/facade_contract.rs" >/dev/null

log "tier-b execution-state restore gate: continuity restore test"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract continuity_rebinds_session_after_primary_vmd_loss -- --nocapture)

log "tier-b execution-state restore gate: passed"
