#!/usr/bin/env bash
# @dive-file: Verifies tier_b_eligible execution-fidelity classification and restore-marker enforcement semantics.
# @dive-rel: Wired into scripts/verify_reson_sandbox.sh for checklist item 12.34 gate coverage.
# @dive-rel: Executes fidelity-policy contract tests in crates/reson-sandbox/tests/facade_contract.rs.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run execution-fidelity policy gate"
    exit 1
  fi
  warn "runtime sources missing; skipping execution-fidelity policy gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "execution-fidelity policy gate: static contract checks"
[[ -f "$REPO_ROOT/specs/runbooks/TIERB_EXECUTION_FIDELITY_POLICY.md" ]] || {
  err "missing Tier-B execution fidelity policy runbook"
  exit 1
}
rg -n "META_TIER_B_ELIGIBLE|META_EXECUTION_FIDELITY_REQUIREMENT|resolve_tier_b_eligibility|vm_tier_b_eligible" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "tier_b_eligibility_classifier_sets_metadata_policy|tier_b_eligible_failover_requires_restore_snapshot_marker" "$REPO_ROOT/crates/reson-sandbox/tests/facade_contract.rs" >/dev/null

log "execution-fidelity policy gate: contract tests"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract tier_b_eligibility_classifier_sets_metadata_policy -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract tier_b_eligible_failover_requires_restore_snapshot_marker -- --nocapture)

log "execution-fidelity policy gate: passed"
