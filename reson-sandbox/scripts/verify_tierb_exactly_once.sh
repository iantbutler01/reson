#!/usr/bin/env bash
# @dive-file: Verifies Tier-B mid-command failover rebinding and exactly-once command dispatch semantics.
# @dive-rel: Wired into scripts/verify_reson_sandbox.sh as checklist gate coverage for item 12.33.
# @dive-rel: Executes facade contract test inflight_exec_rebinds_on_rpc_loss_and_runs_exactly_once.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run tier-b exactly-once gate"
    exit 1
  fi
  warn "runtime sources missing; skipping tier-b exactly-once gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "tier-b exactly-once gate: static contract checks"
[[ -f "$REPO_ROOT/specs/runbooks/TIERB_MID_COMMAND_FAILOVER_EXACTLY_ONCE.md" ]] || {
  err "missing Tier-B mid-command failover exactly-once runbook"
  exit 1
}
rg -n "rebind_session_endpoint\(|is_rebind_candidate_error\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "inflight_exec_rebinds_on_rpc_loss_and_runs_exactly_once|command_invocations" "$REPO_ROOT/crates/reson-sandbox/tests/facade_contract.rs" >/dev/null

log "tier-b exactly-once gate: failover dispatch contract test"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract inflight_exec_rebinds_on_rpc_loss_and_runs_exactly_once -- --nocapture)

log "tier-b exactly-once gate: passed"
