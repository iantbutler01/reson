#!/usr/bin/env bash
# @dive-file: Verifies architecture-aware warm pool prewarm pipeline and refill hooks.
# @dive-rel: Wired into scripts/verify_reson_sandbox.sh as gate coverage for checklist item 12.35.
# @dive-rel: Executes warm_pool_prewarms_profiles_by_architecture in crates/reson-sandbox/tests/facade_contract.rs.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run warm-pool pipeline gate"
    exit 1
  fi
  warn "runtime sources missing; skipping warm-pool pipeline gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "warm-pool pipeline gate: static contract checks"
[[ -f "$REPO_ROOT/specs/runbooks/WARM_POOL_AND_PREWARM_PIPELINE.md" ]] || {
  err "missing warm-pool and prewarm pipeline runbook"
  exit 1
}
rg -n "WarmPoolProfile|prewarm_warm_pool_profiles|prewarm_profiles_on_endpoint|warm_pool_key|session\.create\.warm_pool|session\.create\.cold_cache_hit" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "warm_pool_prewarms_profiles_by_architecture|predownload_requests" "$REPO_ROOT/crates/reson-sandbox/tests/facade_contract.rs" >/dev/null

log "warm-pool pipeline gate: architecture prewarm contract test"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract warm_pool_prewarms_profiles_by_architecture -- --nocapture)

log "warm-pool pipeline gate: passed"
