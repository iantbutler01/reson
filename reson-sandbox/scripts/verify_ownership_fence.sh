#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_ownership_fence contract coverage.
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
    err "runtime sources missing; strict mode cannot run ownership-fence gate"
    exit 1
  fi
  warn "runtime sources missing; skipping ownership-fence gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "ownership-fence gate: static fencing contract checks"
rg -n "FenceConflict|current_session_fence|ownership_fence|bind_session_route\\(.*expected_fence" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "put_session_route\\(.*expected_fence|delete_session_route\\(.*expected_fence|session_fence_key|ownership_fence_allows_transition|Compare::value|Compare::version|Txn::new" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null
rg -n "expected_fence|EtcdOwnershipFenceStore|check_and_rotate|ownership_fence_conflict" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null

log "ownership-fence gate: race-condition coverage tests"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox ownership_fence_transition_rejects_stale_expectation -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p vmd ownership_fence_transition_rejects_stale_expectation -- --nocapture)

log "ownership-fence gate: passed"
