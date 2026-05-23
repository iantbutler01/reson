#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_tierb_continuity contract coverage.
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
    err "runtime sources missing; strict mode cannot run tier-b continuity gate"
    exit 1
  fi
  warn "runtime sources missing; skipping tier-b continuity gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "tier-b continuity gate: static continuity orchestration checks"
rg -n "resolve_session_endpoint\(|ensure_vm_and_get_rpc_port_for_session\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "session\.rebound" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "continuity_rebinds_session_after_primary_vmd_loss" "$REPO_ROOT/crates/reson-sandbox/tests/facade_contract.rs" >/dev/null

log "tier-b continuity gate: node-loss continuity contract test"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract continuity_rebinds_session_after_primary_vmd_loss -- --nocapture)

log "tier-b continuity gate: passed"
