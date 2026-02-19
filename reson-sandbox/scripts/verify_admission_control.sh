#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run admission-control gate"
    exit 1
  fi
  warn "runtime sources missing; skipping admission-control gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "admission-control gate: static scheduler/admission wiring checks"
rg -n "max_active_vms" "$REPO_ROOT/vmd/src/config.rs" >/dev/null
rg -n "enforce_create_vm_capacity|CapacityExceeded" "$REPO_ROOT/vmd/src/state/manager.rs" >/dev/null
rg -n "resource_exhausted|retry_after_ms=2000" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "max_inflight_commands|overload_retry_after_ms|handle_overloaded_command_message|AckKind::Nak\\(Some\\(Duration::from_millis\\(retry_after_ms\\)\\)\\)" "$REPO_ROOT/vmd/src/control_bus.rs" "$REPO_ROOT/vmd/src/config.rs" >/dev/null
rg -n "tenant_session_quota|workspace_session_quota|enforce_admission_budgets|admission\\.decision" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "region|zone|rack|score_node|workspace_endpoint_usage|zone_usage" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" "$REPO_ROOT/vmd/src/registry.rs" "$REPO_ROOT/vmd/src/config.rs" >/dev/null
rg -n "shard_for_key|legacy_session_key|legacy_session_fence_key|decode_session_route_record" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null

log "admission-control gate: unit tests"
(cd "$REPO_ROOT" && cargo test -p vmd create_vm_capacity_limit_rejects_when_active_vm_limit_reached -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --features distributed-control eligible_routes_filters_nodes_at_capacity -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --features distributed-control admission_budget_violation_detects_tenant_and_workspace_limits -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --features distributed-control sharding_and_scope_helpers_are_stable -- --nocapture)

log "admission-control gate: passed"
