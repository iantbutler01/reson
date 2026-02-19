#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run control gateway failover gate"
    exit 1
  fi
  warn "runtime sources missing; skipping control gateway failover gate"
  exit 0
fi

require_cmd rg

log "control gateway failover gate: static wiring checks"
rg -n "control_gateway_endpoints" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "candidate_endpoints\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "endpoint_for_new_session\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
[[ -f "$REPO_ROOT/specs/runbooks/CONTROL_GATEWAY_HA_TOPOLOGY_AND_FAILOVER_DRILL.md" ]] || {
  err "missing control-gateway topology runbook"
  exit 1
}
[[ -x "$REPO_ROOT/scripts/run_control_gateway_failover_drill.sh" ]] || {
  err "missing executable control-gateway failover drill script"
  exit 1
}

log "control gateway failover gate: automated failover drill"
report_file_base="$(mktemp "/tmp/reson_sandbox_gateway_failover_gate.XXXXXX")"
report_file="${report_file_base}.txt"
"$REPO_ROOT/scripts/run_control_gateway_failover_drill.sh" --strict --output "$report_file"
rg -n "^status=passed$" "$report_file" >/dev/null

log "control gateway failover gate: passed"
