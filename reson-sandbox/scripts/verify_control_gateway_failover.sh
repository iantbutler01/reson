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
require_cmd cargo

log "control gateway failover gate: static wiring checks"
rg -n "control_gateway_endpoints" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "candidate_endpoints\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "endpoint_for_new_session\(" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null

log "control gateway failover gate: facade contract test"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox control_gateway_failover_prefers_healthy_secondary_endpoint -- --nocapture)

log "control gateway failover gate: passed"
