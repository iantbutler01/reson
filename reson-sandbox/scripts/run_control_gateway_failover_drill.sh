#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
OUTPUT=""

usage() {
  cat <<'USAGE'
Usage: run_control_gateway_failover_drill.sh [--strict] [--output <path>]

Runs the control-gateway failover drill and writes a machine-readable report.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run control-gateway failover drill"
    exit 1
  fi
  warn "runtime sources missing; skipping control-gateway failover drill"
  exit 0
fi

require_cmd cargo

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="/tmp/reson_sandbox_control_gateway_failover_drill_${timestamp}.txt"
fi
mkdir -p "$(dirname "$OUTPUT")"

status="passed"
failure_reason=""
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

log "control-gateway failover drill: running contract test"
set +e
(cd "$REPO_ROOT" && cargo test -p reson-sandbox control_gateway_failover_prefers_healthy_secondary_endpoint -- --nocapture)
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
  status="failed"
  failure_reason="cargo test exited with code $rc"
fi

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat >"$OUTPUT" <<EOF
drill=control_gateway_failover
status=$status
started_at_utc=$started_at
finished_at_utc=$finished_at
strict_mode=$STRICT
test_selector=control_gateway_failover_prefers_healthy_secondary_endpoint
failure_reason=$failure_reason
EOF

if [[ "$status" != "passed" ]]; then
  err "control-gateway failover drill failed (report: $OUTPUT)"
  exit 1
fi

log "control-gateway failover drill: passed (report: $OUTPUT)"
