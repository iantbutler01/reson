#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run operational game-day gate"
    exit 1
  fi
  warn "runtime sources missing; skipping operational game-day gate"
  exit 0
fi

require_cmd rg

log "operational game-day gate: static contract checks"
[[ -f "$REPO_ROOT/specs/runbooks/CONTROL_PLANE_GAMEDAY_ETCD_MQ_ZONE_FAILURE.md" ]] || {
  err "missing game-day runbook"
  exit 1
}
[[ -f "$REPO_ROOT/deploy/gameday/docker-compose.control-plane.yml" ]] || {
  err "missing game-day docker compose fixture"
  exit 1
}
[[ -x "$REPO_ROOT/scripts/run_control_plane_gameday_drill.sh" ]] || {
  err "missing executable game-day drill script"
  exit 1
}

rg -n "etcd quorum|MQ outage|Zone failure" "$REPO_ROOT/specs/runbooks/CONTROL_PLANE_GAMEDAY_ETCD_MQ_ZONE_FAILURE.md" >/dev/null
rg -n "etcd1|etcd2|etcd3|nats" "$REPO_ROOT/deploy/gameday/docker-compose.control-plane.yml" >/dev/null
rg -n "etcd_quorum_loss_status|mq_outage_status|zone_failure_status" "$REPO_ROOT/scripts/run_control_plane_gameday_drill.sh" >/dev/null

if ! command -v docker >/dev/null 2>&1 || ! docker compose version >/dev/null 2>&1; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "docker + compose plugin are required in strict mode for operational game-day gate"
    exit 1
  fi
  warn "docker compose unavailable; skipping operational game-day execution"
  exit 0
fi

log "operational game-day gate: running automated drill"
report_file_base="$(mktemp "/tmp/reson_sandbox_operational_gameday_gate.XXXXXX")"
report_file="${report_file_base}.txt"
"$REPO_ROOT/scripts/run_control_plane_gameday_drill.sh" --strict --output "$report_file"

rg -n "^status=passed$" "$report_file" >/dev/null
rg -n "^etcd_quorum_loss_status=passed$" "$report_file" >/dev/null
rg -n "^mq_outage_status=passed$" "$report_file" >/dev/null
rg -n "^zone_failure_status=passed$" "$report_file" >/dev/null

log "operational game-day gate: passed"
