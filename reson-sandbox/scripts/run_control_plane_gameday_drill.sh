#!/usr/bin/env bash
# @dive-file: Runs etcd/NATS/zone-failure game-day scenarios against compose control-plane stack.
# @dive-rel: Invoked by verify_operational_gameday.sh gate for operational readiness evidence.
# @dive-rel: Produces structured drill report for CI/local failure triage.

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
OUTPUT=""

usage() {
  cat <<'USAGE'
Usage: run_control_plane_gameday_drill.sh [--strict] [--output <path>]

Runs operational game-day drills for etcd quorum loss, MQ outage, and zone failure.
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
    err "runtime sources missing; strict mode cannot run control-plane game-day drill"
    exit 1
  fi
  warn "runtime sources missing; skipping control-plane game-day drill"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "docker is required in strict mode"
    exit 1
  fi
  warn "docker not found; skipping control-plane game-day drill"
  exit 0
fi

if ! docker compose version >/dev/null 2>&1; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "docker compose plugin is required in strict mode"
    exit 1
  fi
  warn "docker compose plugin missing; skipping control-plane game-day drill"
  exit 0
fi

require_cmd curl

compose_file="$REPO_ROOT/deploy/gameday/docker-compose.control-plane.yml"
[[ -f "$compose_file" ]] || {
  err "missing compose file: ${compose_file#$REPO_ROOT/}"
  exit 1
}

etcd_port="${RESON_SANDBOX_GAMEDAY_ETCD_PORT:-32379}"
nats_port="${RESON_SANDBOX_GAMEDAY_NATS_PORT:-14222}"
nats_monitor_port="${RESON_SANDBOX_GAMEDAY_NATS_MONITOR_PORT:-18222}"
project_name="reson-sandbox-gameday-$$"
report_status="passed"

etcd_status="passed"
etcd_failure=""
mq_status="passed"
mq_failure=""
zone_status="passed"
zone_failure=""

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$OUTPUT" ]]; then
  OUTPUT="/tmp/reson_sandbox_control_plane_gameday_${timestamp}.txt"
fi
mkdir -p "$(dirname "$OUTPUT")"

compose() {
  docker compose -p "$project_name" -f "$compose_file" "$@"
}

cleanup() {
  set +e
  compose down -v --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

etcd_endpoints="http://etcd1:2379,http://etcd2:2379,http://etcd3:2379"

etcdctl_cluster() {
  compose exec -T -e ETCDCTL_API=3 etcd1 /usr/local/bin/etcdctl --endpoints="$etcd_endpoints" "$@"
}

etcdctl_single() {
  compose exec -T -e ETCDCTL_API=3 etcd1 /usr/local/bin/etcdctl --endpoints="http://etcd1:2379" "$@"
}

wait_for_etcd_cluster() {
  local ready=0
  for _ in $(seq 1 45); do
    if etcdctl_cluster endpoint health >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "$ready" -eq 1 ]]
}

wait_for_nats_monitor() {
  local ready=0
  local url="http://127.0.0.1:${nats_monitor_port}/healthz"
  for _ in $(seq 1 45); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]]
}

log "control-plane game-day: starting compose stack"
compose up -d --quiet-pull

if ! wait_for_etcd_cluster; then
  err "control-plane game-day: etcd cluster did not become healthy"
  exit 1
fi

if ! wait_for_nats_monitor; then
  err "control-plane game-day: nats monitor endpoint did not become healthy"
  exit 1
fi

log "control-plane game-day: scenario etcd quorum loss"
set +e
etcdctl_cluster put /reson-sandbox/gameday/baseline ok >/dev/null
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
  etcd_status="failed"
  etcd_failure="baseline write failed before quorum-loss simulation"
else
  compose stop etcd2 etcd3 >/dev/null
  set +e
  etcdctl_single --command-timeout=5s put /reson-sandbox/gameday/quorum-loss expected-fail >/dev/null
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    etcd_status="failed"
    etcd_failure="write unexpectedly succeeded during quorum loss"
  else
    compose start etcd2 etcd3 >/dev/null
    if ! wait_for_etcd_cluster; then
      etcd_status="failed"
      etcd_failure="cluster did not recover after quorum restoration"
    else
      set +e
      etcdctl_cluster put /reson-sandbox/gameday/post-recovery ok >/dev/null
      rc=$?
      set -e
      if [[ "$rc" -ne 0 ]]; then
        etcd_status="failed"
        etcd_failure="write failed after quorum recovery"
      fi
    fi
  fi
fi

log "control-plane game-day: scenario mq outage"
set +e
curl -fsS "http://127.0.0.1:${nats_monitor_port}/healthz" >/dev/null
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
  mq_status="failed"
  mq_failure="nats health endpoint unavailable before outage simulation"
else
  compose stop nats >/dev/null
  set +e
  curl -fsS "http://127.0.0.1:${nats_monitor_port}/healthz" >/dev/null
  rc=$?
  set -e
  if [[ "$rc" -eq 0 ]]; then
    mq_status="failed"
    mq_failure="nats monitor endpoint unexpectedly healthy during outage"
  else
    compose start nats >/dev/null
    if ! wait_for_nats_monitor; then
      mq_status="failed"
      mq_failure="nats monitor endpoint did not recover after restart"
    fi
  fi
fi

log "control-plane game-day: scenario zone failure"
zone_a_payload='{"node_id":"zone-a-node-1","endpoint":"http://node-a:18072","region":"dev","zone":"zone-a","rack":"rack-a"}'
zone_b1_payload='{"node_id":"zone-b-node-1","endpoint":"http://node-b:18072","region":"dev","zone":"zone-b","rack":"rack-b1"}'
zone_b2_payload='{"node_id":"zone-b-node-2","endpoint":"http://node-c:18072","region":"dev","zone":"zone-b","rack":"rack-b2"}'
set +e
etcdctl_cluster put /reson-sandbox/nodes/zone-a-node-1 "$zone_a_payload" >/dev/null
rc1=$?
etcdctl_cluster put /reson-sandbox/nodes/zone-b-node-1 "$zone_b1_payload" >/dev/null
rc2=$?
etcdctl_cluster put /reson-sandbox/nodes/zone-b-node-2 "$zone_b2_payload" >/dev/null
rc3=$?
set -e
if [[ "$rc1" -ne 0 || "$rc2" -ne 0 || "$rc3" -ne 0 ]]; then
  zone_status="failed"
  zone_failure="failed to seed node registry keys"
else
  etcdctl_cluster del /reson-sandbox/nodes/zone-a- --prefix >/dev/null
  set +e
  zone_a_keys="$(etcdctl_cluster get /reson-sandbox/nodes/zone-a- --prefix --keys-only 2>/dev/null)"
  zone_b_keys="$(etcdctl_cluster get /reson-sandbox/nodes/zone-b- --prefix --keys-only 2>/dev/null)"
  set -e
  if grep -q "/reson-sandbox/nodes/zone-a-" <<<"$zone_a_keys"; then
    zone_status="failed"
    zone_failure="zone-a keys still present after simulated zone failure"
  elif ! grep -q "/reson-sandbox/nodes/zone-b-" <<<"$zone_b_keys"; then
    zone_status="failed"
    zone_failure="surviving zone-b keys missing after zone-a failure"
  fi
fi

if [[ "$etcd_status" != "passed" || "$mq_status" != "passed" || "$zone_status" != "passed" ]]; then
  report_status="failed"
fi

finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat >"$OUTPUT" <<REPORT
drill=control_plane_gameday
status=$report_status
started_at_utc=$started_at
finished_at_utc=$finished_at
strict_mode=$STRICT
compose_file=${compose_file#$REPO_ROOT/}
etcd_port=$etcd_port
nats_port=$nats_port
nats_monitor_port=$nats_monitor_port
etcd_quorum_loss_status=$etcd_status
etcd_quorum_loss_failure=$etcd_failure
mq_outage_status=$mq_status
mq_outage_failure=$mq_failure
zone_failure_status=$zone_status
zone_failure_failure=$zone_failure
REPORT

if [[ "$report_status" != "passed" ]]; then
  err "control-plane game-day drill failed (report: $OUTPUT)"
  exit 1
fi

log "control-plane game-day drill: passed (report: $OUTPUT)"
