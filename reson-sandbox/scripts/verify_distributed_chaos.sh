#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0

usage() {
  cat <<'USAGE'
Usage: verify_distributed_chaos.sh [--strict]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
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
    err "runtime sources missing; strict mode cannot run distributed chaos gate"
    exit 1
  fi
  warn "runtime sources missing; skipping distributed chaos gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "distributed chaos gate: static distributed-control wiring checks"
rg -n "control_bus::start_with_trigger|start_control_bus" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "reconcile::run_once|start_reconcile_worker" "$REPO_ROOT/vmd/src/app.rs" "$REPO_ROOT/vmd/src/reconcile.rs" >/dev/null
rg -n "idempotency_key|mark_or_duplicate" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "acquire_port_lease|PortAllocationLease|/ports/" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null

log "distributed chaos gate: daemon fail-stop/restart drill"
(cd "$REPO_ROOT/vmd" && cargo build --bin vmd --bin vmdctl)

default_vmd_bin="$REPO_ROOT/vmd/target/debug/vmd"
default_vmdctl_bin="$REPO_ROOT/vmd/target/debug/vmdctl"
if [[ ! -x "$default_vmd_bin" && -x "$REPO_ROOT/target/debug/vmd" ]]; then
  default_vmd_bin="$REPO_ROOT/target/debug/vmd"
fi
if [[ ! -x "$default_vmdctl_bin" && -x "$REPO_ROOT/target/debug/vmdctl" ]]; then
  default_vmdctl_bin="$REPO_ROOT/target/debug/vmdctl"
fi

VMD_BIN="${VMD_BIN:-$default_vmd_bin}"
VMDCTL_BIN="${VMDCTL_BIN:-$default_vmdctl_bin}"
[[ -x "$VMD_BIN" ]] || { err "vmd binary missing: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { err "vmdctl binary missing: $VMDCTL_BIN"; exit 1; }

SERVER_URL="${VMD_CHAOS_SERVER:-http://127.0.0.1:18062}"
LISTEN_ADDR="${VMD_CHAOS_LISTEN_ADDR:-${SERVER_URL#http://}}"
LISTEN_ADDR="${LISTEN_ADDR#https://}"
TIMEOUT_SECS="${VMD_CHAOS_TIMEOUT_SECS:-30}"

TMP_DIR="$(mktemp -d "/tmp/rsbchaos.XXXXXX")"
DATA_DIR="$TMP_DIR/vmd-data"
LOG_FILE="$TMP_DIR/vmd.log"
VMD_PID=""

cleanup() {
  set +e
  if [[ -n "$VMD_PID" ]] && kill -0 "$VMD_PID" >/dev/null 2>&1; then
    kill "$VMD_PID" >/dev/null 2>&1 || true
    wait "$VMD_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$DATA_DIR"

wait_ready() {
  local ready=0
  for _ in $(seq 1 60); do
    if [[ -n "$VMD_PID" ]] && ! kill -0 "$VMD_PID" >/dev/null 2>&1; then
      err "vmd exited before ready"
      tail -n 80 "$LOG_FILE" >&2 || true
      return 1
    fi
    if "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 list-vms >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  [[ "$ready" -eq 1 ]]
}

log "distributed chaos gate: start vmd"
"$VMD_BIN" --listen "$LISTEN_ADDR" --data-dir "$DATA_DIR" >"$LOG_FILE" 2>&1 &
VMD_PID="$!"
wait_ready || { err "vmd did not become ready"; tail -n 80 "$LOG_FILE" >&2 || true; exit 1; }

log "distributed chaos gate: kill vmd (fail-stop)"
kill -9 "$VMD_PID" >/dev/null 2>&1 || true
wait "$VMD_PID" >/dev/null 2>&1 || true
VMD_PID=""

log "distributed chaos gate: restart vmd on same data-dir"
"$VMD_BIN" --listen "$LISTEN_ADDR" --data-dir "$DATA_DIR" >>"$LOG_FILE" 2>&1 &
VMD_PID="$!"
wait_ready || { err "vmd restart did not become ready"; tail -n 120 "$LOG_FILE" >&2 || true; exit 1; }

"$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" list-vms >/dev/null
log "distributed chaos gate: passed"
