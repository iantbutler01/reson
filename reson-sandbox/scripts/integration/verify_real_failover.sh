#!/usr/bin/env bash
# @dive-file: Runs real failover continuity and exactly-once integration tests against two live vmd daemons on shared storage.
# @dive-rel: Depends on scripts/integration/common.sh and compose control-plane bootstrap from scripts/integration/up.sh.
# @dive-rel: Executes crates/reson-sandbox/tests/real_failover_continuity.rs to satisfy the 7.3 continuity/stream-resume/no-rerun checklist items.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
KEEP_STACK=0
SKIP_PORTPROXY_BUILD=0
DISTRIBUTED_MODE=0
SELECTOR=""

usage() {
  cat <<'USAGE'
Usage: verify_real_failover.sh [--profile <name>] [--profile-file <path>] [--keep-stack] [--skip-portproxy-build] [--distributed] [--selector <test_name>]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --profile-file)
      PROFILE_FILE="$2"
      shift 2
      ;;
    --keep-stack)
      KEEP_STACK=1
      shift
      ;;
    --skip-portproxy-build)
      SKIP_PORTPROXY_BUILD=1
      shift
      ;;
    --distributed)
      DISTRIBUTED_MODE=1
      shift
      ;;
    --selector)
      SELECTOR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      integration_err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

require_cmd cargo
require_cmd docker
require_cmd jq
require_cmd lsof

load_profile_env "$PROFILE" "$PROFILE_FILE"

profile_args=(--profile "$PROFILE")
if [[ -n "$PROFILE_FILE" ]]; then
  profile_args+=(--profile-file "$PROFILE_FILE")
fi

NODE_PREFIX="${RESON_SANDBOX_INTEGRATION_NODE_KEY_PREFIX:-/reson-sandbox}"
NODE_PREFIX="${NODE_PREFIX%/}"
if [[ -z "$NODE_PREFIX" ]]; then
  NODE_PREFIX="/reson-sandbox"
fi

NODE1_ID="${RESON_SANDBOX_INTEGRATION_FAILOVER_NODE1_ID:-integration-failover-node-1}"
NODE2_ID="${RESON_SANDBOX_INTEGRATION_FAILOVER_NODE2_ID:-integration-failover-node-2}"
NODE1_PORT="${RESON_SANDBOX_INTEGRATION_FAILOVER_NODE1_PORT:-19182}"
NODE2_PORT="${RESON_SANDBOX_INTEGRATION_FAILOVER_NODE2_PORT:-19183}"
REGISTRY_TTL_SECS="${RESON_SANDBOX_INTEGRATION_FAILOVER_REGISTRY_TTL_SECS:-6}"
READY_TIMEOUT_SECS="${RESON_SANDBOX_INTEGRATION_FAILOVER_READY_TIMEOUT_SECS:-90}"
VM_SOURCE_REF="${RESON_SANDBOX_INTEGRATION_VM_SOURCE_REF:-${BRACKET_VM_IMAGE:-ghcr.io/bracketdevelopers/uv-builder:main}}"
CONTROL_SUBJECT_PREFIX="${RESON_SANDBOX_INTEGRATION_CONTROL_SUBJECT_PREFIX:-reson.sandbox.control.integration.failover}"
CONTROL_STREAM_NAME="${RESON_SANDBOX_INTEGRATION_CONTROL_STREAM_NAME:-RESON_SANDBOX_CONTROL_FAILOVER}"
# @dive: Default attach flow forces a secondary-side VM stop before reattach in this harness to avoid stale runtime ownership blocking local failover checks.
FORCE_STOP_BEFORE_ATTACH="${RESON_SANDBOX_REAL_FORCE_STOP_BEFORE_ATTACH:-1}"

VMD_BIN="${VMD_BIN:-$REPO_ROOT/target/debug/vmd}"
VMDCTL_BIN="${VMDCTL_BIN:-$REPO_ROOT/target/debug/vmdctl}"
if [[ ! -x "$VMD_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmd" ]]; then
  VMD_BIN="$REPO_ROOT/vmd/target/debug/vmd"
fi
if [[ ! -x "$VMDCTL_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmdctl" ]]; then
  VMDCTL_BIN="$REPO_ROOT/vmd/target/debug/vmdctl"
fi

TMP_DIR="$(mktemp -d "/tmp/rsb-real-failover.XXXXXX")"
SHARED_DATA_DIR="$TMP_DIR/shared-data"
mkdir -p "$SHARED_DATA_DIR"
NODE1_LOG="$TMP_DIR/node1.log"
NODE2_LOG="$TMP_DIR/node2.log"
NODE1_PID=""
NODE2_PID=""
SECONDARY_LAUNCHER_PID=""
SECONDARY_PID_FILE="$TMP_DIR/secondary.pid"
FAILED=0

ensure_portproxy_bins() {
  if [[ -n "${PROXY_BIN:-}" ]]; then
    return
  fi

  local default_proxy_dir="$REPO_ROOT/portproxy/bin"
  if [[ ! -f "$default_proxy_dir/portproxy-linux-amd64" || ! -f "$default_proxy_dir/portproxy-linux-arm64" ]]; then
    if [[ "$SKIP_PORTPROXY_BUILD" -eq 1 ]]; then
      integration_err "missing guest portproxy binaries in $default_proxy_dir and --skip-portproxy-build set"
      exit 1
    fi
    integration_log "building missing guest portproxy binaries"
    "$REPO_ROOT/scripts/build_portproxy_guest_bins_docker.sh" --all
  fi
  export PROXY_BIN="$default_proxy_dir"
}

kill_stale_listeners() {
  local port="$1"
  local stale_pids
  stale_pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$stale_pids" ]]; then
    return 0
  fi

  integration_warn "cleaning stale listener(s) on port ${port}: ${stale_pids//$'\n'/,}"
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    kill "$pid" >/dev/null 2>&1 || true
  done <<<"$stale_pids"

  sleep 1

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    if kill -0 "$pid" >/dev/null 2>&1; then
      integration_warn "forcing stale listener pid $pid off port $port"
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  done <<<"$stale_pids"
}

wait_node_ready() {
  local node_id="$1"
  local server="$2"
  local pid="$3"
  local log_file="$4"
  local elapsed=0

  while (( elapsed < READY_TIMEOUT_SECS )); do
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      integration_err "node process exited before ready: $node_id"
      tail -n 120 "$log_file" >&2 || true
      return 1
    fi
    if "$VMDCTL_BIN" --server "$server" --timeout-secs 3 list-vms >/dev/null 2>&1; then
      integration_log "node ready: $node_id ($server)"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  integration_err "node did not become ready in ${READY_TIMEOUT_SECS}s: $node_id"
  tail -n 120 "$log_file" >&2 || true
  return 1
}

start_nodes() {
  kill_stale_listeners "$NODE1_PORT"
  kill_stale_listeners "$NODE2_PORT"

  rm -rf "$SHARED_DATA_DIR"
  mkdir -p "$SHARED_DATA_DIR"
  : >"$NODE1_LOG"
  : >"$NODE2_LOG"
  rm -f "$SECONDARY_PID_FILE"

  local node1_endpoint="http://127.0.0.1:${NODE1_PORT}"

  integration_log "starting failover node ${NODE1_ID} on ${node1_endpoint}"
  if [[ "$DISTRIBUTED_MODE" -eq 1 ]]; then
    RESON_SANDBOX_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
    RESON_SANDBOX_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
    RESON_SANDBOX_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
    RESON_SANDBOX_NATS_STREAM="$CONTROL_STREAM_NAME" \
      "$VMD_BIN" \
      --listen "127.0.0.1:${NODE1_PORT}" \
      --data-dir "$SHARED_DATA_DIR" \
      --node-id "$NODE1_ID" \
      --advertise-endpoint "$node1_endpoint" \
      --registry-etcd-endpoints "$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
      --registry-prefix "$NODE_PREFIX" \
      --registry-ttl-secs "$REGISTRY_TTL_SECS" \
      --control-nats-url "$RESON_SANDBOX_INTEGRATION_NATS_URL" \
      --control-subject-prefix "$CONTROL_SUBJECT_PREFIX" \
      --control-node-id "$NODE1_ID" >"$NODE1_LOG" 2>&1 &
  else
    "$VMD_BIN" \
      --listen "127.0.0.1:${NODE1_PORT}" \
      --data-dir "$SHARED_DATA_DIR" \
      --node-id "$NODE1_ID" \
      --advertise-endpoint "$node1_endpoint" \
      --registry-etcd-endpoints "$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
      --registry-prefix "$NODE_PREFIX" \
      --registry-ttl-secs "$REGISTRY_TTL_SECS" \
      --disable-control-bus >"$NODE1_LOG" 2>&1 &
  fi
  NODE1_PID="$!"

  wait_node_ready "$NODE1_ID" "$node1_endpoint" "$NODE1_PID" "$NODE1_LOG"

  # @dive: Continuity probe intentionally cold-starts secondary only after primary loss so failover validates restart + rebind semantics.
  (
    while kill -0 "$NODE1_PID" >/dev/null 2>&1; do
      sleep 1
    done
    node2_endpoint="http://127.0.0.1:${NODE2_PORT}"
    if [[ "$DISTRIBUTED_MODE" -eq 1 ]]; then
      RESON_SANDBOX_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
      RESON_SANDBOX_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
      RESON_SANDBOX_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
      RESON_SANDBOX_NATS_STREAM="$CONTROL_STREAM_NAME" \
        "$VMD_BIN" \
        --listen "127.0.0.1:${NODE2_PORT}" \
        --data-dir "$SHARED_DATA_DIR" \
        --node-id "$NODE2_ID" \
        --advertise-endpoint "$node2_endpoint" \
        --registry-etcd-endpoints "$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
        --registry-prefix "$NODE_PREFIX" \
        --registry-ttl-secs "$REGISTRY_TTL_SECS" \
        --control-nats-url "$RESON_SANDBOX_INTEGRATION_NATS_URL" \
        --control-subject-prefix "$CONTROL_SUBJECT_PREFIX" \
        --control-node-id "$NODE2_ID" >"$NODE2_LOG" 2>&1 &
    else
      "$VMD_BIN" \
        --listen "127.0.0.1:${NODE2_PORT}" \
        --data-dir "$SHARED_DATA_DIR" \
        --node-id "$NODE2_ID" \
        --advertise-endpoint "$node2_endpoint" \
        --registry-etcd-endpoints "$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
        --registry-prefix "$NODE_PREFIX" \
        --registry-ttl-secs "$REGISTRY_TTL_SECS" \
        --disable-control-bus >"$NODE2_LOG" 2>&1 &
    fi
    echo $! >"$SECONDARY_PID_FILE"
  ) &
  SECONDARY_LAUNCHER_PID="$!"
}

stop_nodes() {
  set +e
  if [[ -n "$SECONDARY_LAUNCHER_PID" ]] && kill -0 "$SECONDARY_LAUNCHER_PID" >/dev/null 2>&1; then
    kill "$SECONDARY_LAUNCHER_PID" >/dev/null 2>&1 || true
    wait "$SECONDARY_LAUNCHER_PID" >/dev/null 2>&1 || true
  fi
  SECONDARY_LAUNCHER_PID=""

  if [[ -f "$SECONDARY_PID_FILE" ]]; then
    NODE2_PID="$(cat "$SECONDARY_PID_FILE" 2>/dev/null || true)"
  fi

  if [[ -n "$NODE1_PID" ]] && kill -0 "$NODE1_PID" >/dev/null 2>&1; then
    kill "$NODE1_PID" >/dev/null 2>&1 || true
    wait "$NODE1_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$NODE2_PID" ]] && kill -0 "$NODE2_PID" >/dev/null 2>&1; then
    kill "$NODE2_PID" >/dev/null 2>&1 || true
    wait "$NODE2_PID" >/dev/null 2>&1 || true
  fi
  NODE1_PID=""
  NODE2_PID=""
  kill_stale_listeners "$NODE1_PORT"
  kill_stale_listeners "$NODE2_PORT"
  rm -f "$SECONDARY_PID_FILE"
}

collect_failure_artifacts() {
  local bundle_dir
  bundle_dir="$(artifact_bundle_dir)"
  mkdir -p "$bundle_dir/node_logs"
  cp "$NODE1_LOG" "$bundle_dir/node_logs/node1.log" 2>/dev/null || true
  cp "$NODE2_LOG" "$bundle_dir/node_logs/node2.log" 2>/dev/null || true
  "$SCRIPT_DIR/collect-artifacts.sh" "${profile_args[@]}" \
    --output-dir "$bundle_dir/control_plane" \
    --reason "real-failover-failure" >/dev/null 2>&1 || true
  integration_warn "failure artifacts captured at: $bundle_dir"
}

cleanup() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    FAILED=1
  fi
  stop_nodes
  if [[ "$KEEP_STACK" -ne 1 ]]; then
    "$SCRIPT_DIR/down.sh" "${profile_args[@]}" >/dev/null 2>&1 || true
  fi
  if [[ "$FAILED" -eq 1 ]]; then
    collect_failure_artifacts
  fi
  rm -rf "$TMP_DIR"
  exit "$exit_code"
}
trap cleanup EXIT

run_real_test() {
  local selector="$1"
  local primary_endpoint="http://127.0.0.1:${NODE1_PORT}"
  local secondary_endpoint="http://127.0.0.1:${NODE2_PORT}"

  integration_log "running real failover test selector: $selector"
  if [[ "$DISTRIBUTED_MODE" -eq 1 ]]; then
    (
      cd "$REPO_ROOT"
      BRACKET_VM_IMAGE="$VM_SOURCE_REF" \
      RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="$primary_endpoint" \
      RESON_SANDBOX_REAL_SECONDARY_ENDPOINT="$secondary_endpoint" \
      RESON_SANDBOX_REAL_PRIMARY_PID="$NODE1_PID" \
      RESON_SANDBOX_REAL_FORCE_STOP_BEFORE_ATTACH="$FORCE_STOP_BEFORE_ATTACH" \
      RESON_SANDBOX_REAL_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
      RESON_SANDBOX_REAL_ETCD_PREFIX="$NODE_PREFIX" \
      RESON_SANDBOX_REAL_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
      RESON_SANDBOX_REAL_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
      RESON_SANDBOX_REAL_NATS_STREAM="$CONTROL_STREAM_NAME" \
        cargo test \
          -p reson-sandbox \
          --test real_failover_continuity \
          "$selector" \
          -- --ignored --exact --nocapture
    )
  else
    (
      cd "$REPO_ROOT"
      BRACKET_VM_IMAGE="$VM_SOURCE_REF" \
      RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="$primary_endpoint" \
      RESON_SANDBOX_REAL_SECONDARY_ENDPOINT="$secondary_endpoint" \
      RESON_SANDBOX_REAL_PRIMARY_PID="$NODE1_PID" \
      RESON_SANDBOX_REAL_FORCE_STOP_BEFORE_ATTACH="$FORCE_STOP_BEFORE_ATTACH" \
        cargo test \
          -p reson-sandbox \
          --test real_failover_continuity \
          "$selector" \
          -- --ignored --exact --nocapture
    )
  fi
}

integration_log "starting shared integration control plane (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
"$SCRIPT_DIR/up.sh" "${profile_args[@]}" >/dev/null

ensure_portproxy_bins

integration_log "building vmd binaries for real failover probe"
(cd "$REPO_ROOT" && cargo build -p vmd --bin vmd --bin vmdctl >/dev/null)
[[ -x "$VMD_BIN" ]] || { integration_err "missing vmd binary: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { integration_err "missing vmdctl binary: $VMDCTL_BIN"; exit 1; }

if [[ "$DISTRIBUTED_MODE" -eq 1 ]]; then
  tests=(
    "distributed_failover_rebind_updates_route_and_emits_events"
    "distributed_stream_resume_from_checkpoint_is_forward_only_without_replay"
    "distributed_stream_events_include_identity_envelope_and_monotonic_sequence"
    "distributed_terminal_stream_is_not_rerun_after_primary_failover"
    "distributed_mq_retry_dedupe_dead_letter_behavior_under_primary_loss"
  )
else
  tests=(
    "primary_node_loss_during_active_exec_stream_rebinds_session"
    "inflight_exec_is_acknowledged_exactly_once_under_primary_loss"
    "tierb_missing_restore_marker_fails_policy_under_failover_rebind"
    "tierb_restore_marker_rehydrates_and_resumes_under_failover_rebind"
  )
fi

if [[ -n "$SELECTOR" ]]; then
  tests=("$SELECTOR")
fi

for selector in "${tests[@]}"; do
  start_nodes
  run_real_test "$selector"
  stop_nodes
done

integration_log "real failover continuity + exactly-once probes passed"
