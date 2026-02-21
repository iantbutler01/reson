#!/usr/bin/env bash
# @dive-file: Real integration probe that launches two vmd nodes against shared etcd and validates lease-backed registry heartbeats.
# @dive-rel: Builds on scripts/integration/up.sh for control-plane lifecycle and scripts/integration/common.sh compose/profile helpers.
# @dive-rel: Implements checklist item 7.2.1 from specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md with machine-verifiable etcd evidence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
KEEP_STACK=0
RUN_FACADE_TEST=0
RUN_DRAIN_TEST=0

usage() {
  cat <<'USAGE'
Usage: verify_two_node_registry.sh [--profile <name>] [--profile-file <path>] [--keep-stack] [--run-facade-test] [--run-drain-test]
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
    --run-facade-test)
      RUN_FACADE_TEST=1
      shift
      ;;
    --run-drain-test)
      RUN_DRAIN_TEST=1
      shift
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

NODE1_ID="${RESON_SANDBOX_INTEGRATION_NODE1_ID:-integration-node-1}"
NODE2_ID="${RESON_SANDBOX_INTEGRATION_NODE2_ID:-integration-node-2}"
NODE1_PORT="${RESON_SANDBOX_INTEGRATION_NODE1_PORT:-19072}"
NODE2_PORT="${RESON_SANDBOX_INTEGRATION_NODE2_PORT:-19073}"
NODE1_ADMISSION_FROZEN="${RESON_SANDBOX_INTEGRATION_NODE1_ADMISSION_FROZEN:-$RUN_DRAIN_TEST}"
NODE2_ADMISSION_FROZEN="${RESON_SANDBOX_INTEGRATION_NODE2_ADMISSION_FROZEN:-0}"
HEARTBEAT_OBSERVE_SECS="${RESON_SANDBOX_INTEGRATION_HEARTBEAT_OBSERVE_SECS:-3}"
REGISTRY_TTL_SECS="${RESON_SANDBOX_INTEGRATION_REGISTRY_TTL_SECS:-4}"
READY_TIMEOUT_SECS="${RESON_SANDBOX_INTEGRATION_NODE_READY_TIMEOUT_SECS:-60}"

VMD_BIN="${VMD_BIN:-$REPO_ROOT/target/debug/vmd}"
VMDCTL_BIN="${VMDCTL_BIN:-$REPO_ROOT/target/debug/vmdctl}"
if [[ ! -x "$VMD_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmd" ]]; then
  VMD_BIN="$REPO_ROOT/vmd/target/debug/vmd"
fi
if [[ ! -x "$VMDCTL_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmdctl" ]]; then
  VMDCTL_BIN="$REPO_ROOT/vmd/target/debug/vmdctl"
fi

TMP_DIR="$(mktemp -d "/tmp/rsb-two-node.XXXXXX")"
NODE1_DIR="$TMP_DIR/node1"
NODE2_DIR="$TMP_DIR/node2"
NODE1_LOG="$TMP_DIR/node1.log"
NODE2_LOG="$TMP_DIR/node2.log"
NODE1_PID=""
NODE2_PID=""
FAILED=0

collect_failure_artifacts() {
  local bundle_dir
  bundle_dir="$(artifact_bundle_dir)"
  mkdir -p "$bundle_dir/node_logs"
  cp "$NODE1_LOG" "$bundle_dir/node_logs/node1.log" 2>/dev/null || true
  cp "$NODE2_LOG" "$bundle_dir/node_logs/node2.log" 2>/dev/null || true
  "$SCRIPT_DIR/collect-artifacts.sh" "${profile_args[@]}" \
    --output-dir "$bundle_dir/control_plane" \
    --reason "two-node-registry-failure" >/dev/null 2>&1 || true
  integration_warn "failure artifacts captured at: $bundle_dir"
}

cleanup() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    FAILED=1
  fi

  set +e
  if [[ -n "$NODE1_PID" ]] && kill -0 "$NODE1_PID" >/dev/null 2>&1; then
    kill "$NODE1_PID" >/dev/null 2>&1 || true
    wait "$NODE1_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$NODE2_PID" ]] && kill -0 "$NODE2_PID" >/dev/null 2>&1; then
    kill "$NODE2_PID" >/dev/null 2>&1 || true
    wait "$NODE2_PID" >/dev/null 2>&1 || true
  fi
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

mkdir -p "$NODE1_DIR" "$NODE2_DIR"

integration_log "starting shared integration control plane (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
"$SCRIPT_DIR/up.sh" "${profile_args[@]}" >/dev/null

integration_log "building vmd binaries for two-node registry probe"
(cd "$REPO_ROOT" && cargo build -p vmd --bin vmd --bin vmdctl >/dev/null)
[[ -x "$VMD_BIN" ]] || { integration_err "missing vmd binary: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { integration_err "missing vmdctl binary: $VMDCTL_BIN"; exit 1; }

start_node() {
  local node_id="$1"
  local port="$2"
  local data_dir="$3"
  local log_file="$4"
  local admission_frozen="${5:-0}"
  local args=(
    --listen "127.0.0.1:${port}"
    --data-dir "$data_dir"
    --node-id "$node_id"
    --advertise-endpoint "http://127.0.0.1:${port}"
    --registry-etcd-endpoints "$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS"
    --registry-prefix "$NODE_PREFIX"
    --registry-ttl-secs "$REGISTRY_TTL_SECS"
    --disable-control-bus
  )
  if [[ "$admission_frozen" == "1" ]]; then
    args+=(--admission-frozen)
  fi
  "$VMD_BIN" "${args[@]}" >"$log_file" 2>&1 &
  echo $!
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
      tail -n 80 "$log_file" >&2 || true
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
  tail -n 80 "$log_file" >&2 || true
  return 1
}

fetch_registry_json() {
  run_compose exec -T etcd1 etcdctl \
    --endpoints=http://etcd1:2379 \
    get --prefix "${NODE_PREFIX}/nodes/" --write-out=json
}

extract_updated_at_ms() {
  local registry_json="$1"
  local node_key="$2"
  echo "$registry_json" | jq -r --arg key "$node_key" '
    (.kvs // [])
    | map(select((.key | @base64d) == $key))
    | if length == 0 then "" else (.[0].value | @base64d | fromjson | .updated_at_unix_ms | tostring) end
  '
}

assert_node_registered() {
  local node_id="$1"
  local ts="$2"
  if [[ -z "$ts" ]]; then
    integration_err "missing registry heartbeat for node: $node_id"
    return 1
  fi
  if ! [[ "$ts" =~ ^[0-9]+$ ]]; then
    integration_err "invalid heartbeat timestamp for node ${node_id}: $ts"
    return 1
  fi
}

node1_server="http://127.0.0.1:${NODE1_PORT}"
node2_server="http://127.0.0.1:${NODE2_PORT}"
node1_key="${NODE_PREFIX}/nodes/${NODE1_ID}"
node2_key="${NODE_PREFIX}/nodes/${NODE2_ID}"

integration_log "starting node ${NODE1_ID} on ${node1_server}"
NODE1_PID="$(start_node "$NODE1_ID" "$NODE1_PORT" "$NODE1_DIR" "$NODE1_LOG" "$NODE1_ADMISSION_FROZEN")"
integration_log "starting node ${NODE2_ID} on ${node2_server}"
NODE2_PID="$(start_node "$NODE2_ID" "$NODE2_PORT" "$NODE2_DIR" "$NODE2_LOG" "$NODE2_ADMISSION_FROZEN")"

wait_node_ready "$NODE1_ID" "$node1_server" "$NODE1_PID" "$NODE1_LOG"
wait_node_ready "$NODE2_ID" "$node2_server" "$NODE2_PID" "$NODE2_LOG"

registry_before="$(fetch_registry_json)"
node1_ts_before="$(extract_updated_at_ms "$registry_before" "$node1_key")"
node2_ts_before="$(extract_updated_at_ms "$registry_before" "$node2_key")"
assert_node_registered "$NODE1_ID" "$node1_ts_before"
assert_node_registered "$NODE2_ID" "$node2_ts_before"

# @dive: Confirm heartbeat cadence by observing updated_at_unix_ms advance for both node keys.
sleep "$HEARTBEAT_OBSERVE_SECS"

registry_after="$(fetch_registry_json)"
node1_ts_after="$(extract_updated_at_ms "$registry_after" "$node1_key")"
node2_ts_after="$(extract_updated_at_ms "$registry_after" "$node2_key")"
assert_node_registered "$NODE1_ID" "$node1_ts_after"
assert_node_registered "$NODE2_ID" "$node2_ts_after"

if (( node1_ts_after <= node1_ts_before )); then
  integration_err "heartbeat did not advance for node ${NODE1_ID}: before=${node1_ts_before} after=${node1_ts_after}"
  exit 1
fi
if (( node2_ts_after <= node2_ts_before )); then
  integration_err "heartbeat did not advance for node ${NODE2_ID}: before=${node2_ts_before} after=${node2_ts_after}"
  exit 1
fi

integration_log "registry keys verified for two-node cluster"
integration_log "node1 heartbeat: ${node1_ts_before} -> ${node1_ts_after}"
integration_log "node2 heartbeat: ${node2_ts_before} -> ${node2_ts_after}"

if [[ "$RUN_FACADE_TEST" -eq 1 ]]; then
  integration_log "running real facade control-gateway routing integration test"
  # @dive: Force unhealthy primary endpoint so the facade must use configured control gateways.
  (
    cd "$REPO_ROOT"
    RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="http://127.0.0.1:19079" \
    RESON_SANDBOX_REAL_GATEWAY_ENDPOINTS="${node1_server},${node2_server}" \
      cargo test \
        -p reson-sandbox \
        --test real_control_gateway_routing \
        facade_routes_through_control_gateways_on_real_daemons \
        -- --ignored --exact --nocapture
  )
fi

if [[ "$RUN_DRAIN_TEST" -eq 1 ]]; then
  integration_log "running real planned-drain handoff integration test"
  # @dive: Primary starts admission-frozen so distributed admission must hand off new sessions while anchored primary work stays live.
  (
    cd "$REPO_ROOT"
    RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="$node1_server" \
    RESON_SANDBOX_REAL_SECONDARY_ENDPOINT="$node2_server" \
    RESON_SANDBOX_REAL_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
    RESON_SANDBOX_REAL_ETCD_PREFIX="$NODE_PREFIX" \
    RESON_SANDBOX_REAL_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
    RESON_SANDBOX_REAL_NATS_SUBJECT_PREFIX="reson.sandbox.control.integration.registry" \
    RESON_SANDBOX_REAL_NATS_STREAM="RESON_SANDBOX_CONTROL_REGISTRY" \
      cargo test \
        -p reson-sandbox \
        --test real_control_plane_failures \
        planned_drain_admission_freeze_preserves_inflight_and_hands_off_new_sessions \
        -- --ignored --exact --nocapture
  )
fi

integration_log "two-node registry probe passed"
