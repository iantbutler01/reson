#!/usr/bin/env bash
# @dive-file: Runs real distributed control-plane fail-closed + recovery tests for etcd quorum and NATS outages.
# @dive-rel: Uses scripts/integration/common.sh for compose/profile lifecycle and artifact collection.
# @dive-rel: Executes crates/reson-sandbox/tests/real_control_plane_failures.rs against live compose-backed etcd+nats services.
# @dive-rel: Includes explicit node restart probe for bounded reconcile convergence after recovery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
KEEP_STACK=0

usage() {
  cat <<'USAGE'
Usage: verify_control_plane_failures.sh [--profile <name>] [--profile-file <path>] [--keep-stack]
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

NODE1_ID="${RESON_SANDBOX_INTEGRATION_CP_NODE1_ID:-integration-control-node-1}"
NODE1_PORT="${RESON_SANDBOX_INTEGRATION_CP_NODE1_PORT:-19184}"
REGISTRY_TTL_SECS="${RESON_SANDBOX_INTEGRATION_CP_REGISTRY_TTL_SECS:-6}"
READY_TIMEOUT_SECS="${RESON_SANDBOX_INTEGRATION_CP_READY_TIMEOUT_SECS:-90}"
CONTROL_SUBJECT_PREFIX="${RESON_SANDBOX_INTEGRATION_CONTROL_SUBJECT_PREFIX:-reson.sandbox.control.integration.failures}"
CONTROL_STREAM_NAME="${RESON_SANDBOX_INTEGRATION_CONTROL_STREAM_NAME:-RESON_SANDBOX_CONTROL_FAILURES}"

VMD_BIN="${VMD_BIN:-$REPO_ROOT/target/debug/vmd}"
VMDCTL_BIN="${VMDCTL_BIN:-$REPO_ROOT/target/debug/vmdctl}"
if [[ ! -x "$VMD_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmd" ]]; then
  VMD_BIN="$REPO_ROOT/vmd/target/debug/vmd"
fi
if [[ ! -x "$VMDCTL_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmdctl" ]]; then
  VMDCTL_BIN="$REPO_ROOT/vmd/target/debug/vmdctl"
fi

TMP_DIR="$(mktemp -d "/tmp/rsb-control-failures.XXXXXX")"
DATA_DIR="$TMP_DIR/data"
mkdir -p "$DATA_DIR"
NODE1_LOG="$TMP_DIR/node1.log"
NODE1_PID=""
FAILED=0

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

start_node() {
  rm -rf "$DATA_DIR"
  mkdir -p "$DATA_DIR"
  : >"$NODE1_LOG"
  local endpoint="http://127.0.0.1:${NODE1_PORT}"

  RESON_SANDBOX_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
  RESON_SANDBOX_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
  RESON_SANDBOX_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
  RESON_SANDBOX_NATS_STREAM="$CONTROL_STREAM_NAME" \
    "$VMD_BIN" \
    --listen "127.0.0.1:${NODE1_PORT}" \
    --data-dir "$DATA_DIR" \
    --node-id "$NODE1_ID" \
    --advertise-endpoint "$endpoint" \
    --registry-etcd-endpoints "$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
    --registry-prefix "$NODE_PREFIX" \
    --registry-ttl-secs "$REGISTRY_TTL_SECS" \
    --control-nats-url "$RESON_SANDBOX_INTEGRATION_NATS_URL" \
    --control-subject-prefix "$CONTROL_SUBJECT_PREFIX" \
    --control-node-id "$NODE1_ID" >"$NODE1_LOG" 2>&1 &
  NODE1_PID="$!"
  wait_node_ready "$NODE1_ID" "$endpoint" "$NODE1_PID" "$NODE1_LOG"
}

stop_node() {
  set +e
  if [[ -n "$NODE1_PID" ]] && kill -0 "$NODE1_PID" >/dev/null 2>&1; then
    kill "$NODE1_PID" >/dev/null 2>&1 || true
    wait "$NODE1_PID" >/dev/null 2>&1 || true
  fi
  NODE1_PID=""
}

collect_failure_artifacts() {
  local bundle_dir
  bundle_dir="$(artifact_bundle_dir)"
  mkdir -p "$bundle_dir/node_logs"
  cp "$NODE1_LOG" "$bundle_dir/node_logs/node1.log" 2>/dev/null || true
  "$SCRIPT_DIR/collect-artifacts.sh" "${profile_args[@]}" \
    --output-dir "$bundle_dir/control_plane" \
    --reason "control-plane-failure-probe-failure" >/dev/null 2>&1 || true
  integration_warn "failure artifacts captured at: $bundle_dir"
}

cleanup() {
  local exit_code=$?
  if [[ "$exit_code" -ne 0 ]]; then
    FAILED=1
  fi
  stop_node
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

run_real_tests() {
  local endpoint="http://127.0.0.1:${NODE1_PORT}"
  local etcd2_id
  local etcd3_id
  local nats_id
  etcd2_id="$(run_compose ps -q etcd2 | head -n1)"
  etcd3_id="$(run_compose ps -q etcd3 | head -n1)"
  nats_id="$(run_compose ps -q nats | head -n1)"

  if [[ -z "$etcd2_id" || -z "$etcd3_id" || -z "$nats_id" ]]; then
    integration_err "failed to resolve etcd/nats container ids for outage probes"
    exit 1
  fi

  local selectors=(
    "etcd_quorum_loss_fails_closed_and_recovers"
    "nats_outage_fails_closed_and_recovers"
    "ownership_fence_conflicts_under_concurrent_mutators_resolve_deterministically"
  )

  for selector in "${selectors[@]}"; do
    # @dive: Control-plane fault probes use real container stop/start to assert fail-closed behavior and post-outage recovery on live services.
    (
      cd "$REPO_ROOT"
      RUST_TEST_THREADS=1 \
      RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="$endpoint" \
      RESON_SANDBOX_REAL_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
      RESON_SANDBOX_REAL_ETCD_PREFIX="$NODE_PREFIX" \
      RESON_SANDBOX_REAL_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
      RESON_SANDBOX_REAL_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
      RESON_SANDBOX_REAL_NATS_STREAM="$CONTROL_STREAM_NAME" \
      RESON_SANDBOX_REAL_ETCD_DEGRADE_CONTAINERS="$etcd2_id,$etcd3_id" \
      RESON_SANDBOX_REAL_NATS_CONTAINER_ID="$nats_id" \
        cargo test \
          -p reson-sandbox \
          --test real_control_plane_failures \
          "$selector" \
          -- --ignored --exact --nocapture
    )
  done
}

run_reconcile_convergence_after_restart() {
  local endpoint="http://127.0.0.1:${NODE1_PORT}"
  (
    cd "$REPO_ROOT"
    RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="$endpoint" \
    RESON_SANDBOX_REAL_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
    RESON_SANDBOX_REAL_ETCD_PREFIX="$NODE_PREFIX" \
    RESON_SANDBOX_REAL_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
    RESON_SANDBOX_REAL_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
    RESON_SANDBOX_REAL_NATS_STREAM="$CONTROL_STREAM_NAME" \
    RESON_SANDBOX_REAL_RECONCILE_CONVERGENCE_TIMEOUT_SECS="${RESON_SANDBOX_INTEGRATION_RECONCILE_CONVERGENCE_TIMEOUT_SECS:-30}" \
      cargo test \
        -p reson-sandbox \
        --test real_control_plane_failures \
        reconcile_converges_within_bound_after_node_restart \
        -- --ignored --exact --nocapture
  )
}

integration_log "starting shared integration control plane (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
"$SCRIPT_DIR/up.sh" "${profile_args[@]}" >/dev/null

integration_log "building vmd binaries for control-plane failure probes"
(cd "$REPO_ROOT" && cargo build -p vmd --bin vmd --bin vmdctl >/dev/null)
[[ -x "$VMD_BIN" ]] || { integration_err "missing vmd binary: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { integration_err "missing vmdctl binary: $VMDCTL_BIN"; exit 1; }

start_node
run_real_tests

integration_log "restarting node for reconcile convergence bound probe"
stop_node
start_node
run_reconcile_convergence_after_restart

integration_log "real control-plane failure probes passed"
