#!/usr/bin/env bash
# @dive-file: Runs real warm-pool/cold-start integration tests across distributed and local auto-spawn modes.
# @dive-rel: Boots compose-backed etcd+nats and a live vmd node for distributed warm-pool evidence checks.
# @dive-rel: Executes crates/reson-sandbox/tests/real_warm_pool_pipeline.rs selectors for section 7.4 checklist coverage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
KEEP_STACK=0

usage() {
  cat <<'USAGE'
Usage: verify_real_warm_pool.sh [--profile <name>] [--profile-file <path>] [--keep-stack]
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

NODE1_ID="${RESON_SANDBOX_INTEGRATION_WARM_NODE1_ID:-integration-warm-node-1}"
NODE1_PORT="${RESON_SANDBOX_INTEGRATION_WARM_NODE1_PORT:-19186}"
REGISTRY_TTL_SECS="${RESON_SANDBOX_INTEGRATION_WARM_REGISTRY_TTL_SECS:-6}"
READY_TIMEOUT_SECS="${RESON_SANDBOX_INTEGRATION_WARM_READY_TIMEOUT_SECS:-90}"
CONTROL_SUBJECT_PREFIX="${RESON_SANDBOX_INTEGRATION_WARM_SUBJECT_PREFIX:-reson.sandbox.control.integration.warm}"
CONTROL_STREAM_NAME="${RESON_SANDBOX_INTEGRATION_WARM_STREAM_NAME:-RESON_SANDBOX_CONTROL_WARM}"

VMD_BIN="${VMD_BIN:-$REPO_ROOT/target/debug/vmd}"
VMDCTL_BIN="${VMDCTL_BIN:-$REPO_ROOT/target/debug/vmdctl}"
if [[ ! -x "$VMD_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmd" ]]; then
  VMD_BIN="$REPO_ROOT/vmd/target/debug/vmd"
fi
if [[ ! -x "$VMDCTL_BIN" && -x "$REPO_ROOT/vmd/target/debug/vmdctl" ]]; then
  VMDCTL_BIN="$REPO_ROOT/vmd/target/debug/vmdctl"
fi

TMP_DIR="$(mktemp -d "/tmp/rsb-real-warm.XXXXXX")"
DATA_DIR="$TMP_DIR/data"
mkdir -p "$DATA_DIR"
NODE1_LOG="$TMP_DIR/node1.log"
NODE1_PID=""
FAILED=0

ensure_portproxy_bins() {
  # @dive: Local auto-spawn warm-pool probes must boot guest bootstrap payloads, so we guarantee arch-matched guest portproxy artifacts exist before tests run.
  if [[ -n "${PROXY_BIN:-}" ]]; then
    return
  fi

  local default_proxy_dir="$REPO_ROOT/portproxy/bin"
  if [[ ! -f "$default_proxy_dir/portproxy-linux-amd64" || ! -f "$default_proxy_dir/portproxy-linux-arm64" ]]; then
    integration_log "building missing guest portproxy binaries"
    "$REPO_ROOT/scripts/build_portproxy_guest_bins_docker.sh" --all
  fi
  export PROXY_BIN="$default_proxy_dir"
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
    --reason "real-warm-pool-failure" >/dev/null 2>&1 || true
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

integration_log "starting shared integration control plane (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
"$SCRIPT_DIR/up.sh" "${profile_args[@]}" >/dev/null

integration_log "building vmd binaries for real warm-pool probes"
(cd "$REPO_ROOT" && cargo build -p vmd --bin vmd --bin vmdctl >/dev/null)
[[ -x "$VMD_BIN" ]] || { integration_err "missing vmd binary: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { integration_err "missing vmdctl binary: $VMDCTL_BIN"; exit 1; }
ensure_portproxy_bins

start_node

endpoint="http://127.0.0.1:${NODE1_PORT}"
selectors=(
  "startup_prewarm_executes_before_first_session_and_reports_warm_pool_hit"
  "warm_pool_hit_path_avoids_image_download_and_conversion_on_request_path"
  "cold_hit_triggers_async_refill_and_emits_refill_evidence"
  "local_auto_spawn_prewarm_avoids_first_command_transport_reset_spam"
)

# @dive: Warm-pool probes run with single-threaded test execution and selector isolation to avoid cross-test subject stream interleaving.
for selector in "${selectors[@]}"; do
  (
    cd "$REPO_ROOT"
    RUST_TEST_THREADS=1 \
    RESON_SANDBOX_REAL_PRIMARY_ENDPOINT="$endpoint" \
    RESON_SANDBOX_REAL_ETCD_ENDPOINTS="$RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS" \
    RESON_SANDBOX_REAL_ETCD_PREFIX="$NODE_PREFIX" \
    RESON_SANDBOX_REAL_NATS_URL="$RESON_SANDBOX_INTEGRATION_NATS_URL" \
    RESON_SANDBOX_REAL_NATS_SUBJECT_PREFIX="$CONTROL_SUBJECT_PREFIX" \
    RESON_SANDBOX_REAL_NATS_STREAM="$CONTROL_STREAM_NAME" \
    RESON_SANDBOX_REAL_WARM_POOL_DATA_DIR="$DATA_DIR" \
    RESON_SANDBOX_REAL_VMD_BIN="$VMD_BIN" \
      cargo test \
        -p reson-sandbox \
        --test real_warm_pool_pipeline \
        "$selector" \
        -- --ignored --exact --nocapture
  )
done

integration_log "real warm-pool/cold-start probes passed"
