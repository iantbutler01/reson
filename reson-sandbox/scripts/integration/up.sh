#!/usr/bin/env bash
# @dive-file: Brings up canonical real integration control-plane services for reson-sandbox testing.
# @dive-rel: Uses scripts/integration/common.sh for profile loading and compose orchestration.
# @dive-rel: Targets deploy/integration/docker-compose.control-plane.yml as the default canonical manifest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""

usage() {
  cat <<'USAGE'
Usage: up.sh [--profile <name>] [--profile-file <path>]
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

require_cmd docker
load_profile_env "$PROFILE" "$PROFILE_FILE"

integration_log "starting integration control plane (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
run_compose up -d

wait_for_http "http://127.0.0.1:${RESON_SANDBOX_INTEGRATION_ETCD_PORT}/health" 90 "etcd client endpoint"
wait_for_http "http://127.0.0.1:${RESON_SANDBOX_INTEGRATION_NATS_MONITOR_PORT}/varz" 90 "nats monitor endpoint"

integration_log "integration stack is up"
integration_log "etcd endpoints: ${RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS}"
integration_log "nats url: ${RESON_SANDBOX_INTEGRATION_NATS_URL}"
integration_log "project: ${RESON_SANDBOX_INTEGRATION_PROJECT}"
