#!/usr/bin/env bash
# @dive-file: Shared helpers for real integration harness lifecycle scripts (up/down/reset/collect-artifacts).
# @dive-rel: Used by scripts/integration/*.sh to keep profile loading and docker compose execution consistent.
# @dive-rel: Resolves defaults from deploy/integration/profiles/*.env and deploy/integration/docker-compose.control-plane.yml.
set -euo pipefail

INTEGRATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$INTEGRATION_DIR/../.." && pwd)"
DEPLOY_ROOT="$REPO_ROOT/deploy/integration"
PROFILE_ROOT="$DEPLOY_ROOT/profiles"
DEFAULT_PROFILE="local-dev"

integration_log() {
  printf '[integration] %s\n' "$*"
}

integration_warn() {
  printf '[integration][warn] %s\n' "$*" >&2
}

integration_err() {
  printf '[integration][error] %s\n' "$*" >&2
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    integration_err "required command not found: $cmd"
    return 1
  }
}

resolve_compose_bin() {
  if docker compose version >/dev/null 2>&1; then
    echo "docker compose"
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose"
    return
  fi
  integration_err "docker compose CLI is required"
  return 1
}

abs_path() {
  local candidate="$1"
  if [[ "$candidate" = /* ]]; then
    printf '%s\n' "$candidate"
  else
    printf '%s\n' "$REPO_ROOT/$candidate"
  fi
}

load_profile_env() {
  local profile="$1"
  local profile_file="${2:-$PROFILE_ROOT/$profile.env}"

  if [[ ! -f "$profile_file" ]]; then
    integration_err "profile file not found: $profile_file"
    return 1
  fi

  set -a
  # shellcheck disable=SC1090
  source "$profile_file"
  set +a

  export RESON_SANDBOX_INTEGRATION_PROFILE="${RESON_SANDBOX_INTEGRATION_PROFILE:-$profile}"
  export RESON_SANDBOX_INTEGRATION_PROJECT="${RESON_SANDBOX_INTEGRATION_PROJECT:-reson-sandbox-int-${profile}}"
  export RESON_SANDBOX_INTEGRATION_COMPOSE_FILE="${RESON_SANDBOX_INTEGRATION_COMPOSE_FILE:-deploy/integration/docker-compose.control-plane.yml}"
  export RESON_SANDBOX_INTEGRATION_CLUSTER_TOKEN="${RESON_SANDBOX_INTEGRATION_CLUSTER_TOKEN:-reson-sandbox-integration-${profile}}"
  export RESON_SANDBOX_INTEGRATION_ETCD_PORT="${RESON_SANDBOX_INTEGRATION_ETCD_PORT:-32379}"
  export RESON_SANDBOX_INTEGRATION_NATS_PORT="${RESON_SANDBOX_INTEGRATION_NATS_PORT:-14222}"
  export RESON_SANDBOX_INTEGRATION_NATS_MONITOR_PORT="${RESON_SANDBOX_INTEGRATION_NATS_MONITOR_PORT:-18222}"
  export RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS="${RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS:-http://127.0.0.1:${RESON_SANDBOX_INTEGRATION_ETCD_PORT}}"
  export RESON_SANDBOX_INTEGRATION_NATS_URL="${RESON_SANDBOX_INTEGRATION_NATS_URL:-nats://127.0.0.1:${RESON_SANDBOX_INTEGRATION_NATS_PORT}}"
  export RESON_SANDBOX_INTEGRATION_ARTIFACT_DIR="${RESON_SANDBOX_INTEGRATION_ARTIFACT_DIR:-$REPO_ROOT/.integration-artifacts/$profile}"
}

compose_file_path() {
  abs_path "$RESON_SANDBOX_INTEGRATION_COMPOSE_FILE"
}

run_compose() {
  local compose_bin
  compose_bin="$(resolve_compose_bin)"
  local compose_file
  compose_file="$(compose_file_path)"
  local project
  project="$RESON_SANDBOX_INTEGRATION_PROJECT"

  if [[ "$compose_bin" == "docker compose" ]]; then
    docker compose -f "$compose_file" -p "$project" "$@"
  else
    docker-compose -f "$compose_file" -p "$project" "$@"
  fi
}

utc_timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

artifact_bundle_dir() {
  local suffix
  suffix="$(date -u +"%Y%m%dT%H%M%SZ")"
  printf '%s\n' "$RESON_SANDBOX_INTEGRATION_ARTIFACT_DIR/$suffix"
}

wait_for_http() {
  local url="$1"
  local timeout_seconds="${2:-60}"
  local label="${3:-$url}"
  local elapsed=0

  require_cmd curl

  while (( elapsed < timeout_seconds )); do
    if curl --silent --show-error --fail --max-time 2 "$url" >/dev/null 2>&1; then
      integration_log "ready: $label"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  integration_err "timed out waiting for $label (${timeout_seconds}s): $url"
  return 1
}
