#!/usr/bin/env bash
# @dive-file: Collects machine-readable integration artifact bundles (logs, compose state, process snapshots).
# @dive-rel: Uses schema contract in specs/schemas/integration_artifact_bundle.v1.json for artifact manifest shape.
# @dive-rel: Intended for CI and local failure triage across real integration gate runs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
OUTPUT_DIR=""
REASON="manual"

usage() {
  cat <<'USAGE'
Usage: collect-artifacts.sh [--profile <name>] [--profile-file <path>] [--output-dir <path>] [--reason <text>]
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
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --reason)
      REASON="$2"
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

bundle_dir="${OUTPUT_DIR:-$(artifact_bundle_dir)}"
mkdir -p "$bundle_dir"
mkdir -p "$bundle_dir/container_inspect"

compose_ps_file="$bundle_dir/compose_ps.txt"
compose_logs_file="$bundle_dir/compose_logs.txt"
docker_ps_file="$bundle_dir/docker_ps.txt"
process_snapshot_file="$bundle_dir/process_snapshot.txt"
profile_env_file="$bundle_dir/profile_env.txt"
manifest_file="$bundle_dir/artifact_bundle.json"

integration_log "collecting integration artifacts (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"

run_compose ps >"$compose_ps_file" 2>&1 || true
run_compose logs --no-color >"$compose_logs_file" 2>&1 || true
docker ps --format '{{.ID}} {{.Image}} {{.Names}} {{.Status}}' >"$docker_ps_file" 2>&1 || true
ps aux | grep -E 'vmd|qemu-system|portproxy|etcd|nats' | grep -v grep >"$process_snapshot_file" 2>&1 || true

env | grep '^RESON_SANDBOX_INTEGRATION_' | sort >"$profile_env_file" 2>&1 || true

container_ids="$(run_compose ps -q 2>/dev/null || true)"
if [[ -n "$container_ids" ]]; then
  while IFS= read -r container_id; do
    [[ -z "$container_id" ]] && continue
    docker inspect "$container_id" >"$bundle_dir/container_inspect/${container_id}.json" 2>&1 || true
  done <<<"$container_ids"
fi

generated_at="$(utc_timestamp)"
compose_file_rel="${RESON_SANDBOX_INTEGRATION_COMPOSE_FILE}"

cat >"$manifest_file" <<EOF_MANIFEST
{
  "schema": "reson.sandbox.integration.artifact_bundle.v1",
  "generated_at_utc": "$generated_at",
  "profile": "${RESON_SANDBOX_INTEGRATION_PROFILE}",
  "reason": "$REASON",
  "project": "${RESON_SANDBOX_INTEGRATION_PROJECT}",
  "compose_file": "$compose_file_rel",
  "control_plane": {
    "etcd_endpoints": "${RESON_SANDBOX_INTEGRATION_ETCD_ENDPOINTS}",
    "nats_url": "${RESON_SANDBOX_INTEGRATION_NATS_URL}"
  },
  "artifacts": {
    "compose_ps": "compose_ps.txt",
    "compose_logs": "compose_logs.txt",
    "docker_ps": "docker_ps.txt",
    "process_snapshot": "process_snapshot.txt",
    "profile_env": "profile_env.txt",
    "container_inspect_dir": "container_inspect"
  }
}
EOF_MANIFEST

integration_log "artifact bundle written: $bundle_dir"
integration_log "manifest: $manifest_file"
