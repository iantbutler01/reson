#!/usr/bin/env bash
# @dive-file: Tears down canonical integration control-plane services for a selected profile.
# @dive-rel: Uses scripts/integration/common.sh compose helpers and profile defaults.
# @dive-rel: Pairs with scripts/integration/up.sh and scripts/integration/reset.sh lifecycle actions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
PURGE=0

usage() {
  cat <<'USAGE'
Usage: down.sh [--profile <name>] [--profile-file <path>] [--purge]
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
    --purge)
      PURGE=1
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

require_cmd docker
load_profile_env "$PROFILE" "$PROFILE_FILE"

integration_log "stopping integration control plane (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
if [[ "$PURGE" -eq 1 ]]; then
  run_compose down --remove-orphans --volumes
else
  run_compose down --remove-orphans
fi
integration_log "integration stack is down"
