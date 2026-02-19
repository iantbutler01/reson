#!/usr/bin/env bash
# @dive-file: Resets integration environment state by purging compose resources and optionally bringing stack back up.
# @dive-rel: Combines down --purge semantics with up semantics for deterministic clean-start integration runs.
# @dive-rel: Preserves profile-based endpoint conventions from scripts/integration/common.sh templates.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
NO_UP=0

usage() {
  cat <<'USAGE'
Usage: reset.sh [--profile <name>] [--profile-file <path>] [--no-up]
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
    --no-up)
      NO_UP=1
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

integration_log "resetting integration stack (profile=${RESON_SANDBOX_INTEGRATION_PROFILE})"
run_compose down --remove-orphans --volumes || true

if [[ "$NO_UP" -eq 0 ]]; then
  run_compose up -d
  integration_log "reset complete: stack is up"
else
  integration_log "reset complete: stack is down"
fi
