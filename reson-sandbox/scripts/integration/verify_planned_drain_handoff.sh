#!/usr/bin/env bash
# @dive-file: Runs real planned-drain handoff test with admission freeze and in-flight continuity guarantees.
# @dive-rel: Wraps scripts/integration/verify_two_node_registry.sh with --run-drain-test and frozen-primary defaults.
# @dive-rel: Provides checklist coverage for RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md item 7.5.3.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PROFILE="$DEFAULT_PROFILE"
PROFILE_FILE=""
KEEP_STACK=0

usage() {
  cat <<'USAGE'
Usage: verify_planned_drain_handoff.sh [--profile <name>] [--profile-file <path>] [--keep-stack]
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

args=(--profile "$PROFILE" --run-drain-test)
if [[ -n "$PROFILE_FILE" ]]; then
  args+=(--profile-file "$PROFILE_FILE")
fi
if [[ "$KEEP_STACK" -eq 1 ]]; then
  args+=(--keep-stack)
fi

integration_log "running planned-drain handoff probe (profile=${PROFILE})"
RESON_SANDBOX_INTEGRATION_NODE1_ADMISSION_FROZEN=1 \
RESON_SANDBOX_INTEGRATION_NODE2_ADMISSION_FROZEN=0 \
  "$SCRIPT_DIR/verify_two_node_registry.sh" "${args[@]}"

