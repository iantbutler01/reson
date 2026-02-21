#!/usr/bin/env bash
# @dive-file: Strict real-orchestration entrypoint combining fast mock preflight with mandatory real gates 41-48.
# @dive-rel: Uses profile-aware integration harness scripts under scripts/integration/ for real control-plane/node-loss evidence.
# @dive-rel: Intended CI/release command for RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md section 7.7 completion.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

PROFILE="local-dev"
PROFILE_FILE=""
KEEP_STACK=0
SKIP_MOCK_PREFLIGHT=0
MOCK_PREFLIGHT_STRICT=0

usage() {
  cat <<'USAGE'
Usage: verify_strict_real.sh [--profile <name>] [--profile-file <path>] [--keep-stack] [--skip-mock-preflight] [--mock-preflight-strict]
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
    --skip-mock-preflight)
      SKIP_MOCK_PREFLIGHT=1
      shift
      ;;
    --mock-preflight-strict)
      MOCK_PREFLIGHT_STRICT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

profile_args=(--profile "$PROFILE")
if [[ -n "$PROFILE_FILE" ]]; then
  profile_args+=(--profile-file "$PROFILE_FILE")
fi
if [[ "$KEEP_STACK" -eq 1 ]]; then
  profile_args+=(--keep-stack)
fi

if [[ "$SKIP_MOCK_PREFLIGHT" -eq 0 ]]; then
  if [[ "$MOCK_PREFLIGHT_STRICT" -eq 1 ]]; then
    log "strict-real: running strict mock preflight"
    RESON_SANDBOX_SKIP_RUNNING_FORK_RUNTIME=1 \
      "$REPO_ROOT/scripts/verify_reson_sandbox.sh" --strict
  else
    log "strict-real: running fast mock preflight"
    RESON_SANDBOX_SKIP_RUNNING_FORK_RUNTIME=1 \
      "$REPO_ROOT/scripts/verify_reson_sandbox.sh"
  fi
else
  warn "strict-real: mock preflight skipped"
fi

log "strict-real: running real gates 41-48 with profile=${PROFILE}"
"$REPO_ROOT/scripts/verify_real_gate41.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate42.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate43.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate44.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate45.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate46.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate47.sh" "${profile_args[@]}"
"$REPO_ROOT/scripts/verify_real_gate48.sh" "${profile_args[@]}"

log "strict-real: all real gates passed"
