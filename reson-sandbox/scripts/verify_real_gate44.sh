#!/usr/bin/env bash
# @dive-file: Real gate 44 wrapper for warm-pool/prewarm pipeline evidence.
# @dive-rel: Delegates to scripts/integration/verify_real_warm_pool.sh for real distributed/local warm-pool probes.
# @dive-rel: Accepted profile flags are parsed for orchestration compatibility with scripts/verify_strict_real.sh.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

PROFILE="local-dev"
PROFILE_FILE=""
KEEP_STACK=0

usage() {
  cat <<'USAGE'
Usage: verify_real_gate44.sh [--profile <name>] [--profile-file <path>] [--keep-stack]
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
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

log "real gate 44: warm-pool/prewarm pipeline (profile=${PROFILE})"
if [[ -n "$PROFILE_FILE" ]]; then
  log "real gate 44: profile-file=${PROFILE_FILE}"
fi
args=(--profile "$PROFILE")
if [[ -n "$PROFILE_FILE" ]]; then
  args+=(--profile-file "$PROFILE_FILE")
fi
if [[ "$KEEP_STACK" -eq 1 ]]; then
  args+=(--keep-stack)
fi
exec "$REPO_ROOT/scripts/integration/verify_real_warm_pool.sh" "${args[@]}"
