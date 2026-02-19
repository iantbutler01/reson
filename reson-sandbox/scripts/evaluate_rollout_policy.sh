#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

THRESHOLDS_FILE="$REPO_ROOT/specs/RESON_SANDBOX_SLO_THRESHOLDS.json"
OBSERVED_FILE="${RESON_SANDBOX_ROLLOUT_OBSERVED_FILE:-}"

usage() {
  cat <<'USAGE'
Usage: evaluate_rollout_policy.sh [--thresholds <path>] [--observed <path>]

Evaluates rollout pause/rollback policy from error-budget burn metrics.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --thresholds)
      THRESHOLDS_FILE="${2:-}"
      shift 2
      ;;
    --observed)
      OBSERVED_FILE="${2:-}"
      shift 2
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

require_cmd jq

[[ -f "$THRESHOLDS_FILE" ]] || {
  err "missing thresholds file: $THRESHOLDS_FILE"
  exit 1
}
[[ -n "$OBSERVED_FILE" ]] || {
  err "observed rollout file is required (--observed or RESON_SANDBOX_ROLLOUT_OBSERVED_FILE)"
  exit 1
}
[[ -f "$OBSERVED_FILE" ]] || {
  err "missing observed rollout file: $OBSERVED_FILE"
  exit 1
}

pause_threshold="$(jq -r '.error_budget.projected_burn_percent_pause_threshold // empty' "$THRESHOLDS_FILE")"
rollback_threshold="$(jq -r '.error_budget.projected_burn_percent_rollback_threshold // empty' "$THRESHOLDS_FILE")"
window_days="$(jq -r '.error_budget.window_days // empty' "$THRESHOLDS_FILE")"
projected_burn="$(jq -r '.error_budget.projected_burn_percent // empty' "$OBSERVED_FILE")"

if [[ -z "$pause_threshold" || -z "$rollback_threshold" || -z "$window_days" || -z "$projected_burn" ]]; then
  err "rollout policy evaluation missing required fields"
  exit 1
fi

pause_rollout=0
rollback_required=0
if (( projected_burn > pause_threshold )); then
  pause_rollout=1
fi
if (( projected_burn >= rollback_threshold )); then
  rollback_required=1
fi

echo "window_days=$window_days"
echo "projected_burn_percent=$projected_burn"
echo "pause_threshold_percent=$pause_threshold"
echo "rollback_threshold_percent=$rollback_threshold"
echo "pause_rollout=$pause_rollout"
echo "rollback_required=$rollback_required"

if (( rollback_required == 1 )); then
  exit 1
fi
