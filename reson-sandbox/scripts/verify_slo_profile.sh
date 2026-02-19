#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
THRESHOLDS_FILE="$REPO_ROOT/specs/RESON_SANDBOX_SLO_THRESHOLDS.json"
OBSERVED_FILE="${RESON_SANDBOX_SLO_OBSERVED_FILE:-}"

usage() {
  cat <<'USAGE'
Usage: verify_slo_profile.sh [--strict]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
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

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run slo gate"
    exit 1
  fi
  warn "runtime sources missing; skipping slo gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

[[ -f "$THRESHOLDS_FILE" ]] || { err "missing thresholds file: $THRESHOLDS_FILE"; exit 1; }

log "slo gate: threshold schema checks"
rg -n "\"session\\.attach\"" "$THRESHOLDS_FILE" >/dev/null
rg -n "\"control\\.command\\.dispatch\"" "$THRESHOLDS_FILE" >/dev/null
rg -n "\"exec\\.stream\\.establish\\.warm_vm\"" "$THRESHOLDS_FILE" >/dev/null
rg -n "\"session\\.create\\.warm_pool\"" "$THRESHOLDS_FILE" >/dev/null
rg -n "\"session\\.create\\.cold_cache_hit\"" "$THRESHOLDS_FILE" >/dev/null

log "slo gate: instrumentation wiring checks"
rg -n "log_slo_observation\\(\"session\\.create\"" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "log_slo_observation\\(\"session\\.attach\"" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "log_slo_observation\\(\"exec\\.stream\\.establish\\.warm_vm\"" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null

log "slo gate: budget evaluator tests"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox slo_budget_ -- --nocapture)

if [[ -n "$OBSERVED_FILE" ]]; then
  require_cmd jq
  [[ -f "$OBSERVED_FILE" ]] || { err "missing observed metrics file: $OBSERVED_FILE"; exit 1; }
  log "slo gate: observed threshold enforcement"
  for metric in \
    "session.attach" \
    "control.command.dispatch" \
    "exec.stream.establish.warm_vm" \
    "session.create.warm_pool" \
    "session.create.cold_cache_hit"; do
    threshold_p95="$(jq -r --arg m "$metric" '.metrics[$m].p95_ms // empty' "$THRESHOLDS_FILE")"
    threshold_p99="$(jq -r --arg m "$metric" '.metrics[$m].p99_ms // empty' "$THRESHOLDS_FILE")"
    observed_p95="$(jq -r --arg m "$metric" '.metrics[$m].p95_ms // empty' "$OBSERVED_FILE")"
    observed_p99="$(jq -r --arg m "$metric" '.metrics[$m].p99_ms // empty' "$OBSERVED_FILE")"

    if [[ -n "$threshold_p95" && -n "$observed_p95" && "$observed_p95" -gt "$threshold_p95" ]]; then
      err "slo breach: $metric p95 observed=${observed_p95}ms threshold=${threshold_p95}ms"
      exit 1
    fi
    if [[ -n "$threshold_p99" && -n "$observed_p99" && "$observed_p99" -gt "$threshold_p99" ]]; then
      err "slo breach: $metric p99 observed=${observed_p99}ms threshold=${threshold_p99}ms"
      exit 1
    fi
  done
else
  warn "RESON_SANDBOX_SLO_OBSERVED_FILE is unset; observed threshold enforcement skipped (expected in local strict runs)"
fi

log "slo gate: passed"
