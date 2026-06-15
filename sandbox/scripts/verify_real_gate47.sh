#!/usr/bin/env bash
# @dive-file: Real gate 47 wrapper for leak/orphan-process checks after distributed failover drills.
# @dive-rel: Runs distributed failover selector under scripts/verify_no_leaks.sh with optional vmd delta checks enabled.
# @dive-rel: Provides real L2/L3 leak evidence for qemu/portproxy/vmd process lifecycles.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

for arg in "$@"; do
  if [[ "$arg" == "--distributed" ]]; then
    err "verify_real_gate47.sh sets --distributed internally; remove explicit --distributed"
    exit 2
  fi
done

GRACE_SECONDS="${REAL_GATE47_LEAK_GRACE_SECONDS:-6}"
SELECTOR="${REAL_GATE47_SELECTOR:-distributed_failover_rebind_updates_route_and_emits_events}"
ITERATIONS="${REAL_GATE47_ITERATIONS:-3}"
RSS_DELTA_KB_LIMIT="${REAL_GATE47_RSS_DELTA_KB_LIMIT:-262144}"

collect_total_rss_kb() {
  local pattern="$1"
  local pids
  local total=0
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "0"
    return
  fi
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    local rss
    rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    [[ -n "$rss" ]] || rss=0
    total=$((total + rss))
  done <<<"$pids"
  echo "$total"
}

log "real gate 47: leak/orphan process check after distributed failover selector=${SELECTOR} iterations=${ITERATIONS}"

rss_pattern='qemu-system|(^|/)portproxy($|[-_ ])|portproxy-|(^|/)vmd($| )| vmd --listen '
rss_before_kb="$(collect_total_rss_kb "$rss_pattern")"
log "real gate 47: rss baseline (kb)=${rss_before_kb} limit_delta_kb=${RSS_DELTA_KB_LIMIT}"

for ((iteration=1; iteration<=ITERATIONS; iteration++)); do
  log "real gate 47: iteration ${iteration}/${ITERATIONS}"
  LEAK_CHECK_INCLUDE_VMD=1 \
    "$REPO_ROOT/scripts/verify_no_leaks.sh" --grace-seconds "$GRACE_SECONDS" -- \
    "$REPO_ROOT/scripts/integration/verify_real_failover.sh" "$@" --distributed --selector "$SELECTOR"
done

rss_after_kb="$(collect_total_rss_kb "$rss_pattern")"
rss_delta_kb=$((rss_after_kb - rss_before_kb))
log "real gate 47: rss after (kb)=${rss_after_kb} delta_kb=${rss_delta_kb}"

# @dive: Churn guard enforces bounded process RSS growth across repeated real failover cycles to catch creeping resource retention.
if (( rss_delta_kb > RSS_DELTA_KB_LIMIT )); then
  err "real gate 47 rss growth exceeded bound: before=${rss_before_kb}kb after=${rss_after_kb}kb delta=${rss_delta_kb}kb limit=${RSS_DELTA_KB_LIMIT}kb"
  exit 1
fi

log "real gate 47: bounded rss growth verified"
