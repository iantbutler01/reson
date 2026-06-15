#!/usr/bin/env bash
# @dive-file: Real gate 45 wrapper for live control-plane fail-closed/recovery drills.
# @dive-rel: Delegates to scripts/integration/verify_control_plane_failures.sh and verify_planned_drain_handoff.sh.
# @dive-rel: Used by scripts/verify_strict_real.sh to enforce real control-plane failure + planned-drain semantics.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

log "real gate 45: control-plane fail-closed + recovery drills (L3)"
"$REPO_ROOT/scripts/integration/verify_control_plane_failures.sh" "$@"
log "real gate 45: planned drain handoff with admission freeze (L3)"
exec "$REPO_ROOT/scripts/integration/verify_planned_drain_handoff.sh" "$@"
