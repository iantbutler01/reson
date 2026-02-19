#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run storage-profile gate"
    exit 1
  fi
  warn "runtime sources missing; skipping storage-profile gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "storage-profile gate: static durability contract checks"
rg -n "StorageProfile|RESON_SANDBOX_STORAGE_PROFILE|RESON_SANDBOX_HA_MODE|ha mode requires storage profile" "$REPO_ROOT/vmd/src/config.rs" >/dev/null
rg -n "ContinuityTier|RESON_SANDBOX_CONTINUITY_TIER|RESON_SANDBOX_DEGRADED_MODE|ha mode defaults to continuity tier" "$REPO_ROOT/vmd/src/config.rs" >/dev/null
rg -n "\"storage_profile\"|\"continuity_tier\"|\"degraded_mode\"" "$REPO_ROOT/vmd/src/registry.rs" >/dev/null
rg -n "required_storage_profile|required_continuity_tier|allow_tier_a_degraded|eligible_routes_with_profile" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "META_STORAGE_PROFILE|META_FORK_DURABILITY_CLASS|META_FORK_RESTORE_SCOPE|fork_snapshot_durability" "$REPO_ROOT/vmd/src/state/manager.rs" >/dev/null

log "storage-profile gate: targeted unit tests"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox eligible_routes_filters_by_required_storage_profile -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox eligible_routes_enforces_tier_b_default_and_tier_a_degraded_policy -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p vmd continuity_policy_ -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p vmd fork_snapshot_durability_tracks_profile_and_parent_state -- --nocapture)

log "storage-profile gate: passed"
