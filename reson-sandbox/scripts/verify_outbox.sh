#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_outbox contract coverage.
# @dive-rel: Invoked by scripts/verify_reson_sandbox.sh gate orchestration and/or Makefile targets.
# @dive-rel: Uses scripts/common.sh helpers for strict-mode behavior, diagnostics, and command validation.

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run outbox gate"
    exit 1
  fi
  warn "runtime sources missing; skipping outbox gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "outbox gate: static outbox + replay wiring checks"
rg -n "OutboxRecord" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null
rg -n "put_outbox_record|delete_outbox_record|publish_outbox_record" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null
rg -n "replay_outbox_once|spawn_outbox_replay_worker" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null
rg -n "publish_command\\(" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null

log "outbox gate: compile checks"
(cd "$REPO_ROOT" && cargo check -p reson-sandbox --locked)

log "outbox gate: passed"
