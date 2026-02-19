#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run MQ control gate"
    exit 1
  fi
  warn "runtime sources missing; skipping MQ control gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

log "mq control gate: static contract checks"
rg -n "publish_command\\(" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null
rg -n "start_with_trigger\\(" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "ensure_control_stream\\(" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "get_or_create_stream\\(" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "get_or_create_consumer\\(" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "AckPolicy::Explicit" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "dead_letter_subject|replay_subject|replay_dead_letters" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
rg -n "idempotency_key|command_id" "$REPO_ROOT/vmd/src/control_bus.rs" >/dev/null
[[ -f "$REPO_ROOT/vmd/src/bin/mqctl.rs" ]]
[[ -x "$REPO_ROOT/scripts/replay_mq_dead_letters.sh" ]]

log "mq control gate: compile checks"
(cd "$REPO_ROOT" && cargo check -p reson-sandbox -p vmd --bin mqctl --locked)

log "mq control gate: passed"
