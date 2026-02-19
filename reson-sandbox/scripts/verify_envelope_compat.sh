#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run envelope-compat gate"
    exit 1
  fi
  warn "runtime sources missing; skipping envelope-compat gate"
  exit 0
fi

require_cmd rg
require_cmd cargo

COMMAND_SCHEMA="$REPO_ROOT/specs/schemas/control_command_envelope.v1.json"
EVENT_SCHEMA="$REPO_ROOT/specs/schemas/control_event_envelope.v1.json"
POLICY_DOC="$REPO_ROOT/specs/RESON_SANDBOX_ENVELOPE_COMPATIBILITY_POLICY.md"

log "envelope-compat gate: schema + policy contract checks"
for required in "$COMMAND_SCHEMA" "$EVENT_SCHEMA" "$POLICY_DOC"; do
  [[ -f "$required" ]] || {
    err "missing envelope artifact: ${required#$REPO_ROOT/}"
    exit 1
  }
done

rg -n "\"schema_version\"|\"const\": \"v1\"" "$COMMAND_SCHEMA" "$EVENT_SCHEMA" >/dev/null
rg -n "N and N-1|breaking change|Gate 21" "$POLICY_DOC" >/dev/null
rg -n "CONTROL_ENVELOPE_SCHEMA_VERSION|build_command_envelope|build_event_envelope" "$REPO_ROOT/crates/reson-sandbox/src/distributed.rs" >/dev/null

log "envelope-compat gate: envelope unit tests"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox command_envelope_contains_required_fields_and_version -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox event_envelope_contains_required_fields_and_version -- --nocapture)

log "envelope-compat gate: passed"
