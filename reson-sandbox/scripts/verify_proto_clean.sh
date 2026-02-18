#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; cannot run proto gate in strict mode"
    exit 1
  fi
  warn "runtime sources missing; skipping proto gate"
  exit 0
fi

log "proto gate: verifying required proto files exist"
[[ -s "$REPO_ROOT/proto/bracket/vmd/v1/vmd.proto" ]]
[[ -s "$REPO_ROOT/proto/bracket/portproxy/v1/portproxy.proto" ]]

# If a proto generation script exists, run it and ensure no generated drift.
if [[ -x "$REPO_ROOT/build_protos.sh" ]]; then
  log "proto gate: running build_protos.sh"
  (cd "$REPO_ROOT" && ./build_protos.sh)
elif [[ -x "$REPO_ROOT/scripts/gen_protos.sh" ]]; then
  log "proto gate: running scripts/gen_protos.sh"
  (cd "$REPO_ROOT" && ./scripts/gen_protos.sh)
else
  warn "no proto generation script found; existence check only"
fi

log "proto gate: passed"
