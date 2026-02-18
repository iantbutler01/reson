#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if [[ ! -f "$REPO_ROOT/crates/reson-sandbox/Cargo.toml" ]]; then
  err "facade crate missing: crates/reson-sandbox/Cargo.toml"
  exit 1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run facade gate"
    exit 1
  fi
  warn "runtime sources missing; skipping facade gate"
  exit 0
fi

require_cmd cargo

log "facade gate: cargo check -p reson-sandbox"
(cd "$REPO_ROOT" && cargo check -p reson-sandbox)

log "facade gate: cargo test -p reson-sandbox"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox)

log "facade gate: behavioral contract tests"
(cd "$REPO_ROOT" && cargo test -p reson-sandbox --test facade_contract)

log "facade gate: passed"
