#!/usr/bin/env bash
# @dive-file: Shared verifier helper library for logging, repo path resolution, and command checks.
# @dive-rel: Sourced by scripts/verify_*.sh and operational drill scripts.
# @dive-rel: Provides consistent strict-gate behavior and runtime source availability checks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

log() {
  printf '[verify] %s\n' "$*"
}

warn() {
  printf '[verify][warn] %s\n' "$*" >&2
}

err() {
  printf '[verify][error] %s\n' "$*" >&2
}

have_runtime_sources() {
  [[ -f "$REPO_ROOT/vmd/Cargo.toml" ]] && \
  [[ -f "$REPO_ROOT/portproxy/Cargo.toml" ]] && \
  [[ -f "$REPO_ROOT/proto/bracket/vmd/v1/vmd.proto" ]] && \
  [[ -f "$REPO_ROOT/proto/bracket/portproxy/v1/portproxy.proto" ]]
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || {
    err "required command not found: $cmd"
    return 1
  }
}
