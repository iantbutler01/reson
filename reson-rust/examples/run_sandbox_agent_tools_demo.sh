#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SANDBOX_ROOT="$(cd "$ROOT/../reson-sandbox" && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required for guest portproxy binary build" >&2
  exit 1
fi

# Build guest linux portproxy binary for this host arch (arm64 on Apple Silicon).
"$SANDBOX_ROOT/scripts/build_portproxy_guest_bins_docker.sh"

# Build vmd binary.
(cd "$SANDBOX_ROOT" && cargo build -p vmd --bin vmd)

# Resolve vmd path for workspace target layout.
VMD_BIN_DEFAULT="$SANDBOX_ROOT/target/debug/vmd"
if [[ ! -x "$VMD_BIN_DEFAULT" ]]; then
  VMD_BIN_DEFAULT="$SANDBOX_ROOT/vmd/target/debug/vmd"
fi

export RESON_SANDBOX_MODE="${RESON_SANDBOX_MODE:-local}"
export RESON_SANDBOX_IMAGE="${RESON_SANDBOX_IMAGE:-ghcr.io/bracketdevelopers/uv-builder:main}"
export RESON_SANDBOX_DATA_DIR="${RESON_SANDBOX_DATA_DIR:-/tmp/reson-agent-tools-vmd}"
export RESON_VMD_BIN="${RESON_VMD_BIN:-$VMD_BIN_DEFAULT}"
export RESON_PROXY_BIN_DIR="${RESON_PROXY_BIN_DIR:-$SANDBOX_ROOT/portproxy/bin}"
export RESON_EXEC_READY_TIMEOUT_SECS="${RESON_EXEC_READY_TIMEOUT_SECS:-240}"
export RESON_AGENT_MODEL="${RESON_AGENT_MODEL:-${LOCAL_LLM:-}}"

if [[ -z "${RESON_AGENT_MODEL}" ]]; then
  cat >&2 <<'ERR'
missing model config:
  set RESON_AGENT_MODEL (preferred) or LOCAL_LLM

examples:
  export RESON_AGENT_MODEL='openrouter:anthropic/claude-sonnet-4'
  export OPENROUTER_API_KEY='...'

or for a local backend:
  export LOCAL_LLM='custom-openai:your-model@server_url=http://localhost:8000/v1'
  export OPENAI_API_KEY='not-needed'
ERR
  exit 1
fi

cd "$ROOT"
cargo run --features sandbox --example sandbox_agent_tools_demo
