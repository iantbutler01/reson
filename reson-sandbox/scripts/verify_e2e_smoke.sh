#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0

usage() {
  cat <<'EOF'
Usage: verify_e2e_smoke.sh [--strict]

Required env to execute smoke:
  VMD_SMOKE_SOURCE_REF   Docker image reference for VM creation

Optional env:
  VMD_SMOKE_SERVER       gRPC URL for vmdctl (default: http://127.0.0.1:18052)
  VMD_SMOKE_LISTEN_ADDR  Listen addr for vmd (default: derived from VMD_SMOKE_SERVER)
  VMD_SMOKE_TIMEOUT_SECS Request timeout for vmdctl (default: 900)
  VMD_SMOKE_ARCH         VM architecture override (amd64/arm64)
  VMD_SMOKE_VCPU         VM vCPU count (default: 1)
  VMD_SMOKE_MEMORY_MB    VM memory MB (default: 1024)
  VMD_SMOKE_DISK_GB      VM disk GB (default: 10)
  VMD_BIN                vmd binary path override
  VMDCTL_BIN             vmdctl binary path override
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run e2e smoke"
    exit 1
  fi
  warn "runtime sources missing; skipping e2e smoke gate"
  exit 0
fi

require_cmd cargo

SOURCE_REF="${VMD_SMOKE_SOURCE_REF:-}"
if [[ -z "$SOURCE_REF" ]]; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "VMD_SMOKE_SOURCE_REF is required in strict mode"
    exit 1
  fi
  warn "VMD_SMOKE_SOURCE_REF not set; skipping e2e smoke gate"
  exit 0
fi

extract_vm_id() {
  local payload="$1"
  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$payload" | jq -r '.id // empty'
    return 0
  fi
  printf '%s\n' "$payload" | tr -d '\n' | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]\+\)".*/\1/p'
}

SERVER_URL="${VMD_SMOKE_SERVER:-http://127.0.0.1:18052}"
LISTEN_ADDR="${VMD_SMOKE_LISTEN_ADDR:-${SERVER_URL#http://}}"
LISTEN_ADDR="${LISTEN_ADDR#https://}"
TIMEOUT_SECS="${VMD_SMOKE_TIMEOUT_SECS:-900}"
SMOKE_ARCH="${VMD_SMOKE_ARCH:-}"
SMOKE_VCPU="${VMD_SMOKE_VCPU:-1}"
SMOKE_MEMORY_MB="${VMD_SMOKE_MEMORY_MB:-1024}"
SMOKE_DISK_GB="${VMD_SMOKE_DISK_GB:-10}"

log "building vmd binaries for smoke gate"
(cd "$REPO_ROOT/vmd" && cargo build --bin vmd --bin vmdctl)

default_vmd_bin="$REPO_ROOT/vmd/target/debug/vmd"
default_vmdctl_bin="$REPO_ROOT/vmd/target/debug/vmdctl"
if [[ ! -x "$default_vmd_bin" && -x "$REPO_ROOT/target/debug/vmd" ]]; then
  default_vmd_bin="$REPO_ROOT/target/debug/vmd"
fi
if [[ ! -x "$default_vmdctl_bin" && -x "$REPO_ROOT/target/debug/vmdctl" ]]; then
  default_vmdctl_bin="$REPO_ROOT/target/debug/vmdctl"
fi

VMD_BIN="${VMD_BIN:-$default_vmd_bin}"
VMDCTL_BIN="${VMDCTL_BIN:-$default_vmdctl_bin}"

[[ -x "$VMD_BIN" ]] || { err "vmd binary not found or not executable: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { err "vmdctl binary not found or not executable: $VMDCTL_BIN"; exit 1; }

TMP_DIR="$(mktemp -d "/tmp/rsbsmk.XXXXXX")"
DATA_DIR="$TMP_DIR/vmd-data"
LOG_FILE="$TMP_DIR/vmd.log"
VM_ID=""
VMD_PID=""
PROXY_BIN_TMP=""

cleanup() {
  set +e
  if [[ -n "$VM_ID" ]]; then
    "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" delete-vm "$VM_ID" --purge-snapshots >/dev/null 2>&1 || true
  fi
  if [[ -n "$VMD_PID" ]] && kill -0 "$VMD_PID" >/dev/null 2>&1; then
    kill "$VMD_PID" >/dev/null 2>&1 || true
    wait "$VMD_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$DATA_DIR"

if [[ -z "${PROXY_BIN:-}" ]]; then
  default_proxy_dir="$REPO_ROOT/portproxy/bin"
  if [[ ! -f "$default_proxy_dir/portproxy-linux-amd64" || ! -f "$default_proxy_dir/portproxy-linux-arm64" ]]; then
    PROXY_BIN_TMP="$TMP_DIR/proxy-bin"
    mkdir -p "$PROXY_BIN_TMP"
    cat >"$PROXY_BIN_TMP/portproxy-linux-amd64" <<'EOF'
#!/bin/sh
while true; do sleep 3600; done
EOF
    cat >"$PROXY_BIN_TMP/portproxy-linux-arm64" <<'EOF'
#!/bin/sh
while true; do sleep 3600; done
EOF
    chmod +x "$PROXY_BIN_TMP/portproxy-linux-amd64" "$PROXY_BIN_TMP/portproxy-linux-arm64"
    export PROXY_BIN="$PROXY_BIN_TMP"
    warn "portproxy/bin artifacts missing; using fallback PROXY_BIN stubs for smoke gate"
  fi
fi

log "starting vmd daemon on $LISTEN_ADDR"
"$VMD_BIN" --listen "$LISTEN_ADDR" --data-dir "$DATA_DIR" >"$LOG_FILE" 2>&1 &
VMD_PID="$!"

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$VMD_PID" >/dev/null 2>&1; then
    err "vmd exited before becoming ready"
    tail -n 80 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 list-vms >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  err "vmd did not become ready at $SERVER_URL"
  tail -n 80 "$LOG_FILE" >&2 || true
  exit 1
fi

VM_NAME="reson-smoke-$(date +%s)"
log "creating VM via vmdctl create-vm (name=$VM_NAME)"
create_cmd=(
  "$VMDCTL_BIN"
  --server "$SERVER_URL"
  --timeout-secs "$TIMEOUT_SECS"
  create-vm
  --source-ref "$SOURCE_REF"
  --source-type docker
  --name "$VM_NAME"
  --vcpu "$SMOKE_VCPU"
  --memory-mb "$SMOKE_MEMORY_MB"
  --disk-gb "$SMOKE_DISK_GB"
  --json
)
if [[ -n "$SMOKE_ARCH" ]]; then
  create_cmd+=(--arch "$SMOKE_ARCH")
fi

if ! CREATE_OUT="$("${create_cmd[@]}" 2>"$TMP_DIR/create.stderr")"; then
  err "create-vm failed"
  cat "$TMP_DIR/create.stderr" >&2 || true
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 1
fi

VM_ID="$(extract_vm_id "$CREATE_OUT")"
if [[ -z "$VM_ID" ]]; then
  err "could not parse VM id from create-vm output"
  printf '%s\n' "$CREATE_OUT" >&2
  exit 1
fi

log "starting VM ($VM_ID)"
"$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" start-vm "$VM_ID" >/dev/null

log "stopping VM ($VM_ID)"
if ! "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" stop-vm "$VM_ID" >/dev/null 2>"$TMP_DIR/stop.stderr"; then
  warn "stop-vm timed out or failed; falling back to force-stop-vm"
  "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" force-stop-vm "$VM_ID" >/dev/null
fi

log "deleting VM ($VM_ID)"
"$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" delete-vm "$VM_ID" --purge-snapshots >/dev/null
VM_ID=""

log "e2e smoke gate passed"
