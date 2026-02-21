#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
if [[ "${1:-}" == "--strict" ]]; then
  STRICT=1
fi

if ! have_runtime_sources; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode cannot run fork CoW gate"
    exit 1
  fi
  warn "runtime sources missing; skipping fork CoW gate"
  exit 0
fi

require_cmd rg

log "fork CoW gate: static contract checks"
rg -n "rpc ForkVM\(" "$REPO_ROOT/proto/bracket/vmd/v1/vmd.proto" >/dev/null
rg -n "async fn fork_vm\(" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "pub async fn fork_vm\(" "$REPO_ROOT/vmd/src/state/manager.rs" >/dev/null

if sed -n '/pub async fn fork_vm(/,/pub async fn restore_snapshot(/p' "$REPO_ROOT/vmd/src/state/manager.rs" | rg -n "copy_file" >/dev/null; then
  err "fork implementation still references full disk copy path"
  exit 1
fi
if ! sed -n '/pub async fn fork_vm(/,/pub async fn restore_snapshot(/p' "$REPO_ROOT/vmd/src/state/manager.rs" | rg -n "clone_file_cow" >/dev/null; then
  err "running fork path must enforce CoW clone semantics (clone_file_cow missing)"
  exit 1
fi
rg -n "max_fork_chain_depth|fork_compaction_depth_threshold|META_FORK_DEPTH|enqueue_fork_compaction_task|garbage_collect_orphaned_fork_roots" "$REPO_ROOT/vmd/src/state/manager.rs" "$REPO_ROOT/vmd/src/config.rs" >/dev/null

require_cmd cargo
log "fork CoW gate: stopped-parent CoW runtime test"
(cd "$REPO_ROOT" && cargo test -p vmd fork_vm_stopped_parent_uses_shared_cow_backing)
(cd "$REPO_ROOT" && cargo test -p vmd fork_vm_rejects_when_chain_depth_limit_exceeded -- --nocapture)

if [[ "${RESON_SANDBOX_SKIP_RUNNING_FORK_RUNTIME:-0}" == "1" ]]; then
  # @dive: strict-real preflight can skip this flaky local runtime sub-check because real gate 42 exercises running-parent fork rehydrate on live failover machinery.
  warn "running-parent fork runtime sub-check skipped (RESON_SANDBOX_SKIP_RUNNING_FORK_RUNTIME=1); real gate 42 provides coverage"
  log "fork CoW gate: passed (static + stopped-parent runtime checks)"
  exit 0
fi

DEFAULT_SOURCE_REF="${VMD_SMOKE_DEFAULT_SOURCE_REF:-ghcr.io/bracketdevelopers/uv-builder:main}"
SOURCE_REF="${VMD_SMOKE_SOURCE_REF:-$DEFAULT_SOURCE_REF}"
if [[ -z "$SOURCE_REF" ]]; then
  if [[ "$STRICT" -eq 1 ]]; then
    err "VMD_SMOKE_SOURCE_REF is required for strict fork CoW verification (or set VMD_SMOKE_DEFAULT_SOURCE_REF)"
    exit 1
  fi
  warn "no smoke source ref configured; static checks only"
  log "fork CoW gate: passed (static only)"
  exit 0
fi

require_cmd qemu-img

extract_json_field() {
  local payload="$1"
  local jq_expr="$2"
  if command -v jq >/dev/null 2>&1; then
    printf '%s\n' "$payload" | jq -r "$jq_expr // empty"
    return 0
  fi
  case "$jq_expr" in
    ".id")
      printf '%s\n' "$payload" | tr -d '\n' | sed -n 's/.*"id"[[:space:]]*:[[:space:]]*"\([^"]\+\)".*/\1/p'
      ;;
    ".child_vm.id")
      printf '%s\n' "$payload" | tr -d '\n' | sed -n 's/.*"child_vm"[[:space:]]*:[[:space:]]*{[^}]*"id"[[:space:]]*:[[:space:]]*"\([^"]\+\)".*/\1/p'
      ;;
    ".fork_id")
      printf '%s\n' "$payload" | tr -d '\n' | sed -n 's/.*"fork_id"[[:space:]]*:[[:space:]]*"\([^"]\+\)".*/\1/p'
      ;;
    *)
      echo ""
      ;;
  esac
}

extract_metadata_field() {
  local file_path="$1"
  local jq_expr="$2"
  if command -v jq >/dev/null 2>&1; then
    jq -r "$jq_expr // empty" "$file_path"
    return 0
  fi
  case "$jq_expr" in
    ".boot_snapshot")
      tr -d '\n' <"$file_path" | sed -n 's/.*"boot_snapshot"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p'
      ;;
    *)
      echo ""
      ;;
  esac
}

backing_file() {
  local disk="$1"
  local out
  out="$(qemu-img info -U "$disk")"
  printf '%s\n' "$out" \
    | sed -n 's/^backing file:[[:space:]]*//p' \
    | sed 's/[[:space:]]*(actual.*$//' \
    | head -n 1 \
    | sed 's/[[:space:]]*$//'
}

log "fork CoW gate: building vmd/vmdctl"
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
[[ -x "$VMD_BIN" ]] || { err "vmd binary missing: $VMD_BIN"; exit 1; }
[[ -x "$VMDCTL_BIN" ]] || { err "vmdctl binary missing: $VMDCTL_BIN"; exit 1; }

SERVER_URL="${VMD_SMOKE_SERVER:-http://127.0.0.1:18053}"
LISTEN_ADDR="${VMD_SMOKE_LISTEN_ADDR:-${SERVER_URL#http://}}"
LISTEN_ADDR="${LISTEN_ADDR#https://}"
TIMEOUT_SECS="${VMD_SMOKE_TIMEOUT_SECS:-900}"
STOP_TIMEOUT_SECS="${VMD_SMOKE_STOP_TIMEOUT_SECS:-45}"

TMP_DIR="$(mktemp -d "/tmp/rsbfk.XXXXXX")"
DATA_DIR="$TMP_DIR/vmd-data"
LOG_FILE="$TMP_DIR/vmd.log"
PARENT_VM_ID=""
CHILD_VM_ID=""
VMD_PID=""
PROXY_BIN_TMP=""

cleanup() {
  set +e
  if [[ -n "$CHILD_VM_ID" ]]; then
    "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" delete-vm "$CHILD_VM_ID" --purge-snapshots >/dev/null 2>&1 || true
  fi
  if [[ -n "$PARENT_VM_ID" ]]; then
    "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" delete-vm "$PARENT_VM_ID" --purge-snapshots >/dev/null 2>&1 || true
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
    warn "portproxy/bin artifacts missing; using fallback PROXY_BIN stubs for fork gate"
  fi
fi

log "fork CoW gate: starting vmd on $LISTEN_ADDR"
"$VMD_BIN" --listen "$LISTEN_ADDR" --data-dir "$DATA_DIR" >"$LOG_FILE" 2>&1 &
VMD_PID="$!"

ready=0
for _ in $(seq 1 60); do
  if ! kill -0 "$VMD_PID" >/dev/null 2>&1; then
    err "vmd exited before becoming ready"
    tail -n 100 "$LOG_FILE" >&2 || true
    exit 1
  fi
  if "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 list-vms >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  err "vmd did not become ready"
  tail -n 100 "$LOG_FILE" >&2 || true
  exit 1
fi

VM_NAME="reson-fork-parent-$(date +%s)"
log "fork CoW gate: creating running parent VM"
log "fork CoW gate: using source ref $SOURCE_REF"
CREATE_OUT="$($VMDCTL_BIN --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" create-vm --source-ref "$SOURCE_REF" --source-type docker --name "$VM_NAME" --auto-start --json 2>"$TMP_DIR/create.stderr")" || {
  err "create-vm failed"
  cat "$TMP_DIR/create.stderr" >&2 || true
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 1
}
PARENT_VM_ID="$(extract_json_field "$CREATE_OUT" '.id')"
[[ -n "$PARENT_VM_ID" ]] || { err "unable to parse parent VM id"; exit 1; }

log "fork CoW gate: forking running parent"
FORK_OUT="$($VMDCTL_BIN --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" fork-vm "$PARENT_VM_ID" --child-name "reson-fork-child" --json 2>"$TMP_DIR/fork.stderr")" || {
  err "fork-vm failed"
  cat "$TMP_DIR/fork.stderr" >&2 || true
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 1
}
CHILD_VM_ID="$(extract_json_field "$FORK_OUT" '.child_vm.id')"
FORK_ID="$(extract_json_field "$FORK_OUT" '.fork_id')"
[[ -n "$CHILD_VM_ID" ]] || { err "unable to parse child VM id"; exit 1; }
[[ -n "$FORK_ID" ]] || { err "unable to parse fork id"; exit 1; }

PARENT_DISK="$DATA_DIR/$PARENT_VM_ID/disk.qcow2"
CHILD_DISK="$DATA_DIR/$CHILD_VM_ID/disk.qcow2"
[[ -f "$PARENT_DISK" ]] || { err "parent disk not found: $PARENT_DISK"; exit 1; }
[[ -f "$CHILD_DISK" ]] || { err "child disk not found: $CHILD_DISK"; exit 1; }

CHILD_META_FILE="$DATA_DIR/$CHILD_VM_ID/vm.json"
[[ -f "$CHILD_META_FILE" ]] || { err "child metadata file missing: $CHILD_META_FILE"; exit 1; }
CHILD_BOOT_SNAPSHOT="$(extract_metadata_field "$CHILD_META_FILE" '.boot_snapshot')"
if [[ -z "$CHILD_BOOT_SNAPSHOT" ]]; then
  err "running fork child must carry a boot_snapshot pointer"
  exit 1
fi

if ! qemu-img snapshot -l "$CHILD_DISK" | rg -F "$CHILD_BOOT_SNAPSHOT" >/dev/null; then
  err "child disk is missing fork snapshot '$CHILD_BOOT_SNAPSHOT'"
  exit 1
fi

log "fork CoW gate: starting child VM from captured snapshot"
"$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$TIMEOUT_SECS" start-vm "$CHILD_VM_ID" >/dev/null
if ! "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$STOP_TIMEOUT_SECS" stop-vm "$CHILD_VM_ID" >/dev/null 2>"$TMP_DIR/child-stop.stderr"; then
  warn "child stop-vm failed; forcing stop"
  "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs "$STOP_TIMEOUT_SECS" force-stop-vm "$CHILD_VM_ID" >/dev/null
fi

if [[ ! -s "$CHILD_DISK" ]]; then
  err "child disk appears empty after fork: $CHILD_DISK"
  exit 1
fi

log "fork CoW gate: passed"
