#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_security_profile contract coverage.
# @dive-rel: Invoked by scripts/verify_reson_sandbox.sh gate orchestration and/or Makefile targets.
# @dive-rel: Uses scripts/common.sh helpers for strict-mode behavior, diagnostics, and command validation.

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0

usage() {
  cat <<'USAGE'
Usage: verify_security_profile.sh [--strict]
USAGE
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
    err "runtime sources missing; strict mode cannot run security gate"
    exit 1
  fi
  warn "runtime sources missing; skipping security gate"
  exit 0
fi

require_cmd cargo
require_cmd rg

log "security gate: static authn/authz + tls contract checks"
rg -n "AccessLevel|authorize\\(|extract_bearer_token|authorize_metadata" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "ServerTlsConfig|load_server_tls_config|client_auth_optional" "$REPO_ROOT/vmd/src/app.rs" >/dev/null
rg -n "AuthConfig|SecurityConfig|TlsServerConfig" "$REPO_ROOT/vmd/src/config.rs" >/dev/null
rg -n "auth_token|tls_cert|tls_key|readonly_auth_token" "$REPO_ROOT/vmd/src/bin/vmd.rs" >/dev/null
rg -n "request_with_auth|compile_auth_header|ClientTlsConfig|build_client_tls_config" "$REPO_ROOT/crates/reson-sandbox/src/lib.rs" >/dev/null
rg -n "auth_token|auth_token_file|request_with_auth" "$REPO_ROOT/vmd/src/bin/vmdctl.rs" >/dev/null

log "security gate: compile checks"
(cd "$REPO_ROOT" && cargo check -p vmd -p reson-sandbox --locked)

log "security gate: unit tests"
(cd "$REPO_ROOT" && cargo test -p vmd app::tests::authorize_metadata -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox compile_auth_header -- --nocapture)

log "security gate: runtime auth smoke"
(cd "$REPO_ROOT/vmd" && cargo build --bin vmd --bin vmdctl --locked)

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

SERVER_URL="${VMD_SECURITY_SERVER:-http://127.0.0.1:18063}"
LISTEN_ADDR="${VMD_SECURITY_LISTEN_ADDR:-${SERVER_URL#http://}}"
LISTEN_ADDR="${LISTEN_ADDR#https://}"
AUTH_TOKEN="${VMD_SECURITY_AUTH_TOKEN:-security-admin-token}"
READONLY_TOKEN="${VMD_SECURITY_READONLY_TOKEN:-security-readonly-token}"

TMP_DIR="$(mktemp -d "/tmp/rsbsec.XXXXXX")"
DATA_DIR="$TMP_DIR/vmd-data"
LOG_FILE="$TMP_DIR/vmd.log"
VMD_PID=""

cleanup() {
  set +e
  if [[ -n "$VMD_PID" ]] && kill -0 "$VMD_PID" >/dev/null 2>&1; then
    kill "$VMD_PID" >/dev/null 2>&1 || true
    wait "$VMD_PID" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$DATA_DIR"

wait_ready_with_admin() {
  for _ in $(seq 1 60); do
    if [[ -n "$VMD_PID" ]] && ! kill -0 "$VMD_PID" >/dev/null 2>&1; then
      err "vmd exited before ready"
      tail -n 100 "$LOG_FILE" >&2 || true
      return 1
    fi
    if "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 --auth-token "$AUTH_TOKEN" list-vms >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

"$VMD_BIN" \
  --listen "$LISTEN_ADDR" \
  --data-dir "$DATA_DIR" \
  --auth-token "$AUTH_TOKEN" \
  --readonly-auth-token "$READONLY_TOKEN" \
  >"$LOG_FILE" 2>&1 &
VMD_PID="$!"

wait_ready_with_admin || {
  err "vmd with auth profile did not become ready"
  tail -n 120 "$LOG_FILE" >&2 || true
  exit 1
}

if "$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 list-vms >/dev/null 2>&1; then
  err "unauthenticated request unexpectedly succeeded"
  exit 1
fi

"$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 --auth-token "$AUTH_TOKEN" list-vms >/dev/null
"$VMDCTL_BIN" --server "$SERVER_URL" --timeout-secs 5 --auth-token "$READONLY_TOKEN" list-vms >/dev/null

log "security gate: passed"
