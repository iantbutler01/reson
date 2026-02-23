#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_security_ops contract coverage.
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
    err "runtime sources missing; strict mode cannot run security-ops gate"
    exit 1
  fi
  warn "runtime sources missing; skipping security-ops gate"
  exit 0
fi

require_cmd rg
require_cmd cargo
require_cmd openssl

RUNBOOK_PATH="$REPO_ROOT/specs/runbooks/SECURITY_CERT_ROTATION_AND_SECRETS_PLAYBOOK.md"
ROTATE_SCRIPT="$REPO_ROOT/scripts/security/rotate_tls_bundle.sh"
REVOKE_SCRIPT="$REPO_ROOT/scripts/security/revoke_client_cert.sh"

log "security-ops gate: playbook + script contract checks"
[[ -f "$RUNBOOK_PATH" ]] || {
  err "missing runbook: ${RUNBOOK_PATH#$REPO_ROOT/}"
  exit 1
}
[[ -x "$ROTATE_SCRIPT" ]] || {
  err "missing executable rotation script: ${ROTATE_SCRIPT#$REPO_ROOT/}"
  exit 1
}
[[ -x "$REVOKE_SCRIPT" ]] || {
  err "missing executable revocation script: ${REVOKE_SCRIPT#$REPO_ROOT/}"
  exit 1
}

rg -n "rotation|revocation|secret|managed secret|quarterly" "$RUNBOOK_PATH" >/dev/null
rg -n "openssl|server.crt|client.crt|revocations.txt" "$ROTATE_SCRIPT" >/dev/null
rg -n "serial|revocations.txt|reason" "$REVOKE_SCRIPT" >/dev/null

log "security-ops gate: rotation + revocation automation smoke"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/reson-sandbox-security-ops.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

"$ROTATE_SCRIPT" --out-dir "$tmp_dir/bundle" --manifest "$tmp_dir/rotation.manifest.json" >/dev/null

for path in \
  "$tmp_dir/bundle/ca.crt" \
  "$tmp_dir/bundle/server.crt" \
  "$tmp_dir/bundle/client.crt" \
  "$tmp_dir/bundle/revocations.txt" \
  "$tmp_dir/rotation.manifest.json"; do
  [[ -f "$path" ]] || {
    err "expected artifact missing: $path"
    exit 1
  }
done

"$REVOKE_SCRIPT" \
  --bundle-dir "$tmp_dir/bundle" \
  --cert "$tmp_dir/bundle/client.crt" \
  --reason "gate20-test" \
  --manifest "$tmp_dir/revocation.manifest.json" >/dev/null

[[ -f "$tmp_dir/revocation.manifest.json" ]] || {
  err "revocation manifest missing"
  exit 1
}
rg -n "gate20-test" "$tmp_dir/bundle/revocations.txt" >/dev/null

log "security-ops gate: authz enforcement tests"
(cd "$REPO_ROOT" && cargo test -p vmd app::tests::authorize_metadata -- --nocapture)
(cd "$REPO_ROOT" && cargo test -p reson-sandbox compile_auth_header -- --nocapture)

log "security-ops gate: passed"
