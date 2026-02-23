#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_dr_restore contract coverage.
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
    err "runtime sources missing; strict mode cannot run DR restore gate"
    exit 1
  fi
  warn "runtime sources missing; skipping DR restore gate"
  exit 0
fi

require_cmd rg

log "dr restore gate: static automation contract checks"
for required in \
  "$REPO_ROOT/scripts/dr/backup_etcd.sh" \
  "$REPO_ROOT/scripts/dr/restore_etcd.sh" \
  "$REPO_ROOT/scripts/dr/run_restore_drill.sh"; do
  [[ -x "$required" ]] || {
    err "missing executable DR automation script: ${required#$REPO_ROOT/}"
    exit 1
  }
done

rg -n "snapshot save|--endpoints" "$REPO_ROOT/scripts/dr/backup_etcd.sh" >/dev/null
rg -n "snapshot restore|--initial-cluster|--data-dir" "$REPO_ROOT/scripts/dr/restore_etcd.sh" >/dev/null
rg -n "backup_etcd.sh|restore_etcd.sh|drill_report.txt" "$REPO_ROOT/scripts/dr/run_restore_drill.sh" >/dev/null

log "dr restore gate: dry-run restore drill"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/reson-sandbox-dr-gate.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT
"$REPO_ROOT/scripts/dr/run_restore_drill.sh" --output-dir "$tmp_dir" >/dev/null

[[ -f "$tmp_dir/backup.manifest.json" ]] || {
  err "missing backup manifest from drill"
  exit 1
}
[[ -f "$tmp_dir/restore.manifest.json" ]] || {
  err "missing restore manifest from drill"
  exit 1
}
[[ -f "$tmp_dir/drill_report.txt" ]] || {
  err "missing drill report from drill"
  exit 1
}

rg -n "\"mode\":\"dry-run\"" "$tmp_dir/backup.manifest.json" "$tmp_dir/restore.manifest.json" >/dev/null
rg -n "^mode=dry-run$" "$tmp_dir/drill_report.txt" >/dev/null

log "dr restore gate: passed"
