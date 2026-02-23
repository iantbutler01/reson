#!/usr/bin/env bash
# @dive-file: Verifier gate script for verify_distributed_soak contract coverage.
# @dive-rel: Invoked by scripts/verify_reson_sandbox.sh gate orchestration and/or Makefile targets.
# @dive-rel: Uses scripts/common.sh helpers for strict-mode behavior, diagnostics, and command validation.

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0

usage() {
  cat <<'USAGE'
Usage: verify_distributed_soak.sh [--strict]
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
    err "runtime sources missing; strict mode cannot run distributed soak gate"
    exit 1
  fi
  warn "runtime sources missing; skipping distributed soak gate"
  exit 0
fi

ITERATIONS="${VMD_SOAK_ITERATIONS:-}"
if [[ -z "$ITERATIONS" ]]; then
  if [[ "$STRICT" -eq 1 ]]; then
    ITERATIONS=2
  else
    ITERATIONS=1
  fi
fi

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]] || [[ "$ITERATIONS" -lt 1 ]]; then
  err "VMD_SOAK_ITERATIONS must be a positive integer"
  exit 2
fi

TMP_DIR="$(mktemp -d "/tmp/rsbsoak.XXXXXX")"
RUNNER="$TMP_DIR/run_soak.sh"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

STRICT_FLAG=""
if [[ "$STRICT" -eq 1 ]]; then
  STRICT_FLAG="--strict"
fi

cat >"$RUNNER" <<SOAK
#!/usr/bin/env bash
set -euo pipefail
for i in \$(seq 1 "$ITERATIONS"); do
  echo "[verify] distributed soak iteration \$i/$ITERATIONS"
  "$REPO_ROOT/scripts/verify_e2e_smoke.sh" $STRICT_FLAG
  "$REPO_ROOT/scripts/verify_mq_control.sh" $STRICT_FLAG
  "$REPO_ROOT/scripts/verify_reconcile.sh" $STRICT_FLAG
done
SOAK
chmod +x "$RUNNER"

log "distributed soak gate: running $ITERATIONS iterations under leak harness"
"$REPO_ROOT/scripts/verify_no_leaks.sh" -- "$RUNNER"
log "distributed soak gate: passed"
