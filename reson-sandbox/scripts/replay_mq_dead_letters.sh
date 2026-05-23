#!/usr/bin/env bash
# @dive-file: Operator helper to replay dead-letter control messages through mqctl.
# @dive-rel: Wraps vmd mqctl binary so DLQ recovery is scriptable from repo root.
# @dive-rel: Supports outbox/control-bus operational runbooks and drills.

set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

limit="${1:-100}"
shift || true

if ! [[ "$limit" =~ ^[0-9]+$ ]]; then
  err "first argument must be numeric replay limit"
  exit 2
fi

log "replaying up to $limit dead-letter control messages"
(cd "$REPO_ROOT" && cargo run -p vmd --bin mqctl -- replay-dlq --limit "$limit" "$@")
