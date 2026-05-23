#!/usr/bin/env bash
# @dive-file: Orchestrates backup+restore disaster-recovery drill and report generation.
# @dive-rel: Composes scripts/dr/backup_etcd.sh and scripts/dr/restore_etcd.sh.
# @dive-rel: Used by verify_dr_restore.sh gate and DR runbook execution.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage() {
  cat <<'USAGE'
Usage: run_restore_drill.sh [--output-dir <dir>] [--endpoints <csv>] [--etcdctl <bin>] [--live]
USAGE
}

output_dir=""
endpoints="${RESON_SANDBOX_ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
etcdctl_bin="${ETCDCTL_BIN:-etcdctl}"
live=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="${2:-}"
      shift 2
      ;;
    --endpoints)
      endpoints="${2:-}"
      shift 2
      ;;
    --etcdctl)
      etcdctl_bin="${2:-}"
      shift 2
      ;;
    --live)
      live=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "$output_dir" ]]; then
  output_dir="$(cd "$SCRIPT_DIR/../.." && pwd)/.dr-drills/$timestamp"
fi

mkdir -p "$output_dir"

snapshot="$output_dir/etcd.snapshot.db"
backup_manifest="$output_dir/backup.manifest.json"
restore_dir="$output_dir/restored-data"
restore_manifest="$output_dir/restore.manifest.json"
report="$output_dir/drill_report.txt"

mode_flag=(--dry-run)
mode_label="dry-run"
if [[ "$live" -eq 1 ]]; then
  mode_flag=()
  mode_label="live"
fi

"$SCRIPT_DIR/backup_etcd.sh" \
  --snapshot "$snapshot" \
  --manifest "$backup_manifest" \
  --endpoints "$endpoints" \
  --etcdctl "$etcdctl_bin" \
  "${mode_flag[@]}"

"$SCRIPT_DIR/restore_etcd.sh" \
  --snapshot "$snapshot" \
  --restore-dir "$restore_dir" \
  --manifest "$restore_manifest" \
  --etcdctl "$etcdctl_bin" \
  "${mode_flag[@]}"

cat >"$report" <<TXT
drill_timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
mode=$mode_label
etcd_endpoints=$endpoints
backup_manifest=$backup_manifest
restore_manifest=$restore_manifest
snapshot=$snapshot
restore_dir=$restore_dir
TXT

echo "[dr] restore drill complete ($mode_label): $report"
