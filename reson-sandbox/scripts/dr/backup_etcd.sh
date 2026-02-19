#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: backup_etcd.sh --snapshot <path> [--manifest <path>] [--endpoints <csv>] [--etcdctl <bin>] [--dry-run]
USAGE
}

snapshot=""
manifest=""
endpoints="${RESON_SANDBOX_ETCD_ENDPOINTS:-http://127.0.0.1:2379}"
etcdctl_bin="${ETCDCTL_BIN:-etcdctl}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot)
      snapshot="${2:-}"
      shift 2
      ;;
    --manifest)
      manifest="${2:-}"
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
    --dry-run)
      dry_run=1
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

if [[ -z "$snapshot" ]]; then
  echo "--snapshot is required" >&2
  usage
  exit 2
fi

if [[ -z "$manifest" ]]; then
  manifest="${snapshot}.manifest.json"
fi

mkdir -p "$(dirname "$snapshot")" "$(dirname "$manifest")"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$dry_run" -eq 1 ]]; then
  cat >"$manifest" <<JSON
{"type":"backup","mode":"dry-run","snapshot":"$snapshot","endpoints":"$endpoints","timestamp_utc":"$timestamp"}
JSON
  echo "[dr] dry-run backup manifest written: $manifest"
  exit 0
fi

command -v "$etcdctl_bin" >/dev/null 2>&1 || {
  echo "etcdctl not found: $etcdctl_bin" >&2
  exit 1
}

"$etcdctl_bin" --endpoints "$endpoints" snapshot save "$snapshot"
"$etcdctl_bin" --write-out json snapshot status "$snapshot" >"${snapshot}.status.json"

cat >"$manifest" <<JSON
{"type":"backup","mode":"live","snapshot":"$snapshot","status":"${snapshot}.status.json","endpoints":"$endpoints","timestamp_utc":"$timestamp"}
JSON

echo "[dr] backup complete: $snapshot"
