#!/usr/bin/env bash
# @dive-file: Restores etcd data directories from snapshot backups with manifest output.
# @dive-rel: Used by scripts/dr/run_restore_drill.sh for DR rehearsal.
# @dive-rel: Supports dry-run/live restore workflows for operational readiness checks.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: restore_etcd.sh --snapshot <path> --restore-dir <dir> [--manifest <path>] [--name <name>] [--initial-cluster <cluster>] [--initial-advertise-peer-urls <urls>] [--etcdctl <bin>] [--dry-run]
USAGE
}

snapshot=""
restore_dir=""
manifest=""
name="${RESON_SANDBOX_DR_NODE_NAME:-reson-sandbox-dr}"
initial_cluster="${RESON_SANDBOX_DR_INITIAL_CLUSTER:-reson-sandbox-dr=http://127.0.0.1:2380}"
initial_advertise_peer_urls="${RESON_SANDBOX_DR_INITIAL_ADVERTISE_PEER_URLS:-http://127.0.0.1:2380}"
etcdctl_bin="${ETCDCTL_BIN:-etcdctl}"
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot)
      snapshot="${2:-}"
      shift 2
      ;;
    --restore-dir)
      restore_dir="${2:-}"
      shift 2
      ;;
    --manifest)
      manifest="${2:-}"
      shift 2
      ;;
    --name)
      name="${2:-}"
      shift 2
      ;;
    --initial-cluster)
      initial_cluster="${2:-}"
      shift 2
      ;;
    --initial-advertise-peer-urls)
      initial_advertise_peer_urls="${2:-}"
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

if [[ -z "$snapshot" || -z "$restore_dir" ]]; then
  echo "--snapshot and --restore-dir are required" >&2
  usage
  exit 2
fi

if [[ -z "$manifest" ]]; then
  manifest="${restore_dir}/restore.manifest.json"
fi

mkdir -p "$(dirname "$manifest")"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if [[ "$dry_run" -eq 1 ]]; then
  cat >"$manifest" <<JSON
{"type":"restore","mode":"dry-run","snapshot":"$snapshot","restore_dir":"$restore_dir","name":"$name","initial_cluster":"$initial_cluster","initial_advertise_peer_urls":"$initial_advertise_peer_urls","timestamp_utc":"$timestamp"}
JSON
  echo "[dr] dry-run restore manifest written: $manifest"
  exit 0
fi

if [[ ! -f "$snapshot" ]]; then
  echo "snapshot not found: $snapshot" >&2
  exit 1
fi
command -v "$etcdctl_bin" >/dev/null 2>&1 || {
  echo "etcdctl not found: $etcdctl_bin" >&2
  exit 1
}

rm -rf "$restore_dir"
"$etcdctl_bin" snapshot restore "$snapshot" \
  --name "$name" \
  --initial-cluster "$initial_cluster" \
  --initial-advertise-peer-urls "$initial_advertise_peer_urls" \
  --data-dir "$restore_dir"

cat >"$manifest" <<JSON
{"type":"restore","mode":"live","snapshot":"$snapshot","restore_dir":"$restore_dir","name":"$name","initial_cluster":"$initial_cluster","initial_advertise_peer_urls":"$initial_advertise_peer_urls","timestamp_utc":"$timestamp"}
JSON

echo "[dr] restore complete: $restore_dir"
