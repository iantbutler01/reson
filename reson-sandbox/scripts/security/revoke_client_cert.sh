#!/usr/bin/env bash
# @dive-file: Appends certificate revocation entries and emits revocation manifest metadata.
# @dive-rel: Used by verify_security_ops.sh gate and security operations runbooks.
# @dive-rel: Maintains revocation ledger for client-cert invalidation workflows.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: revoke_client_cert.sh --bundle-dir <dir> --cert <path> [--reason <text>] [--manifest <path>]
USAGE
}

bundle_dir=""
cert_path=""
reason="${RESON_SANDBOX_CERT_REVOKE_REASON:-manual-revocation}"
manifest=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      bundle_dir="${2:-}"
      shift 2
      ;;
    --cert)
      cert_path="${2:-}"
      shift 2
      ;;
    --reason)
      reason="${2:-}"
      shift 2
      ;;
    --manifest)
      manifest="${2:-}"
      shift 2
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

if [[ -z "$bundle_dir" || -z "$cert_path" ]]; then
  echo "--bundle-dir and --cert are required" >&2
  usage
  exit 2
fi
if [[ ! -f "$cert_path" ]]; then
  echo "certificate not found: $cert_path" >&2
  exit 1
fi

command -v openssl >/dev/null 2>&1 || {
  echo "openssl is required" >&2
  exit 1
}

revocations="$bundle_dir/revocations.txt"
mkdir -p "$bundle_dir"
touch "$revocations"
if [[ -z "$manifest" ]]; then
  manifest="$bundle_dir/revocation.manifest.json"
fi

serial="$(openssl x509 -in "$cert_path" -noout -serial | sed 's/^serial=//')"
timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s serial=%s reason=%s cert=%s\n' "$timestamp" "$serial" "$reason" "$cert_path" >>"$revocations"

cat >"$manifest" <<JSON
{"type":"tls_revocation","timestamp_utc":"$timestamp","serial":"$serial","reason":"$reason","cert":"$cert_path","revocations":"$revocations"}
JSON

echo "[security] revoked cert serial=$serial"
