#!/usr/bin/env bash
# @dive-file: Generates CA/server/client TLS bundle artifacts for rotation workflows.
# @dive-rel: Used by verify_security_ops.sh gate and security runbooks.
# @dive-rel: Produces deterministic manifest outputs for rotation auditability.

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: rotate_tls_bundle.sh --out-dir <dir> [--server-cn <cn>] [--client-cn <cn>] [--days <n>] [--manifest <path>]
USAGE
}

out_dir=""
server_cn="${RESON_SANDBOX_TLS_SERVER_CN:-reson-sandbox-server}"
client_cn="${RESON_SANDBOX_TLS_CLIENT_CN:-reson-sandbox-client}"
days="${RESON_SANDBOX_TLS_DAYS:-365}"
manifest=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      out_dir="${2:-}"
      shift 2
      ;;
    --server-cn)
      server_cn="${2:-}"
      shift 2
      ;;
    --client-cn)
      client_cn="${2:-}"
      shift 2
      ;;
    --days)
      days="${2:-}"
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

if [[ -z "$out_dir" ]]; then
  echo "--out-dir is required" >&2
  usage
  exit 2
fi

command -v openssl >/dev/null 2>&1 || {
  echo "openssl is required" >&2
  exit 1
}

mkdir -p "$out_dir"
if [[ -z "$manifest" ]]; then
  manifest="$out_dir/rotation.manifest.json"
fi

ca_key="$out_dir/ca.key"
ca_cert="$out_dir/ca.crt"
server_key="$out_dir/server.key"
server_csr="$out_dir/server.csr"
server_cert="$out_dir/server.crt"
client_key="$out_dir/client.key"
client_csr="$out_dir/client.csr"
client_cert="$out_dir/client.crt"
revocations="$out_dir/revocations.txt"

openssl genrsa -out "$ca_key" 4096 >/dev/null 2>&1
openssl req -x509 -new -nodes -key "$ca_key" -sha256 -days "$days" -subj "/CN=reson-sandbox-ca" -out "$ca_cert" >/dev/null 2>&1

openssl genrsa -out "$server_key" 2048 >/dev/null 2>&1
openssl req -new -key "$server_key" -subj "/CN=$server_cn" -out "$server_csr" >/dev/null 2>&1
openssl x509 -req -in "$server_csr" -CA "$ca_cert" -CAkey "$ca_key" -CAcreateserial -out "$server_cert" -days "$days" -sha256 >/dev/null 2>&1

openssl genrsa -out "$client_key" 2048 >/dev/null 2>&1
openssl req -new -key "$client_key" -subj "/CN=$client_cn" -out "$client_csr" >/dev/null 2>&1
openssl x509 -req -in "$client_csr" -CA "$ca_cert" -CAkey "$ca_key" -CAcreateserial -out "$client_cert" -days "$days" -sha256 >/dev/null 2>&1

: >"$revocations"
chmod 600 "$ca_key" "$server_key" "$client_key"

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
cat >"$manifest" <<JSON
{"type":"tls_rotation","timestamp_utc":"$timestamp","ca_cert":"$ca_cert","server_cert":"$server_cert","client_cert":"$client_cert","revocations":"$revocations"}
JSON

echo "[security] tls bundle rotated: $out_dir"
