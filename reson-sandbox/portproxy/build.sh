#!/usr/bin/env bash
# @dive-file: Build helper for portproxy binaries and guest artifacts.
# @dive-rel: Used by developers/CI to produce executable outputs for runtime packaging.
# @dive-rel: Complements Cargo build paths used by verification and integration harnesses.

set -euo pipefail

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found in PATH" >&2
  exit 1
fi

if ! command -v cross >/dev/null 2>&1; then
  echo "Installing cross..."
  cargo install cross --git https://github.com/cross-rs/cross --locked
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
BIN_DIR="${ROOT}/bin"
mkdir -p "${BIN_DIR}"

platforms=(
  "linux/amd64:x86_64-unknown-linux-gnu"
  "linux/arm64:aarch64-unknown-linux-gnu"
)

if [[ "$(uname -s)" == "Darwin" ]]; then
  platforms+=(
    "darwin/amd64:x86_64-apple-darwin"
    "darwin/arm64:aarch64-apple-darwin"
  )
fi

for entry in "${platforms[@]}"; do
  platform="${entry%%:*}"
  target="${entry##*:}"

  echo "Building ${platform} (${target})..."
  (cd "${REPO_ROOT}" && cross build --manifest-path "portproxy/Cargo.toml" --release --target "${target}")

  go_arch="${platform#*/}"
  go_os="${platform%/*}"
  outfile="${BIN_DIR}/portproxy-${go_os}-${go_arch}"
  src="${ROOT}/target/${target}/release/portproxy"

  if [[ ! -f "${src}" ]]; then
    echo "Expected binary ${src} not found; cross build may have failed." >&2
    exit 1
  fi

  cp "${src}" "${outfile}"
  chmod +x "${outfile}"
done

echo "Binaries available in ${BIN_DIR}"
