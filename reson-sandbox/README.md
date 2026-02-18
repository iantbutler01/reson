# reson-sandbox

Sandbox runtime subproject for Reson.

This subproject is the extraction target for Rust VM sandbox runtime components currently implemented in OpenBracket (`vmd`, `portproxy`, and their proto contracts). It provides a stable dependency boundary for `reson-rust` via the high-level `reson-sandbox` facade crate.

## Current Status

Runtime code and facade crate are present with strict verifiers.

## Verification Entry Points

- `make verify` - Local verification with optional runtime gates
- `make verify-strict` - Enforces all gates (build/lint/test/proto/e2e/leaks/facade/fork)
- `make verify-e2e` - Runs strict verification with e2e enabled
- `make verify-fork` - Runs strict CoW fork gate directly
- `make verify-api` - Runs strict facade API gate directly

## Required Runtime Artifacts (for strict mode)

- `vmd/Cargo.toml`
- `portproxy/Cargo.toml`
- `crates/reson-sandbox/Cargo.toml`
- `proto/bracket/vmd/v1/vmd.proto`
- `proto/bracket/portproxy/v1/portproxy.proto`

## Notes

- Rust-only migration boundary: Python integrations and generated Python stubs are intentionally excluded.
- Strict mode is the CI enforcement target and includes fork CoW checks.
