<!-- @dive-file: Acceptance checklist for migration parity and runtime/facade behavior guarantees. -->
<!-- @dive-rel: Serves as requirement source for scripts/verify_reson_sandbox.sh gate ordering. -->
<!-- @dive-rel: Complements the HA contract by enumerating concrete behavioral acceptance points. -->
# Reson Sandbox Migration Checklist

This checklist is the locked acceptance surface for extraction from OpenBracket into `reson-sandbox`.

## Functional Requirements

- [x] Session acquire creates or reuses VM deterministically.
- [x] Session release is idempotent and always cleans resources.
- [x] Bidi exec supports stdin/stdout/stderr/exit/timeout.
- [x] Interactive shell supports PTY and clean termination.
- [x] File ops read/write/list/delete have deterministic error behavior.
- [x] Port forwarding is stable and torn down on release.
- [x] Concurrent sessions are isolated.
- [x] Restart/discovery behavior prevents stale running state drift.
- [x] Error mapping to client boundary is stable and typed.
- [x] Proto contracts are locked and reproducible.
- [x] No process/port leaks after repeated lifecycle runs.
- [x] Fork creates independent parent/child branches with durable lineage IDs.
- [x] Fork path uses CoW overlays without full disk copy.

## Verifier Gates

- [x] Gate 0: script hygiene and repository contract checks.
- [x] Gate 1: build gate (`cargo check`) for sandbox crates.
- [x] Gate 2: lint gate (`cargo clippy -- -D warnings`).
- [x] Gate 3: unit/integration tests for `vmd` and `portproxy`.
- [x] Gate 4: proto gate (`verify_proto_clean.sh`).
- [x] Gate 5: e2e smoke gate (`verify_e2e_smoke.sh`).
- [x] Gate 6: leak gate (`verify_no_leaks.sh`).
- [x] Gate 7: facade API gate (`verify_api_facade.sh`).
- [x] Gate 8: fork CoW gate (`verify_fork_cow.sh`).

## Migration Constraints

- [x] Preserve existing gRPC interfaces first; avoid redesign during extraction.
- [x] Keep VM internals (`qemu`, image conversion, bootstrap) behind a stable client boundary.
- [x] Keep behavior parity with source implementation before optimization/refactor.
- [x] Keep migration boundary Rust-only; Python runtime integration is out of scope.

## Host-Dependent Pending

- [x] Run gate 5 and gate 6 in a host profile with VM prerequisites and `VMD_SMOKE_SOURCE_REF` configured.
- [x] Run strict mode (`verify_reson_sandbox.sh --strict`) to completion in CI once host inputs are available.
- [x] Validate leak gate (gate 6) against live qemu/portproxy process baselines in host CI.

## Notes

- Strict verification currently provisions fallback `PROXY_BIN` stubs when `portproxy/bin/portproxy-linux-{amd64,arm64}` artifacts are absent.
- Production runtime should supply real Linux portproxy artifacts in `portproxy/bin`.
