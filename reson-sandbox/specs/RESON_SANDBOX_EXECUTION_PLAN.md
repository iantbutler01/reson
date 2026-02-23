<!-- @dive-file: Execution plan summary for reson-sandbox migration and delivery status. -->
<!-- @dive-rel: Tracks implementation progress against locked migration checklist expectations. -->
<!-- @dive-rel: Supports handoff context for verifier scripts and integration workstreams. -->
# Reson Sandbox Execution Plan

This document tracks implementation of the locked migration plan for `reson-sandbox`.

## Scope

- Include: `vmd`, `portproxy`, `proto/bracket/{vmd,portproxy}` and Rust facade crate.
- Exclude: Python integration paths and generated Python stubs.

## Delivered

- Workspace extraction of runtime crates.
- Root workspace wiring for `vmd`, `portproxy`, `crates/reson-sandbox`.
- High-level facade crate with:
  - daemon connect/auto-spawn
  - session acquire/attach/list/discard
  - channel-like bidi exec and shell streams
  - file ops and forward port helpers
  - `session.fork()` returning child session handle
- `ForkVM` proto and gRPC plumbing in `vmd`.
- CoW fork manager path that avoids full `copy_file` during fork.
- Verifier expansion with required facade and fork gates.
- Behavioral facade verifier coverage with mock gRPC contract tests:
  - session acquire/reuse, attach, close/discard semantics
  - file ops, bidi exec/shell ordered event streams
  - fork lineage and parent/child discard independence
  - concurrent session isolation semantics
  - forward-port handle teardown on `close()` and `drop` across repeated cycles
- Fork CoW runtime verifier coverage for stopped parent via `qemu-img` backing-file assertions.
- Restart/discovery normalization test to prevent stale running-state drift (`running -> stopped` on discover).
- `reson-rust` smoke test expansion with env-gated session + fork flow.

## Remaining Hardening

- Package and publish real Linux `portproxy` binaries into `portproxy/bin` for production runtime startup (current verifier fallback uses stubs when artifacts are absent).
- Add deep running-parent fork fidelity tests (memory snapshot resume semantics) in host-backed CI.
