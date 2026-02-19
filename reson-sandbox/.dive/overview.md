<!-- @dive-file: High-level map of reson-sandbox components and how contract decisions map to implementation and verification. -->
<!-- @dive-rel: Summarizes execution boundaries enforced by specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md. -->
<!-- @dive-rel: Links readiness definitions to scripts/verify_reson_sandbox.sh gate orchestration. -->
# System Overview
reson-sandbox provides a Rust-first VM sandbox runtime and facade API with local auto-spawn and distributed control-plane operation.

## Components
- **HA Distributed Contract** - Defines current execution-scope checklist and explicit out-of-scope topics -> `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md`
- **Real Integration Plan** - Migration plan from mock-heavy gates to real distributed machinery verification -> `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md`
- **Verifier Orchestrator** - Runs strict gate sequence and enforces repository contract -> `scripts/verify_reson_sandbox.sh`
- **Facade Crate** - Stable consumer API for session lifecycle, exec/shell, fork, and attach/list flows -> `crates/reson-sandbox/src/lib.rs`
- **Tier-B Continuity + Fidelity Layer** - Rebind, restore, exactly-once failover handling, and `tier_b_eligible` policy enforcement -> `crates/reson-sandbox/src/lib.rs`
- **Warm Pool Pipeline** - Architecture-aware image prewarm and cold-path refill hooks -> `crates/reson-sandbox/src/lib.rs`
- **Runtime Daemon** - VM orchestration, control bus, reconcile, and partition handling -> `vmd/src`

## Relationships
- HA Distributed Contract -> Verifier Orchestrator: contract checklists and gates define what strict verification must enforce.
- Real Integration Plan -> Verifier Orchestrator: defines additional real-machinery gates required to elevate evidence quality.
- HA Distributed Contract -> Roadmap Scope: out-of-scope topics are explicitly excluded from the current execution pass.
- HA Distributed Contract -> Facade Crate: locked user-facing API boundaries constrain distributed evolution.
- Tier-B Continuity + Fidelity Layer -> Verifier Orchestrator: gates 27/33/35 enforce restore, failover exactly-once, and fidelity policy behavior.
- Warm Pool Pipeline -> Verifier Orchestrator: gate 36 enforces architecture-aware prewarm execution and warm/cold path hooks.
- Verifier Orchestrator -> Runtime Daemon: gate scripts validate runtime behavior and safety invariants.
