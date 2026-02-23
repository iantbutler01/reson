<!-- @dive-file: High-level map of reson-sandbox components and how contract decisions map to implementation and verification. -->
<!-- @dive-rel: Summarizes execution boundaries enforced by specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md. -->
<!-- @dive-rel: Links readiness definitions to scripts/verify_reson_sandbox.sh gate orchestration. -->
# System Overview
reson-sandbox provides a Rust-first VM sandbox runtime and facade API with local auto-spawn and distributed control-plane operation.

## Components
- **HA Distributed Contract** - Defines current execution-scope checklist and explicit out-of-scope topics -> `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md`
- **Real Integration Plan** - Migration plan from mock-heavy gates to real distributed machinery verification -> `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md`
- **Verification Harness Module Map** - Dive metadata for strict gate scripts, drill scripts, and shared verifier helpers -> `.dive/modules/verification_harness.md`
- **Runtime Support Module Map** - Dive metadata for supporting runtime modules in `vmd`, `portproxy`, and facade SLO evaluation -> `.dive/modules/runtime_support.md`
- **Verifier Orchestrator** - Runs strict gate sequence and enforces repository contract -> `scripts/verify_reson_sandbox.sh`
- **Integration Harness** - Boots real control-plane dependencies and runs profile-scoped distributed probes -> `scripts/integration/`
- **Real Facade Routing Test** - Verifies public facade uses healthy control gateways when primary endpoint is unhealthy -> `crates/reson-sandbox/tests/real_control_gateway_routing.rs`
- **Real Failover Harness** - Runs active-stream continuity and exactly-once failover probes on live daemons -> `scripts/integration/verify_real_failover.sh`
- **Real Control-Plane Failure Harness** - Runs etcd quorum-loss, NATS outage, and post-restart reconcile convergence probes on live daemons -> `scripts/integration/verify_control_plane_failures.sh`
- **Real Planned Drain Harness** - Runs two-node admission-freeze handoff probe with in-flight continuity assertions -> `scripts/integration/verify_planned_drain_handoff.sh`
- **Real Warm-Pool Harness** - Runs startup prewarm, warm-hit, cold-refill, and local auto-spawn readiness probes on live daemons -> `scripts/integration/verify_real_warm_pool.sh`
- **Real Stability Churn Gate** - Runs repeated distributed failover cycles under leak harness and bounded RSS growth assertions -> `scripts/verify_real_gate47.sh`
- **Facade Crate** - Stable consumer API for session lifecycle, exec/shell, fork, and attach/list flows -> `crates/reson-sandbox/src/lib.rs`
- **Tier-B Continuity + Fidelity Layer** - Rebind, restore, exactly-once failover handling, and `tier_b_eligible` policy enforcement -> `crates/reson-sandbox/src/lib.rs`
- **Warm Pool Pipeline** - Architecture-aware image prewarm and cold-path refill hooks -> `crates/reson-sandbox/src/lib.rs`
- **Runtime Daemon** - VM orchestration, control bus, reconcile, and partition handling -> `vmd/src`
- **VM Ownership Reclaim** - Pre-start local fencing for orphan qemu processes via QMP/pid/path-matched process sweep -> `vmd/src/state/manager.rs`
- **Exec Transport Recovery** - Session exec stream establish recovery path that invalidates RPC cache and requests restart on first transport failure -> `crates/reson-sandbox/src/lib.rs`
- **Control-Plane Exec Routing** - Distributed `Session::exec` path now uses stream-scoped `exec.stream.start`/`exec.stream.input` commands, ordered event consumption, and fast failover rebind attempts while preserving facade API shape -> `crates/reson-sandbox/src/lib.rs`
- **Stream Event Identity Contract** - Distributed stream events now require `cluster_id`, `logical_stream_id`, `event_seq`, `event_id`, and `producer_epoch` with checkpointed resume/no-replay semantics across failover -> `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md`
- **Checkpointed Stream Resume Runtime** - Distributed exec stream recovery currently resumes event consumption from last committed sequence without replay and avoids command rerun on terminal producer errors; producer reattach after node loss is still pending -> `crates/reson-sandbox/src/lib.rs`
- **L3 Producer Reattach Contract** - Locked continuity requirement that active logical streams must recover on surviving node via producer reattach after VM execution-state restore -> `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md`
- **Fork Restore Marker Propagation** - Running-parent fork now stamps child metadata with execution restore snapshot markers for Tier-B rehydrate logic -> `vmd/src/state/manager.rs`
- **Real Control-Plane Failure Tests** - Rust integration tests for L3 fail-closed semantics and recovery under etcd/NATS outages -> `crates/reson-sandbox/tests/real_control_plane_failures.rs`
- **Node-Local Exec Command Handler** - Control-bus consumer executes `exec.run` plus stream-oriented `exec.stream.start`/`exec.stream.input` on the owning node and emits ordered stream events -> `vmd/src/control_bus.rs`
- **Distributed Exec Stream Subscriber** - Distributed control adapter subscribes to stream-scoped JetStream subjects for ordered `exec.stream` event delivery to facade handles -> `crates/reson-sandbox/src/distributed.rs`
- **VMD Portproxy Proto Wiring** - vmd build/proto modules now generate and expose portproxy client stubs for node-local command execution -> `vmd/build.rs`
- **Attachable Exec Producer Path** - `exec.stream.start` now uses named daemon exec + attach semantics so failover resume can reattach existing producers without command rerun -> `vmd/src/control_bus.rs`
- **Daemon Exit Propagation** - Daemon manager now emits explicit exit codes on attach streams so distributed exec producers can publish terminal events after reattach -> `portproxy/src/daemon.rs`

## Relationships
- HA Distributed Contract -> Verifier Orchestrator: contract checklists and gates define what strict verification must enforce.
- Real Integration Plan -> Verifier Orchestrator: defines additional real-machinery gates required to elevate evidence quality.
- HA Distributed Contract -> Roadmap Scope: out-of-scope topics are explicitly excluded from the current execution pass.
- HA Distributed Contract -> Facade Crate: locked user-facing API boundaries constrain distributed evolution.
- Tier-B Continuity + Fidelity Layer -> Verifier Orchestrator: gates 27/33/35 enforce restore, failover exactly-once, and fidelity policy behavior.
- Warm Pool Pipeline -> Verifier Orchestrator: gate 36 enforces architecture-aware prewarm execution and warm/cold path hooks.
- Verifier Orchestrator -> Runtime Daemon: gate scripts validate runtime behavior and safety invariants.
- Verification Harness Module Map -> Verifier Orchestrator: script-level metadata explains which gate scripts implement each contract dimension.
- Runtime Support Module Map -> Runtime Daemon: module-level metadata maps helper/runtime files to lifecycle and control-plane behavior.
- Integration Harness -> Runtime Daemon: launches real `vmd` processes and validates externally visible etcd registration/heartbeat behavior.
- Real Facade Routing Test -> Integration Harness: consumes live node endpoints from integration scripts and validates gateway routing via public facade APIs.
- Real Failover Harness -> Facade Crate: exercises live `Session::exec` failover semantics under primary-loss conditions.
- Stream Event Identity Contract -> Real Integration Plan: defines new real test scope for checkpointed resume and terminal-no-rerun behavior (`gate 48` path).
- Real Control-Plane Failure Harness -> Real Control-Plane Failure Tests: supplies container IDs/endpoints used for live outage injection and recovery checks.
- Real Planned Drain Harness -> Real Control-Plane Failure Tests: executes planned drain admission-freeze handoff semantics on two live nodes.
- Real Warm-Pool Harness -> `crates/reson-sandbox/tests/real_warm_pool_pipeline.rs`: executes section `7.4` real selectors, including local auto-spawn first-call transport readiness.
- Real Stability Churn Gate -> Real Failover Harness: repeatedly executes distributed failover selectors and enforces no-orphan + bounded-growth stability checks.
- VM Ownership Reclaim -> Real Failover Harness: prevents stale local qemu disk-lock ownership from blocking secondary restart/adopt in shared-data failover tests.
- Exec Transport Recovery -> Real Failover Harness: retries exec establishment with endpoint/rpc cache reset + restart to recover post-failover stream setup.
- Control-Plane Exec Routing -> Node-Local Exec Command Handler: facade-issued `exec.stream.start`/`exec.stream.input` commands are executed by the target node control consumer.
- Node-Local Exec Command Handler -> Distributed Exec Stream Subscriber: stream-scoped `evt.exec.stream.<stream_id>` events are consumed by per-stream JetStream subscriptions.
- Checkpointed Stream Resume Runtime -> Distributed Exec Stream Subscriber: resume subscriptions enforce `last_seq + 1` forward-only delivery after rebind/reconnect.
- L3 Producer Reattach Contract -> Control-Plane Exec Routing: distributed failover must restore VM execution state and continue the same logical stream producer without fresh command rerun.
- VMD Portproxy Proto Wiring -> Node-Local Exec Command Handler: portproxy `ShellExec` client stubs enable control-bus command execution without direct facade guest-RPC dialing.
- Fork Restore Marker Propagation -> Tier-B Continuity + Fidelity Layer: enables marker-based restore selection without relying on child snapshot list materialization.
- Real Control-Plane Failure Tests -> Facade Crate: validates distributed-control fail-closed behavior through public facade operations only.
- Attachable Exec Producer Path -> Distributed Exec Stream Subscriber: rebind now republishes `exec.stream.start` as attach-only resume so event streams continue from checkpoint instead of rerunning commands.
- Daemon Exit Propagation -> Attachable Exec Producer Path: attach stream `exit_code` frames map to canonical `exit/timeout` stream events for terminal-no-rerun semantics.
