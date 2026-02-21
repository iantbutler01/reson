<!-- @dive-file: Module-level metadata for the HA distributed contract and readiness-scope decisions. -->
<!-- @dive-rel: Tracks how contract edits influence strict verification and implementation priorities. -->
<!-- @dive-rel: Anchors explicit out-of-scope decisions so execution does not drift into deferred work. -->
# HA Contract Module
Defines distributed HA readiness criteria and the execution checklist used to drive implementation sequencing.

## Files
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` - Locked distributed contract with blocking checklist and explicit out-of-scope exclusions.
- `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md` - Locked draft plan for moving Tier-B/HA proof paths from mock-heavy checks to real integration evidence.
- `.dive/overview.md` - System-level index linking contract intent to implementation and verifier orchestration.
- `scripts/verify_tierb_execution_restore.sh` - Gate for cross-node execution-state restore behavior (`12.32`).
- `scripts/verify_tierb_exactly_once.sh` - Gate for failover exactly-once command dispatch coverage (`12.33`).
- `scripts/verify_execution_fidelity_policy.sh` - Gate for `tier_b_eligible` classifier/enforcement coverage (`12.34`).
- `scripts/verify_warm_pool_pipeline.sh` - Gate for architecture-aware prewarm pipeline coverage (`12.35`).
- `scripts/integration/verify_two_node_registry.sh` - Real two-node integration probe for etcd-backed node registration/heartbeat evidence (`7.2.1`).
- `crates/reson-sandbox/tests/real_control_gateway_routing.rs` - Real facade integration test proving control-gateway endpoint routing with unhealthy primary endpoint (`7.2.2`).
- `scripts/integration/verify_real_failover.sh` - Real failover harness for 7.3 continuity/exactly-once checks with deferred secondary activation on primary loss.
- `crates/reson-sandbox/tests/real_failover_continuity.rs` - Real failover test cases for active-stream continuity and exactly-once acknowledgement semantics.
- `scripts/integration/verify_control_plane_failures.sh` - Real integration harness that injects etcd quorum and NATS outages and validates fail-closed + recovery behavior.
- `crates/reson-sandbox/tests/real_control_plane_failures.rs` - Real distributed control-plane outage tests run through public facade APIs.
- `crates/reson-sandbox/tests/real_control_plane_failures.rs` - Also includes planned-drain handoff and post-restart reconcile convergence probes.
- `scripts/integration/verify_planned_drain_handoff.sh` - Real two-node planned-drain harness for admission-freeze handoff and in-flight continuity.
- `scripts/integration/verify_real_warm_pool.sh` - Real warm-pool harness for startup prewarm, warm-hit, cold-refill, and local auto-spawn readiness probes (`7.4`).
- `crates/reson-sandbox/tests/real_warm_pool_pipeline.rs` - Real warm-pool integration selectors covering startup prewarm, request-path cache hits, refill evidence, and local auto-spawn transport readiness.
- `scripts/verify_real_gate47.sh` - Real stability gate running repeated distributed failover cycles under leak and bounded-RSS checks.
- `vmd/src/assets/portproxy.rs` - Runtime discovery for guest `portproxy` binaries; now resolves from executable-relative paths to avoid cwd-dependent bootstrap failures.
- `vmd/src/state/manager.rs` - VM lifecycle manager now includes stricter orphan-runtime fencing (QMP/pid/path sweep) before start/reclaim paths.
- `crates/reson-sandbox/src/lib.rs` - Facade exec establish path now applies staged transport recovery, broadens retryable failover transport classes, and retries local VM restart when guest RPC readiness is stale.
- `crates/reson-sandbox/src/distributed.rs` - Distributed control adapter now publishes events on `.evt.*` subjects so JetStream stream filters capture lifecycle events durably.
- `crates/reson-sandbox/src/lib.rs` - Distributed `Session::exec` now routes through control-plane `exec.stream.start`/`exec.stream.input` commands with ordered event delivery and rebind-driven stream continuity while direct local/non-distributed paths remain guest-RPC based.
- `vmd/src/control_bus.rs` - Control consumer now emits stream identity envelope fields (`cluster_id`, `logical_stream_id`, `event_seq`, `event_id`, `producer_epoch`) and cluster-scoped event ids on stream frames.
- `crates/reson-sandbox/src/lib.rs` - Distributed exec recovery now republishs `exec.stream.start` resume commands on rebind so producer reattach is attempted before terminal failure.
- `vmd/src/control_bus.rs` - Distributed stream start now uses named daemon exec + attach mode; resume requests are attach-only and emit deterministic error instead of rerun when producer is missing.
- `portproxy/src/daemon.rs` - Daemon registry now tracks exit status and exposes it to attach consumers for terminal stream event publication.
- `portproxy/src/services.rs` - Attach daemon stream now forwards `exit_code` frames in addition to stdout/stderr.
- `crates/reson-sandbox/src/distributed.rs` - Adds node-id resolution by endpoint and command-id-scoped exec result waits using ephemeral JetStream consumers.
- `crates/reson-sandbox/src/distributed.rs` - Stream subscription API now accepts resume checkpoints and drops replayed frames at/below committed sequence.
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` - Locks distributed stream event identity/resume semantics (`cluster_id`, `logical_stream_id`, `event_seq`, `event_id`, `producer_epoch`) and no-replay terminal handling.
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` - Locks `L3` producer-reattach continuity semantics for unplanned node loss and marks unfinished stream-resume checklist items back to pending.
- `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md` - Adds real test/gate work for checkpointed resume (`last_seq + 1`), no replay, and terminal-no-rerun guarantees (`gate 48` planning).
- `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md` - Extends `7.3` with explicit real tests for producer reattach continuity, cross-epoch sequence continuation, and no fresh-command rerun on rebind.
- `vmd/build.rs` - Generates vmd + portproxy protobuf bindings so control-bus workers can invoke `ShellExec` clients locally.
- `vmd/src/lib.rs` - Exposes generated `bracket.portproxy.v1` protobuf module used by control-bus exec command handlers.
- `vmd/src/state/manager.rs` - Running-parent fork now stamps child metadata with execution-restore snapshot markers (`reson.execution_restore_snapshot_id/name`) for Tier-B rehydrate paths.
- `specs/runbooks/TIERB_EXECUTION_STATE_RESTORE.md` - Runbook for restore markers and rebound ordering.
- `specs/runbooks/TIERB_MID_COMMAND_FAILOVER_EXACTLY_ONCE.md` - Runbook for failover exactly-once validation flow.
- `specs/runbooks/TIERB_EXECUTION_FIDELITY_POLICY.md` - Runbook for eligibility classification and strict restore requirements.
- `specs/runbooks/WARM_POOL_AND_PREWARM_PIPELINE.md` - Runbook for startup prewarm and warm-pool refill path.
- `crates/reson-sandbox/tests/facade_contract.rs` - Contract tests backing Tier-B restore, failover exactly-once, fidelity policy, and warm-pool prewarm semantics.

## Relationships
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` -> `scripts/verify_reson_sandbox.sh`: checklist and gate definitions drive strict pass requirements.
- `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md` -> `scripts/verify_reson_sandbox.sh`: defines next-wave gate additions (`41-47`) for real machinery validation.
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` -> execution scope: explicit out-of-scope topics prevent nonessential hardening work from blocking runtime completion.
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` -> `crates/reson-sandbox/src/lib.rs`: API and continuity constraints shape facade behavior.
- `scripts/verify_tierb_execution_restore.sh` -> `crates/reson-sandbox/tests/facade_contract.rs`: runs node-loss restore contract test.
- `scripts/verify_tierb_exactly_once.sh` -> `crates/reson-sandbox/tests/facade_contract.rs`: runs in-flight failover exactly-once dispatch test.
- `scripts/verify_execution_fidelity_policy.sh` -> `crates/reson-sandbox/src/lib.rs`: validates metadata classifier and strict Tier-B restore policy path.
- `scripts/verify_warm_pool_pipeline.sh` -> `crates/reson-sandbox/src/lib.rs`: validates architecture-aware prewarm + refill behavior.
- `scripts/integration/verify_two_node_registry.sh` -> `vmd/src/registry.rs`: validates lease-backed node key writes and heartbeat progression on real etcd.
- `scripts/integration/verify_two_node_registry.sh` -> `crates/reson-sandbox/tests/real_control_gateway_routing.rs`: executes real facade gateway-routing coverage on live daemons.
- `scripts/integration/verify_real_failover.sh` -> `crates/reson-sandbox/tests/real_failover_continuity.rs`: runs real failover scenarios with live daemon lifecycle orchestration.
- `scripts/integration/verify_real_failover.sh` -> distributed mode: `--distributed` enables control-bus-backed node startup and runs dedicated L3 failover test selector.
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` -> `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md`: stream identity/ordering rules require explicit real failover checkpoint assertions, not mock-only validation.
- `scripts/integration/verify_control_plane_failures.sh` -> `crates/reson-sandbox/tests/real_control_plane_failures.rs`: runs real outage injection scenarios for etcd/NATS fail-closed/recovery semantics.
- `scripts/integration/verify_planned_drain_handoff.sh` -> `scripts/integration/verify_two_node_registry.sh`: reuses two-node registry harness with admission-freeze profile for planned drain test execution.
- `scripts/integration/verify_real_warm_pool.sh` -> `crates/reson-sandbox/tests/real_warm_pool_pipeline.rs`: runs selector-isolated real warm-pool tests and requires local auto-spawn readiness coverage by default.
- `scripts/verify_real_gate47.sh` -> `scripts/verify_no_leaks.sh`: composes repeated distributed failover runs with process-count leak assertions and RSS growth bounds.
- `vmd/src/assets/portproxy.rs` -> VM bootstrap path: ensures `create-vm` can package guest proxy binaries regardless of invocation working directory.
- `vmd/src/state/manager.rs` -> `scripts/integration/verify_real_failover.sh`: pre-start ownership reclaim attempts to eliminate stale-qemu lock contention before secondary takeover.
- `crates/reson-sandbox/src/lib.rs` -> `crates/reson-sandbox/tests/real_failover_continuity.rs`: transport recovery and timeout behavior directly influence post-failover `Session::exec` resiliency.
- `crates/reson-sandbox/src/lib.rs` -> `vmd/src/control_bus.rs`: distributed-mode exec dispatch now flows through control commands instead of direct facade guest-RPC execution.
- `vmd/src/control_bus.rs` -> `crates/reson-sandbox/src/distributed.rs`: emits command-id-scoped exec result events consumed by correlated waiters.
- `crates/reson-sandbox/src/lib.rs` -> `crates/reson-sandbox/src/distributed.rs`: rebind path re-subscribes by logical stream id with sequence checkpoint instead of republishing exec start commands.
- `crates/reson-sandbox/src/lib.rs` -> `crates/reson-sandbox/src/distributed.rs`: rebind path now re-subscribes and republishes `exec.stream.start` in resume mode to trigger producer reattach on the rebound owner node.
- `vmd/src/control_bus.rs` -> `portproxy/src/services.rs`: control-bus stream producer binds to daemon-manager attach streams (stdout/stderr/exit) instead of one-shot shell exec RPC.
- `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md` -> `specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md`: producer-reattach continuity contract requires dedicated real failover tests before `L3` stream continuity can be marked complete.
- `vmd/build.rs` -> `vmd/src/lib.rs`: generated portproxy client bindings are required for node-local control-bus exec execution.
- `vmd/src/state/manager.rs` -> `crates/reson-sandbox/tests/real_failover_continuity.rs`: fork child metadata now carries explicit restore markers so Tier-B policy tests can validate rehydrate intent without snapshot-list inference.
- `crates/reson-sandbox/src/distributed.rs` -> control stream subjects: `.evt.*` publish path alignment is required for observability assertions in real distributed tests.
