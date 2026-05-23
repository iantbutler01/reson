<!-- @dive-file: Defines the distributed HA contract, verifier expectations, and completion criteria for reson-sandbox. -->
<!-- @dive-rel: Drives implementation and gate scope in scripts/verify_reson_sandbox.sh. -->
<!-- @dive-rel: Serves as readiness source for reson-rust integration expectations in reson/reson-rust. -->
# Reson Sandbox HA Distributed Contract

Status: Locked v2 (Tier-B Distributed HA Target)  
Owner: Reson Sandbox  
Intent: Define the exact remaining scope from current single-host + local auto-spawn behavior to production-grade distributed HA, without changing the public facade API.

## 1) Locked User-Facing API Contract

The consumer-facing API remains unchanged:

- `Sandbox::new(config)`  
  Local-first, default auto-spawn when no daemon is reachable.
- `Sandbox::connect(endpoint, config)`  
  Explicit connect-only mode to externally managed daemon/control endpoint.
- `Sandbox::session(opts)`
- `Sandbox::attach_session(session_id)`
- `Sandbox::list_sessions()`
- `Session::exec(...)`
- `Session::shell(...)`
- `Session::fork(...)`
- `Session::close()`
- `Session::discard()`

No additional consumer-facing distributed mode API is allowed.

## 2) Deployment Modes (Contracted)

The same facade must support all three:

- Local dynamic daemon mode (default): no endpoint required by user.
- Remote single-host mode: user provides endpoint.
- Remote distributed mode: user provides one control endpoint (or equivalent URL), backend routing is transparent.

Profile boundary for this contract:

- Target production profile is Tier-B distributed HA and is deployable in single-region or multi-region topologies.
- Multi-region behavior is explicit:
  - sessions are region-homed with declared failover target region(s)
  - region failure recovery uses reconnect/rebind + execution-state restore per Tier-B eligibility/SLO
  - no requirement to preserve the same underlying transport socket across region failover

Distributed control endpoint requirement:

- The single distributed endpoint presented to clients MUST be backed by an HA control gateway set (minimum 2 replicas) behind stable service discovery or load balancing.
- Loss of one control gateway instance MUST NOT break active client sessions.
- Production HA readiness for the distributed endpoint MUST satisfy Tier-B continuity semantics (section 7.5 and section 7.7).

## 3) Current State Snapshot

Implemented now:

- Local auto-spawn semantics.
- Durable session identity and fork lineage.
- etcd-backed node and session route lookups.
- vmd node self-registration with etcd lease heartbeat.
- NATS lifecycle event publication (`session.bound`, `session.discarded`).
- Direct node gRPC data path for exec/shell/file operations.
- Shared-mount contract now carries availability, continuity, and backend-profile requirements.
- Distributed placement can now reject nodes that do not advertise the required shared mount
  backend profile before a session is created or rebound.

Not yet implemented:

- MQ-backed cross-node command dispatch.
- MQ-backed reconciliation/repair workers.
- Production-grade multi-node port multiplexing plane (single per-node multiplexer contract with global routing model).
- Full HA failure/recovery verifier set for distributed operation.
- Cross-node continuity for mounted filesystems is only as strong as the selected shared backend;
  the contract exists, but a real shared backend plus failover validation is still required for
  true Tier-B mounted-filesystem readiness.

## 4) Target Distributed Control-Plane Contract

etcd is authoritative state. MQ (NATS) is command/event bus.

### 4.1 etcd keyspace (authoritative)

- `/reson-sandbox/nodes/<node_id>`  
  Node liveness + endpoint + capabilities + generation.
- `/reson-sandbox/sessions/<session_id>`  
  Session to node/vm binding, branch lineage refs, version.
- `/reson-sandbox/forks/<fork_id>`  
  Parent/child IDs, source node, snapshot identity, version.
- `/reson-sandbox/portmaps/<node_id>/<alloc_id>`  
  Port allocation records with lease/owner.
- `/reson-sandbox/reconcile/<task_id>`  
  Reconciliation intents and progress.

All records must include:

- `version`
- `updated_at_unix_ms`
- `owner_node_id` (if applicable)
- idempotency key where needed

### 4.2 MQ subjects (required)

Subject prefix: `reson.sandbox.control`

Required command subjects:

- `reson.sandbox.control.cmd.session.create`
- `reson.sandbox.control.cmd.session.attach`
- `reson.sandbox.control.cmd.session.discard`
- `reson.sandbox.control.cmd.vm.fork`
- `reson.sandbox.control.cmd.port.alloc`
- `reson.sandbox.control.cmd.port.release`
- `reson.sandbox.control.cmd.reconcile.run`

Required event subjects:

- `reson.sandbox.control.evt.session.bound`
- `reson.sandbox.control.evt.session.discarded`
- `reson.sandbox.control.evt.vm.forked`
- `reson.sandbox.control.evt.node.heartbeat`
- `reson.sandbox.control.evt.node.unhealthy`
- `reson.sandbox.control.evt.reconcile.completed`
- `reson.sandbox.control.evt.reconcile.failed`

### 4.3 Control workflow rules

- Every command message must be idempotent.
- Command handlers must write authoritative outcome to etcd.
- Event emit must be at-least-once.
- Consumers must dedupe by message idempotency key.
- Reconciliation workers must converge etcd state and runtime state.
- State transitions MUST use etcd compare-and-swap (CAS) on record version/fence token.

### 4.4 Event/State Atomicity (Outbox Contract)

- Command handling MUST use transactional outbox semantics:
  - one etcd transaction writes authoritative state change and appends outbox record
  - outbox publisher emits MQ event and marks outbox record delivered
  - undelivered outbox records are retried until acknowledged
- Reprocessing the same outbox record MUST be safe (idempotent event identity).
- No direct “write state then fire-and-forget publish” path is allowed in production profile.

### 4.5 MQ Durability and Delivery Profile

- MQ implementation MUST use durable streams (NATS JetStream profile or equivalent).
- Commands:
  - durable consumer
  - explicit ack
  - bounded retry with dead-letter subject
  - per-key ordering for `session_id`/`vm_id` command groups
- Events:
  - at-least-once delivery
  - replayable retention window (minimum 7 days in production profile)
- Backpressure behavior MUST be explicit and observable (queue depth thresholds and rejection policy).

### 4.6 Command/Event Envelope Contract (Locked)

All MQ commands MUST use a canonical envelope:

- `schema_version` (string, required)
- `command_id` (UUIDv7, required, globally unique)
- `command_type` (enum string, required)
- `tenant_id` (string, required)
- `workspace_id` (string, required)
- `session_id` (string, optional)
- `vm_id` (string, optional)
- `target_node_id` (string, optional)
- `ordering_key` (string, required)
- `issued_at_unix_ms` (u64, required)
- `timeout_ms` (u32, required)
- `idempotency_key` (string, required)
- `trace_id` (string, required)
- `causation_id` (string, optional)
- `expected_fence` (object, optional)
- `expected_versions` (map<string,u64>, optional)
- `payload` (object, required)

All MQ events MUST use a canonical envelope:

- `schema_version`
- `event_id` (UUIDv7)
- `event_type`
- `source_command_id` (optional for non-command events)
- `tenant_id`
- `workspace_id`
- `session_id` (optional)
- `vm_id` (optional)
- `node_id` (optional)
- `trace_id`
- `occurred_at_unix_ms`
- `payload`

Schema compatibility rules:

- Additive-only changes allowed within same major schema version.
- Field removals/renames require major schema version bump.
- Consumers MUST ignore unknown fields.

### 4.7 Fencing Token and Ownership Contract (Locked)

Each mutable ownership record MUST include:

- `owner_node_id`
- `owner_lease_id`
- `owner_epoch`
- `record_version`

Ownership transition rules:

- Mutations MUST use etcd transaction compare on `record_version` and current owner fence tuple.
- New owner MUST prove active lease ownership before acquiring.
- Stale owners MUST be rejected on first failed compare and enter reconcile flow.
- Workers MUST never mutate ownership without a current lease + successful CAS.

### 4.8 Reconciliation State Machine (Locked)

Session binding states:

- `Pending`
- `Bound`
- `Draining`
- `Released`
- `Orphaned`
- `Error`

Reconciliation triggers:

- periodic sweep (default every 30s)
- event-driven triggers from MQ
- lease-expiry trigger
- explicit command trigger (`cmd.reconcile.run`)

Convergence rule:

- Reconciliation MUST terminate in a non-transient state or emit terminal `reconcile.failed` with reason code and retry schedule.

### 4.9 etcd Scale and Hotspot Avoidance Contract (Locked)

- Key design MUST avoid single-prefix hotspots under high write rate:
  - session/fork/port records MUST be sharded by stable hash prefix
  - write-heavy keys MUST not share one hot partition path
- Watch consumers MUST support partitioned watches with resumable revision checkpoints.
- etcd compaction/defrag policy MUST be defined and automated for production profile.
- Loss/restart of watch consumers MUST not create correctness gaps; reconcile sweep MUST close missed-event windows.

### 4.10 Shared Mount Continuity Contract (Locked)

Every shared mount attached to a VM/session MUST declare:

- `availability`
  - `NodeLocal`
  - `SharedStorage`
- `continuity`
  - `RestartSameNode`
  - `RestoreCrossNode`
- `backend_profile`

Placement and rebind rules:

- `NodeLocal` mounts MUST NOT claim `RestoreCrossNode`.
- `SharedStorage` mounts MUST declare a non-empty `backend_profile`.
- Nodes MUST advertise the shared mount backend profiles they can satisfy.
- Session create and session rebind MUST filter candidate nodes against required shared mount
  backend profiles before attempting transport recovery.
- `tier_b_eligible=true` sessions MUST require `RestoreCrossNode` shared mounts for any mounted
  filesystem that is expected to survive node failure as part of the session contract.

Readiness rules:

- A mounted filesystem is Tier-A compatible when it can be restarted on the same node and the
  node still satisfies the declared backend contract.
- A mounted filesystem is Tier-B compatible only when the backend is truly cross-node restorable
  and the failover tests prove that continuity on surviving nodes.

## 5) Target Data-Plane Contract

Exec/shell/stdout/stderr remain ordered and canonical per session handle.

Allowed data-plane options:

- Direct node gRPC (dev and trusted-network profile).
- Relay/ingress stream path (required for host-agnostic production profile).

Data-plane must not require etcd writes on every streamed frame.

Production HA data-plane requirements:

- If direct node reachability is not guaranteed for clients, relay/ingress is mandatory.
- Relay routing MUST be sticky per active stream and keyed by session/node binding revision.
- Control-plane reroute MUST NOT reorder stream frames within a single active stream.
- Stream establishment path MUST tolerate control-gateway instance loss.

Stream semantics:

- Per-stream ordering is strict.
- Stream identifiers are monotonic per session handle.
- Reconnect creates a new stream id; replay is not implicit unless explicitly requested by protocol extension.
- Control commands (`discard`, ownership move) during active stream MUST produce deterministic terminal stream event.

### 5.1 Distributed Exec Event Identity + Resume Contract (Locked)

<!-- @dive: Split event identity into global uniqueness (`event_id`) and per-stream order (`event_seq`) so failover resume can be deterministic without global sequencing. -->
- Stream events MUST carry both:
  - `event_id` for global uniqueness (node-independent, cross-cluster safe)
  - `event_seq` for strict monotonic ordering within one logical stream
- Required event envelope fields for distributed exec/shell streams:
  - `cluster_id` (deployment-scoped stable id)
  - `logical_stream_id` (stable across failover/rebind for one logical exec stream)
  - `event_seq` (`u64`, strictly increasing per `logical_stream_id`)
  - `event_id` (recommended `cluster_id` + UUIDv7/ULID form)
  - `event_kind`
  - `producer_epoch` (increments on ownership failover/rebind)
- Ordering rule:
  - Global monotonic ordering across all streams is explicitly not required.
  - Strict monotonicity is required per `logical_stream_id`.
- Resume/dedupe rule:
  - Consumers MUST dedupe/checkpoint on `(cluster_id, logical_stream_id, event_seq)`.
  - Rebind/resume MUST continue from `last_committed_event_seq + 1`.
  - Already delivered events MUST NOT be replayed after rebind.
  - If a terminal event is already committed for `logical_stream_id`, command execution MUST NOT be re-run.

### 5.2 Distributed Exec Producer Reattach Contract (Locked)

<!-- @dive: Tier-B node-loss continuity requires producer reattachment semantics, not only consumer-side event replay filtering. -->
- `L3` continuity for active streams is defined as: unplanned node loss, followed by ownership transfer to a surviving node, VM execution-state recovery, and continuation of the same logical stream.
- Control plane MUST persist stream control state keyed by `(cluster_id, logical_stream_id)` including:
  - `session_id`, `vm_id`, current owner `node_id`, `producer_epoch`
  - `last_committed_event_seq`
  - terminal marker (`terminal_kind`, `terminal_event_seq`, `terminal_event_id`) when committed
- `exec.stream.start` for an existing non-terminal `logical_stream_id` MUST behave as reattach/resume (idempotent continue), not a fresh command rerun.
- Failover rebind sequence MUST be:
  - acquire ownership fence
  - restore VM execution state for Tier-B eligible session
  - reattach producer for the same `logical_stream_id`
  - resume delivery from `last_committed_event_seq + 1`
  - emit `stream.rebinding` and `stream.rebound` control events
- If terminal state is already committed for `logical_stream_id`, recovery MUST return the terminal outcome and MUST NOT rerun command execution.
- If continuation is impossible after bounded recovery attempts, system MUST emit deterministic `stream.failed` terminal outcome and MUST NOT rerun command execution.

## 6) Port Multiplexing Contract (Distributed)

### 6.1 Node-level multiplexing

- Exactly one port multiplexer service per node (not per VM process).
- VM traffic is namespace-routed by `(node_id, vm_id, guest_port, protocol)`.
- Host port uniqueness is required only per node.

### 6.2 Allocation semantics

- Port allocations are lease-backed records in etcd.
- Allocation/release operations are idempotent.
- Restart-safe: multiplexer rebuilds active routes from etcd + runtime discovery.
- Stale allocations are reclaimed by reconciliation.

### 6.3 Capacity constraints

- Exhaustion handling must be explicit error, never silent fallback.
- Port range partitioning and quota enforcement per node must be configurable.

### 6.4 Global Exposure Contract

- Internal host-port uniqueness is node-scoped.
- External access contract MUST expose globally unique endpoint identities via one of:
  - `(node_address, host_port)` tuple
  - globally routed virtual endpoint resolved by ingress/LB
- Tenant scoping MUST prevent cross-tenant endpoint collision and unauthorized route reuse.

### 6.5 Port Multiplexer Protocol Contract

- Node multiplexer exposes a fixed small listener set (not one listener per VM route).
- Incoming connection MUST carry route selector metadata resolving to `(tenant, workspace, vm_id, guest_port, protocol)`.
- Route selector may be conveyed by:
  - authenticated control token + initial control frame
  - mTLS identity + SNI/ALPN mapping
- Selector validation MUST be authz-checked before route activation.
- Route setup latency and failure reason MUST be observable per connection.

### 6.6 Port Access Token Lifecycle (Locked)

- If route selectors use bearer tokens, tokens MUST be:
  - short-lived (default <= 60s TTL)
  - single-route scoped (`tenant/workspace/vm_id/guest_port/protocol`)
  - audience-bound to the target multiplexer service
- Token replay across routes or tenants MUST be rejected.
- Token revocation and lease-expiry checks MUST be enforced before backend dial.

## 7) HA / Failure Semantics

Required behavior:

- Node death: leases expire, node removed from scheduling set.
- In-flight command retry is safe and idempotent.
- Session reattach after control-plane restart uses durable etcd state.
- Parent/child fork lineage remains intact across node and daemon restarts.
- No split-brain ownership for same session binding version.
- Ownership transitions MUST be fenced by lease token + CAS revision checks.

### 7.1 VM Storage Durability and Placement

- The storage model MUST be explicit per deployment profile:
  - local ephemeral node storage (non-HA profile)
  - durable shared/distributed storage (HA profile)
- Running-parent fork fidelity contract (disk + memory snapshot) MUST define:
  - persistence location
  - restoreability after daemon restart
  - behavior after node loss
- If cross-node resume of memory snapshots is unsupported, spec MUST declare this clearly and gate HA claims accordingly.

### 7.2 Scheduling and Admission Control

- Scheduler MUST enforce:
  - CPU/memory/disk quotas per node
  - max active VMs per node
  - max active port allocations per node
- Admission control MUST reject work explicitly when capacity is exceeded.
- Scheduler decisions MUST be observable and reproducible from persisted state.

### 7.3 Disaster Recovery Contract

- etcd backup/restore policy MUST be defined with periodic snapshot cadence.
- Control-plane RPO/RTO targets MUST be defined and tested.
- Restore drills MUST include:
  - session binding recovery
  - fork lineage integrity recovery
  - port allocation consistency recovery

### 7.4 Network Partition Behavior (Locked)

- Control-plane partitions MUST fail closed for ownership mutations.
- Nodes that lose etcd quorum visibility MAY continue serving established local data-plane streams for bounded grace period, but MUST reject new mutating control commands.
- Upon quorum restoration, node MUST revalidate all ownership fences before serving further mutating operations.

### 7.5 VM Continuity Tiering (Locked)

Deployments MUST declare one tier explicitly:

- `Tier-A (Control HA + Disk Durability)`: node loss may interrupt running VM execution; disk/session metadata survive and can recover via restart semantics.
- `Tier-B (Execution Continuity HA)`: node loss supports cross-node resume semantics (including memory snapshot guarantees where claimed).

Claims and SLOs MUST match the selected tier. A deployment may not advertise Tier-B behavior while operating as Tier-A.

Readiness policy:

- `Prod HA Ready` status for distributed host operation REQUIRES Tier-B validation.
- Tier-A is permitted only for:
  - local/dev environments
  - pre-production environments
  - explicit degraded mode during incident response
- When degraded to Tier-A, operations status MUST be explicit and externally visible.

### 7.6 Failure-Domain Placement Contract (Locked)

- Nodes, control gateways, and etcd members MUST publish failure-domain labels (`region`, `zone`, `rack` at minimum).
- Scheduler MUST support anti-affinity constraints for:
  - control gateways
  - high-priority tenant sessions
  - parent/child branches when configured
- HA production profile MUST tolerate loss of any single zone without control-plane unavailability.
- Placement decisions MUST be persisted and auditable.

### 7.7 Tier-B Execution Continuity Contract (Locked)

- Node-loss continuity:
  - For an unplanned single-node failure, active session execution MUST continue on a surviving node without user-initiated reattach.
  - `session_id` and lineage identity MUST remain stable across continuity events.
- In-flight command semantics:
  - Commands accepted before failover MUST resolve exactly once from client perspective (`completed` or deterministic terminal failure).
  - Duplicate side effects are prohibited unless command contract is explicitly idempotent and deduped by idempotency key.
- Stream semantics through failover:
  - Active exec/shell streams MUST recover automatically with preserved per-stream event ordering after reconnect boundary.
  - Recovery MUST use checkpointed per-stream resume (`last_seq + 1`) and MUST NOT replay already-delivered events.
  - Terminal-committed streams MUST not restart command execution during recovery.
  - Continuity recovery MUST emit explicit control events (`stream.rebinding`, `stream.rebound`, `stream.failed`) for observability.
- Planned maintenance continuity:
  - Node drain MUST support proactive handoff for active sessions before process termination.
- Storage/memory fidelity:
  - Tier-B claims require cross-node restorable execution state matching advertised fidelity level.

## 8) Security and Multi-Tenant Boundaries

Required:

- mTLS for node-control and client-control channels in production profile.
- Authn/authz for control commands.
- Tenant/workspace scoping in etcd keys and MQ subjects or payload policy.
- Audit trail for destructive operations (`discard`, `delete`, `fork`).
- Certificate lifecycle operations are required:
  - issuance bootstrap
  - rotation before expiry
  - revocation propagation
- Secrets management for control-plane credentials MUST use managed secret stores in production profile.
- Network policy isolation between tenants MUST be enforced for relay/multiplexer paths.

Operational security requirements:

- RBAC matrix MUST be defined for every command type.
- Authz decisions MUST be auditable with policy version and actor identity.
- Break-glass administrative actions MUST be time-bound and fully audited.

## 9) Observability Contract

Required:

- Structured logs with `session_id`, `vm_id`, `node_id`, `fork_id`, `trace_id`.
- Metrics:
  - command queue latency
  - command success/failure rates
  - reconcile backlog
  - node heartbeat freshness
  - port allocation utilization
  - exec/shell stream setup latency
- Traces across facade -> control command -> node execution path.

Observability quality requirements:

- Metric cardinality budgets MUST be defined to prevent telemetry-induced outages.
- Audit logs MUST be tamper-evident and retained per compliance policy.
- Reconciliation diagnostics MUST expose machine-parsable reason codes.

## 10) SLO Targets (Production Profile)

- P95 session attach: <= 2s
- P95 control command dispatch: <= 500ms
- P99 control command dispatch: <= 1.5s
- P95 exec stream establishment (warm VM): <= 1.5s
- P99 exec stream establishment (warm VM): <= 4s
- P95 execution continuity pause during single-node failure (Tier-B): <= 3s
- P99 execution continuity pause during single-node failure (Tier-B): <= 10s
- Continuity success rate for eligible active sessions under single-node failure (Tier-B): >= 99.9%
- P95 session create from warm pool (Tier-B eligible profile): <= 8s
- P99 session create from warm pool (Tier-B eligible profile): <= 20s
- P95 session create from cold base-image cache hit: <= 30s
- P95 command latency degradation under 80% node saturation (mixed noisy-neighbor workload): <= 2x baseline
- Node failure detection (lease expiry + scheduling exclusion): <= 2x heartbeat TTL
- Reconciliation convergence after node restart: <= 60s for steady-state fleet
- Control endpoint availability: >= 99.95%
- Control-plane RPO: <= 60s
- Control-plane RTO: <= 15m
- Error budget burn is observed and informs manual rollout decisions in this contract revision.

These numbers are acceptance targets and may be revised with measured data.

## 11) Verifier Gates (Distributed HA Extension)

Existing gates remain mandatory. Additional required gates:

- Gate 9: MQ command dispatch contract tests (idempotency + dedupe).
- Gate 10: Reconciliation convergence tests.
- Gate 11: Distributed port multiplexer correctness and reclaim tests.
- Gate 12: Chaos tests (node kill, etcd leader failover, MQ partition simulation).
- Gate 13: Soak test (long-running sessions/forks/forwarding, no leaks).
- Gate 14: SLO compliance smoke in representative environment.
- Gate 15: Outbox atomicity and replay correctness tests.
- Gate 16: Control endpoint failover and stream stickiness tests.
- Gate 17: Storage durability and fork restore fidelity tests across restarts.
- Gate 18: Scheduler/admission correctness and overload rejection tests.
- Gate 19: Disaster recovery restore drill tests (RPO/RTO assertions).
- Gate 20: Security operations tests (mTLS rotation/revocation/authz enforcement).
- Gate 21: Envelope compatibility tests (forward/backward schema evolution).
- Gate 22: Fencing/ownership race tests under concurrent mutators.
- Gate 23: Network partition behavior tests (fail-closed mutation guarantees).
- Gate 25: Load-shedding/fairness tests under multi-tenant burst traffic.
- Gate 26: Fork-chain depth/GC safety tests (no live-branch corruption, no hot-path full-copy regression).
- Gate 27: Failure-domain placement and zonal outage resilience tests.
- Gate 28: Mixed-version rolling upgrade + rollback rehearsal tests.
- Gate 29: Operational game-day tests with runbook execution and MTTR evidence.
- Gate 30: etcd hotspot and watch-scale resilience tests.
- Gate 31: Tier-B unplanned node-failure continuity tests (active exec/shell survives without user reattach).
- Gate 32: Tier-B planned drain handoff tests (no dropped accepted commands).
- Gate 33: In-flight command exactly-once/dedupe correctness across failover tests.
- Gate 34: Tier-B continuity SLO compliance tests.
- Gate 35: Tier-B execution-state fidelity tests (disk+memory continuity for eligible sessions).
- Gate 36: Warm-pool/cold-start SLO tests per architecture and platform profile.
- Gate 39: Full game-day failover drills including continuity + ingress + port multiplexer paths.
- Gate 40: Distributed stream checkpoint/resume tests (`event_seq` monotonicity, no replay after rebind, terminal-no-rerun enforcement).

All enabled production gates must pass for HA readiness claim.

## 12) Work Remaining (Executable Checklist)

<!-- @dive: Scope was narrowed by owner decision; only checklist items below are in execution scope. -->
### 12) Blocking (Current Execution Scope)

- [x] Implement MQ command consumers/producers for session/fork/port lifecycle.
- [x] Implement idempotency keys and dedupe store semantics for command handling.
- [x] Implement reconciliation workers and convergence loops.
- [x] Replace per-session forward helper process model with node-level multiplexer contract.
- [x] Add etcd-backed port allocation records with lease and recovery behavior.
- [x] Add distributed chaos and soak verification gates.
- [x] Add production authn/authz + mTLS profile.
- [x] Add SLO instrumentation and pass/fail thresholds in CI/nightly.
- [x] Add transactional outbox implementation and replay worker.
- [x] Add durable MQ stream configuration, dead-letter handling, and replay tooling.
- [x] Add HA control gateway deployment profile and failover verification.
- [x] Add scheduler/admission controller with explicit capacity rejection semantics.
- [x] Lock storage profile requirements for HA mode, including fork snapshot durability semantics.
- [x] Add DR backup/restore automation and periodic restore drills.
- [x] Add certificate rotation/revocation and secret-management operational playbooks.
- [x] Lock command/event envelope protobuf/json schemas and publish compatibility policy.
- [x] Implement ownership fence tokens end-to-end with race-condition verifier coverage.
- [x] Implement network-partition fail-closed policy and bounded-grace local stream serving behavior.
- [x] Implement and validate Tier-B continuity as default distributed production tier; document Tier-A degraded-mode policy.
- [x] Add control gateway HA topology docs and automated failover drills.
- [x] Implement bounded control-plane queues + deterministic overload signaling (`ResourceExhausted` + retry hint).
- [x] Implement tenant/workspace fairness policy (quota + admission budgeting) with auditable decisions.
- [x] Implement fork-chain depth limits, background compaction, and branch-safe GC.
- [x] Implement failure-domain-aware scheduler policies (anti-affinity + zone spread).
- [x] Add operational runbooks and game-day automation for etcd quorum loss, MQ outage, and zone failure.
- [x] Implement hashed etcd key distribution + partitioned watchers with compaction-safe resume handling.
- [x] Implement continuity orchestration for unplanned node loss (session rebinding + stream recovery).
- [x] Implement planned drain handoff workflow with admission freeze and in-flight command fencing.
- [x] Implement cross-node execution-state restore path required for Tier-B claims.
- [x] Add Tier-B chaos suite for mid-command failover and exactly-once verification.
- [x] Implement execution-state fidelity classifier and policy (`tier_b_eligible` with disk+memory continuity requirement).
- [x] Implement architecture-aware warm pools and prewarmed image pipeline.
- [ ] Implement distributed stream event identity envelope (`cluster_id`, `logical_stream_id`, `event_seq`, `event_id`, `producer_epoch`) end-to-end.
- [ ] Implement checkpointed stream resume (`last_seq + 1`) with strict no-replay and terminal-no-rerun guarantees across failover.
- [ ] Implement control-plane-owned producer reattach flow for active logical streams across unplanned node failover.
- [ ] Enforce per-stream `event_seq` continuity across `producer_epoch` changes (`resume from last checkpoint + 1`).

### Explicitly Out Of Scope For This Execution Pass

<!-- @dive: Owner explicitly removed these from the current contract execution scope. -->
- Staged rollout + automatic rollback policy tied to error-budget burn.
- SLA/performance envelope publishing.
- Per-tenant cgroup/IO/network QoS controls.
- Signed runtime/image manifest verification and SBOM attestation.
- Full platform matrix CI (mac local-dev + linux production parity assertions).

Execution verification snapshot (2026-02-19):

- `make verify-strict` passes with default smoke source (`ghcr.io/bracketdevelopers/uv-builder:main`) and all enabled gates green, including continuity/restore (`27`), mid-command exactly-once failover (`33`), execution-fidelity policy (`35`), and warm-pool prewarm pipeline (`36`).

## 13) Definition Of Done: “Prod HA Ready”

This contract can be declared prod HA ready only when:

- All checklist items in section 12 are complete.
- All enabled gates in this contract revision pass in CI with production profile inputs.
- No facade API changes are required from current locked user surface.
- Distributed mode remains transparent to end users once they target the distributed host endpoint.
- Deployment continuity tier and SLO claims are consistent and externally documented.
- Tier-B continuity requirements are validated in production profile; Tier-A usage is restricted to explicit degraded/local/pre-prod contexts.

Until then, status is “Distributed Control Partial”.

## 14) Gap Resolutions (Locked)

The following review gaps are now resolved by explicit contract:

- Gap 1 (event/state atomicity): solved by section 4.4 transactional outbox contract.
- Gap 2 (VM storage durability and fork recoverability): solved by section 7.1 storage/restore contract.
- Gap 3 (data-plane HA under variable network topology): solved by section 5 production relay requirement.
- Gap 4 (single control endpoint SPOF): solved by section 2 HA control gateway requirement.
- Gap 5 (MQ durability semantics): solved by section 4.5 durable stream profile.
- Gap 6 (split-brain ownership): solved by sections 4.3 and 7 CAS + fence-token ownership rules.
- Gap 7 (scheduler/admission policy): solved by section 7.2 scheduler/admission contract.
- Gap 8 (disaster recovery expectations): solved by section 7.3 DR contract and section 10 RPO/RTO targets.
- Gap 9 (global port/service exposure semantics): solved by section 6.4 global exposure contract.
- Gap 10 (operational security completeness): solved by section 8 cert lifecycle, revocation, and secret isolation requirements.
- Gap 11 (port route token replay risk): solved by section 6.6 token lifecycle contract.
- Gap 12 (failure-domain blast radius): solved by section 7.6 placement/anti-affinity contract.
- Gap 13 (overload fairness and unbounded queue risk): solved by section 16 load-shedding/fairness contract.
- Gap 14 (fork-chain lifecycle safety over time): solved by section 17 fork-chain GC/compaction contract.
- Gap 15 (etcd hot-key/watch scalability risk): solved by section 4.9 sharding + watch-resume contract.
- Gap 16 (ambiguous Tier-B requirement vs optional tiering): solved by section 7.5 readiness policy.
- Gap 17 (failover continuity semantics under node loss): solved by section 7.7 continuity contract, section 10 Tier-B SLOs, and gates 31-34.
- Gap 18 (Tier-B execution-state fidelity ambiguity): solved by section 20 fidelity contract and gate 35.
- Gap 19 (cold-start and base-image readiness ambiguity): solved by section 21 warm-pool/image contract and gate 36.
- Gap 20 (multi-tenant noisy-neighbor performance collapse risk): explicitly out of scope for this execution pass.
- Gap 21 (runtime/image trust chain ambiguity): explicitly out of scope for this execution pass.
- Gap 22 (platform behavior drift between mac local-dev and linux prod): explicitly out of scope for this execution pass.

## 15) Upgrade and Compatibility Contract (Locked)

Rolling upgrade requirements:

- Control gateways, workers, and node daemons MUST support N and N-1 control envelope schema versions.
- Mixed-version cluster operation MUST preserve correctness for ownership fencing and idempotency.
- Upgrade order MUST be documented and validated in verifier gates:
  - gateways first
  - control workers
  - node daemons
  - optional cleanup migration

Rollback requirements:

- Rollback from N to N-1 MUST be supported without data loss for authoritative etcd records.
- Forward-only irreversible migrations are prohibited unless guarded by explicit maintenance mode and backup checkpoint.

## 16) Scale, Fairness, and Load-Shedding Contract (Locked)

- No unbounded in-memory queues are allowed for:
  - command ingress
  - reconcile work
  - stream fan-out buffering
- Admission control MUST enforce per-tenant and per-workspace budgets for:
  - concurrent sessions
  - command QPS
  - long-running stream count
- Overload behavior MUST be deterministic:
  - reject with explicit typed error
  - include retry hint (`retry_after_ms`)
  - never silently drop accepted commands
- Priority classes MUST be explicit; destructive cleanup and ownership-recovery operations MUST not starve behind best-effort work.

## 17) Fork-Chain Lifecycle Contract (Locked)

- Fork lineage MUST track ancestry depth and live descendant counts.
- Maximum CoW chain depth MUST be configurable; when exceeded, system MUST perform controlled compaction/rebase outside hot command path.
- Compaction/rebase MUST preserve branch isolation invariants and lineage identity.
- GC eligibility for snapshot/overlay artifacts requires:
  - no active VM process references
  - no live descendant references
  - no retention-policy hold
- Fork/compaction operations MUST remain O(metadata/overlay ops) on the request path; full image copy is forbidden on request path.

## 18) Operational Readiness Contract (Locked)

- Production profile MUST include runbooks for:
  - etcd quorum degradation/loss
  - MQ unavailability/degradation
  - zone failure
  - certificate expiry/revocation incident
- Every paging alert MUST map to a runbook entry with owner and review cadence.
- Node drain and maintenance mode MUST be first-class control operations with auditable state transitions.
- Game-day drills MUST execute at least quarterly and produce remediation actions tracked to closure.

## 19) Deferred Topic: Performance Envelope (Out Of Scope)

- Performance-envelope publication is intentionally out of scope for this execution pass.
- Runtime correctness and failure semantics remain in scope.

## 20) Tier-B Execution-State Fidelity Contract (Locked)

- Every session class MUST declare continuity eligibility:
  - `tier_b_eligible=true`: requires disk+memory continuity across single-node failure.
  - `tier_b_eligible=false`: continuity is best-effort and excluded from Tier-B SLO numerator.
- Tier-B eligibility policy MUST be explicit and auditable per session.
- For `tier_b_eligible=true`, recovered execution state MUST preserve:
  - in-memory process state
  - open interactive shell/session context
  - command idempotency fence state
- If full disk+memory continuity is impossible for a session class, that class MUST NOT be counted toward Tier-B success metrics.

## 21) Cold-Start and Warm-Pool Contract (Locked)

- Base image pipeline MUST produce architecture-specific prepared artifacts ahead of first session:
  - image fetched
  - verified
  - converted/prepared
  - cached
- Warm pools MUST be maintained per `(architecture, image_profile)` with configurable minimum inventory.
- Session creation path for warm pool hits MUST avoid image conversion work on request path.
- Warm pool depletion MUST trigger asynchronous refill and emit explicit capacity metrics/events.
- Local auto-spawn mode MUST include endpoint prewarm so first user command does not observe repeated transport-reset spam.

## 22) Deferred Topic: Per-Tenant QoS Controls (Out Of Scope)

- Per-tenant cgroup/IO/network QoS controls are intentionally out of scope for this execution pass.
- Queue/admission fairness semantics in section 16 remain in scope.

## 23) Deferred Topic: Runtime Supply-Chain Attestation (Out Of Scope)

- Signed runtime/image manifest verification and SBOM publication are intentionally out of scope for this execution pass.

## 24) Deferred Topic: Full Platform Matrix CI (Out Of Scope)

- Full mac/linux platform-matrix CI parity assertions are intentionally out of scope for this execution pass.
- Local-first UX requirements remain unchanged.

codex has certified this plan as "production-viable for normal hyperscaler tolerances"
