<!-- @dive-file: Execution plan to replace mock-heavy verification paths with real integration tests across control-plane and VM runtime machinery. -->
<!-- @dive-rel: Extends HA readiness guarantees from specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md with higher-fidelity evidence. -->
<!-- @dive-rel: Defines new verifier gates to be orchestrated by scripts/verify_reson_sandbox.sh and CI profiles. -->
# Reson Sandbox Real Integration Testing Plan (Locked Draft)

## 1) Objective

Raise confidence from "contract-correct in mock harnesses" to "behavior-correct on real distributed machinery" without changing the locked facade API.

## 2) Why This Plan Exists

The current gate suite is strong on API/contract logic, but many distributed claims are still validated through mock gRPC harnesses.

This plan upgrades evidence quality for:

- node-loss continuity
- exactly-once behavior under failover
- execution-state restore semantics
- distributed stream checkpoint/resume semantics (`event_seq` monotonicity and no replay)
- warm-pool and cold-start behavior
- distributed control-plane behavior under real etcd/NATS failure modes

## 3) Scope

In scope:

- real integration environments for local and CI profiles
- replacement/augmentation of mock-backed gates for Tier-B claims
- real control-plane chaos and reconciliation assertions
- real process and resource leak assertions

Out of scope:

- changing public Rust facade surface area
- redesigning runtime architecture
- removing all unit/mocked tests (they remain for fast feedback)

## 4) Testing Principles (Locked)

- Any production claim in the contract must have at least one real-machinery test path.
- Mock tests are allowed for fast determinism but cannot be the only proof for Tier-B/HA claims.
- Real tests must assert externally visible outcomes (API responses, state convergence, process lifecycle), not only internal logs.
- Gate failures must be attributable with machine-readable artifacts.

## 5) Environment Profiles (Locked)

`profile.local-dev`

- target: mac/linux developer machine
- control plane: docker compose (`etcd` quorum + `nats`)
- node plane: 1-2 `vmd` nodes
- purpose: fast reproducible checks, non-KVM-specific semantics

`profile.ci-linux-kvm`

- target: self-hosted Linux runner with `/dev/kvm`
- control plane: docker compose
- node plane: 2+ `vmd` nodes with real QEMU/KVM
- purpose: authoritative runtime and continuity evidence

`profile.nightly-soak`

- target: persistent Linux environment
- duration: long-running churn/soak
- purpose: leak, stability, and convergence confidence

### 5.1 Abstraction Lane Matrix (Locked)

Test coverage is partitioned into three API abstraction lanes:

- `L1` local dynamic mode: no host provided, facade auto-spawns local daemon.
- `L2` host provided, single-host mode: explicit endpoint, no distributed control plane requirement.
- `L3` host provided, distributed mode: explicit control endpoint with etcd + MQ-backed control plane.

Inheritance rule:

- `L2` MUST pass all `L1` behavioral assertions in connect-only mode.
- `L3` MUST pass all `L2` behavioral assertions plus distributed-control assertions.
- No lane can skip lower-lane behavior and still be considered production-ready.

Lane matrix:

| Lane | Required core coverage |
|---|---|
| `L1` | auto-spawn readiness, session create/attach/list/discard, exec/shell bidi ordering, fork invariants, local daemon restart + durable reattach |
| `L2` | all `L1` behavior via explicit endpoint, single-node fail-stop/reconnect behavior, host-mode warm/cold path checks |
| `L3` | all `L2` behavior + etcd registry/route/fence CAS, MQ command delivery/retry/DLQ, reconcile convergence, cross-node failover semantics |

### 5.2 Profile-to-Lane Mapping

- `profile.local-dev` must cover `L1`, `L2`, and bounded `L3` smoke.
- `profile.ci-linux-kvm` must cover authoritative `L2` + `L3` gates.
- `profile.nightly-soak` must cover long-running `L3` churn/leak/convergence.

## 6) Gate Migration Matrix (Locked)

- Current `gate 25` (Tier-B continuity orchestration) -> add real variant `gate 41` (real node-loss continuity).
- Current `gate 27` (execution-state restore) -> add real variant `gate 42` (real restore marker + recovered execution verification).
- Current `gate 33` (exactly-once failover) -> add real variant `gate 43` (real in-flight command failover dedupe).
- Current `gate 36` (warm-pool pipeline) -> add real variant `gate 44` (real prewarm/cold-hit timing and refill behavior).
- Current `gate 11/12` (chaos/soak) -> add real variants `gate 45/46` (control-plane failure drills on live cluster).
- Current `gate 6` (leak) -> add real variant `gate 47` (cross-node leak and orphan process checks after chaos).
- New distributed stream identity/resume contract -> add real variant `gate 48` (global `event_id` uniqueness + per-stream `event_seq` checkpoint/no-replay/terminal-no-rerun behavior).

## 7) Implementation Checklist

### 7.1 Harness Foundation

- [x] (`L1/L2/L3`) Create `scripts/integration/` harness entrypoints for environment lifecycle (`up`, `down`, `reset`, `collect-artifacts`).
- [x] (`L3`) Add canonical compose manifests for control-plane services under `deploy/integration/`.
- [x] (`L1/L2/L3`) Add per-profile config templates (`local-dev`, `ci-linux-kvm`, `nightly-soak`).
- [x] (`L1/L2/L3`) Add machine-readable artifact bundle format (JSON + logs + process snapshots).

### 7.2 Real Cluster Wiring

- [x] (`L3`) Launch two real `vmd` nodes against shared control plane and verify registration/heartbeat in etcd.
- [x] (`L2/L3`) Route facade to control gateway endpoints in integration tests (no direct test-only shortcuts).
- [x] (`L1/L2/L3`) Add health/ready waiters with bounded timeout and explicit failure reasons.

### 7.3 Real Continuity + Exactly-Once Coverage

- [x] (`L2`) Add real test: unplanned primary-node loss during active command stream, client session continues on secondary.
- [x] (`L2`) Add real test: in-flight command acknowledged exactly once from client perspective under failover.
- [x] (`L2/L3`) Add real test: Tier-B eligible session without restore marker fails policy as expected.
- [x] (`L2/L3`) Add real test: Tier-B eligible session with restore marker rehydrates and resumes command flow.
- [x] (`L3`) Add distributed failover test variant: control bus enabled, route/fence state transitions verified in etcd during failover.
- [x] (`L3`) Add distributed failover test variant: MQ command retry/dedupe/dead-letter behavior validated under node loss.
- [x] (`L2/L3`) Add real test: rebind resumes stream from `last_committed_event_seq + 1` with no replay of previously delivered events.
- [x] (`L3`) Add real test: stream event envelope includes `cluster_id`, `logical_stream_id`, `event_seq`, `event_id`, and `producer_epoch`; `event_id` is globally unique across node failover.
- [x] (`L2/L3`) Add real test: if a terminal event is already committed for a logical stream, failover recovery does not re-run the command.
- [x] (`L3`) Add real test: active distributed stream survives unplanned node loss via producer reattach on secondary without command rerun.
- [x] (`L3`) Add real test: after producer-epoch change on failover, resumed stream emits `event_seq` starting at `last_committed_event_seq + 1`.
- [x] (`L3`) Add real test: rebind path for non-terminal logical stream does not dispatch a fresh command start (side-effect counter remains `1`).

<!-- @dive: These tests close the practical duplicate-side-effect gap by proving checkpointed forward-only resume across node loss. -->

Status note:

- Real distributed selectors now cover stream checkpoint/no-replay, identity envelope, producer-epoch continuity, terminal no-rerun, and MQ retry/dedupe/dead-letter under node loss.
- Warm-pool/cold-start real selectors now cover startup prewarm, warm-hit request path, cold-hit async refill evidence, and local auto-spawn first-call transport readiness.

### 7.4 Real Warm-Pool/Cool-Start Coverage

- [x] (`L1`) Add real test: startup prewarm for configured architectures/images executes before first session.
- [x] (`L1/L2`) Add real test: warm-pool hit path avoids image conversion/download on request path.
- [x] (`L1/L2`) Add real test: cold hit triggers async refill and emits refill evidence.
- [x] (`L1`) Add real test: local auto-spawn prewarm avoids first-command transport-reset spam.

### 7.5 Real Control-Plane Failure Coverage

- [x] (`L3`) Add real test: etcd quorum degradation blocks mutations fail-closed and recovers after quorum return.
- [x] (`L3`) Add real test: NATS outage produces deterministic overload/retry signaling and drains cleanly on recovery.
- [x] (`L3`) Add real test: zone/node drain handoff with admission freeze preserves in-flight command guarantees.
- [x] (`L3`) Add real test: ownership fence conflicts under concurrent mutators resolve deterministically.

### 7.6 Leak and Stability Coverage

- [x] (`L2/L3`) Add real test: repeated session/fork/failover cycles leave no orphan qemu/portproxy/vmd processes.
- [x] (`L3`) Add real test: long-run churn workload maintains bounded resource growth.
- [x] (`L3`) Add real test: reconcile convergence bound remains within contract target after node restart.

### 7.7 Gate Orchestration and CI

- [x] (`L1/L2/L3`) Add gates `41-47` scripts and wire them into strict orchestration with profile selection.
- [x] (`L2/L3`) Add `gate 48` script for stream identity/checkpoint semantics and wire into `verify-strict-real`.
- [x] (`L1/L2/L3`) Keep mock gates for fast preflight; require real gates in `verify-strict-real` and release CI.
- [x] (`L1/L2/L3`) Add explicit CI docs listing required runner capabilities (`docker`, `docker compose`, `/dev/kvm` for KVM profile).
- [x] (`L1/L2/L3`) Publish failure triage playbook mapping each new gate to remediation runbook.

Status note:

- `verify_strict_real.sh` now runs mock preflight + mandatory real gates `41-48`; `.github/workflows/verify-strict-real.yml` enforces this path on self-hosted Linux/KVM runners.

## 8) Completion Criteria

This plan is done when:

- each Tier-B claim has at least one passing real-machinery gate
- `verify-strict-real` passes on `profile.ci-linux-kvm`
- local developers can run a bounded subset via `profile.local-dev`
- artifacts from failed gates are sufficient for deterministic diagnosis
