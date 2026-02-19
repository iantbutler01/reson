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

## 6) Gate Migration Matrix (Locked)

- Current `gate 25` (Tier-B continuity orchestration) -> add real variant `gate 41` (real node-loss continuity).
- Current `gate 27` (execution-state restore) -> add real variant `gate 42` (real restore marker + recovered execution verification).
- Current `gate 33` (exactly-once failover) -> add real variant `gate 43` (real in-flight command failover dedupe).
- Current `gate 36` (warm-pool pipeline) -> add real variant `gate 44` (real prewarm/cold-hit timing and refill behavior).
- Current `gate 11/12` (chaos/soak) -> add real variants `gate 45/46` (control-plane failure drills on live cluster).
- Current `gate 6` (leak) -> add real variant `gate 47` (cross-node leak and orphan process checks after chaos).

## 7) Implementation Checklist

### 7.1 Harness Foundation

- [x] Create `scripts/integration/` harness entrypoints for environment lifecycle (`up`, `down`, `reset`, `collect-artifacts`).
- [x] Add canonical compose manifests for control-plane services under `deploy/integration/`.
- [x] Add per-profile config templates (`local-dev`, `ci-linux-kvm`, `nightly-soak`).
- [x] Add machine-readable artifact bundle format (JSON + logs + process snapshots).

### 7.2 Real Cluster Wiring

- [ ] Launch two real `vmd` nodes against shared control plane and verify registration/heartbeat in etcd.
- [ ] Route facade to control gateway endpoints in integration tests (no direct test-only shortcuts).
- [x] Add health/ready waiters with bounded timeout and explicit failure reasons.

### 7.3 Real Continuity + Exactly-Once Coverage

- [ ] Add real test: unplanned primary-node loss during active command stream, client session continues on secondary.
- [ ] Add real test: in-flight command acknowledged exactly once from client perspective under failover.
- [ ] Add real test: Tier-B eligible session without restore marker fails policy as expected.
- [ ] Add real test: Tier-B eligible session with restore marker rehydrates and resumes command flow.

### 7.4 Real Warm-Pool/Cool-Start Coverage

- [ ] Add real test: startup prewarm for configured architectures/images executes before first session.
- [ ] Add real test: warm-pool hit path avoids image conversion/download on request path.
- [ ] Add real test: cold hit triggers async refill and emits refill evidence.
- [ ] Add real test: local auto-spawn prewarm avoids first-command transport-reset spam.

### 7.5 Real Control-Plane Failure Coverage

- [ ] Add real test: etcd quorum degradation blocks mutations fail-closed and recovers after quorum return.
- [ ] Add real test: NATS outage produces deterministic overload/retry signaling and drains cleanly on recovery.
- [ ] Add real test: zone/node drain handoff with admission freeze preserves in-flight command guarantees.
- [ ] Add real test: ownership fence conflicts under concurrent mutators resolve deterministically.

### 7.6 Leak and Stability Coverage

- [ ] Add real test: repeated session/fork/failover cycles leave no orphan qemu/portproxy/vmd processes.
- [ ] Add real test: long-run churn workload maintains bounded resource growth.
- [ ] Add real test: reconcile convergence bound remains within contract target after node restart.

### 7.7 Gate Orchestration and CI

- [ ] Add gates `41-47` scripts and wire them into strict orchestration with profile selection.
- [ ] Keep mock gates for fast preflight; require real gates in `verify-strict-real` and release CI.
- [ ] Add explicit CI docs listing required runner capabilities (`docker`, `docker compose`, `/dev/kvm` for KVM profile).
- [ ] Publish failure triage playbook mapping each new gate to remediation runbook.

## 8) Completion Criteria

This plan is done when:

- each Tier-B claim has at least one passing real-machinery gate
- `verify-strict-real` passes on `profile.ci-linux-kvm`
- local developers can run a bounded subset via `profile.local-dev`
- artifacts from failed gates are sufficient for deterministic diagnosis
