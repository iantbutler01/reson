<!-- @dive-file: Failure triage mapping for real gates 41-48 to likely causes, rerun commands, and remediation runbooks. -->
<!-- @dive-rel: Supports specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md section 7.7 operationalization goals. -->
<!-- @dive-rel: Used by on-call/developers when scripts/verify_strict_real.sh fails in CI or local profiles. -->
# Real Gate Failure Triage Playbook

## Overview

This playbook maps each real gate (`41-48`) to:

- the command to rerun in isolation
- most likely failure classes
- first remediation actions

## Gate Map

| Gate | Isolated command | Primary intent |
|---|---|---|
| 41 | `./scripts/verify_real_gate41.sh --profile <profile>` | L2 active-stream continuity on node loss |
| 42 | `./scripts/verify_real_gate42.sh --profile <profile>` | Tier-B restore-marker rehydrate + resume |
| 43 | `./scripts/verify_real_gate43.sh --profile <profile>` | in-flight exactly-once failover semantics |
| 44 | `./scripts/verify_real_gate44.sh --profile <profile>` | warm-pool/prewarm pipeline checks |
| 45 | `./scripts/verify_real_gate45.sh --profile <profile>` | real control-plane fail-closed/recovery + planned drain handoff |
| 46 | `./scripts/verify_real_gate46.sh --profile <profile>` | full distributed failover continuity suite |
| 47 | `./scripts/verify_real_gate47.sh --profile <profile>` | leak/orphan process checks after distributed failover |
| 48 | `./scripts/verify_real_gate48.sh --profile <profile>` | stream identity/checkpoint/no-replay semantics |

## Triage Steps

1. Re-run the failing gate in isolation with `--keep-stack`.
2. Collect artifacts:
`./scripts/integration/collect-artifacts.sh --profile <profile> --reason gate-<N>-failure`
3. Inspect node logs and control-plane snapshots under `.integration-artifacts/...`.
4. Confirm the profile file and host capabilities from `specs/RESON_SANDBOX_REAL_CI_REQUIREMENTS.md`.
5. Apply remediation by failure class below, then rerun isolated gate.

## Failure Classes and First Actions

`transport/readiness failures`

- Symptoms: connection reset, timeout waiting for events, node not ready.
- Actions:
1. Verify compose services are healthy: `./scripts/integration/up.sh --profile <profile>`
2. Verify ports are free; rerun with clean state:
`./scripts/integration/reset.sh --profile <profile>`
3. Re-run isolated gate.

`ownership fence / route conflicts`

- Symptoms: `ownership_fence_conflict`, stale-route assertions.
- Actions:
1. Confirm etcd route/fence keys in artifacts.
2. Re-run gate 45 or 48 in isolation.
3. Validate fence CAS logic against `specs/runbooks/PLANNED_DRAIN_HANDOFF_AND_FENCING.md`.

`tier-b restore continuity failures`

- Symptoms: missing restore marker, resume attach target missing, no-rerun failures.
- Actions:
1. Re-run gate 42 and gate 48 selectors in isolation.
2. Validate marker expectations against `specs/runbooks/TIERB_EXECUTION_STATE_RESTORE.md`.
3. Validate exactly-once/no-rerun against `specs/runbooks/TIERB_MID_COMMAND_FAILOVER_EXACTLY_ONCE.md`.

`mq retry/dlq failures`

- Symptoms: expected DLQ message absent, dedupe mismatch, retry count mismatch.
- Actions:
1. Re-run gate 46 (or selector-specific failover test) and gate 48.
2. Inspect NATS stream subjects and DLQ payload in artifacts.
3. Use replay tooling if needed: `./scripts/replay_mq_dead_letters.sh <limit>`.

`process leak failures`

- Symptoms: gate 47 reports increased `qemu`, `portproxy`, or `vmd` count.
- Actions:
1. Confirm baseline vs post-run process snapshots in artifacts.
2. Re-run gate 47 standalone with higher grace:
`REAL_GATE47_LEAK_GRACE_SECONDS=10 ./scripts/verify_real_gate47.sh --profile <profile>`
3. Investigate teardown paths in integration scripts and runtime shutdown hooks.
