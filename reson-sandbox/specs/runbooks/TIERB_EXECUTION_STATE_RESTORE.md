<!-- @dive-file: Operational contract for cross-node execution-state restore in Tier-B continuity workflows. -->
<!-- @dive-rel: Validated by scripts/verify_tierb_execution_restore.sh for section 12.32 completion evidence. -->
<!-- @dive-rel: Documents restore semantics implemented in crates/reson-sandbox/src/lib.rs. -->
# Tier-B Execution-State Restore Runbook

Status: Locked  
Scope: Section 12 item `12.32` cross-node execution-state restore path.

## 1) Objective

Guarantee that continuity handoff can restore execution state on a surviving node before session resume.

## 2) Restore Markers

A VM/session is restore-candidate when metadata includes one of:

- `reson.execution_restore_snapshot_id`
- `reson.execution_restore_snapshot_name`
- `reson.fork_snapshot` (name fallback)

Resolution order:

1. Explicit snapshot ID marker.
2. Snapshot name marker resolved against VM snapshot list.
3. No marker => no restore attempt.

## 3) Handoff Behavior

During endpoint rebinding / reattach:

1. Resolve surviving node endpoint.
2. Detect restore marker from VM metadata.
3. Call `RestoreSnapshot(vm_id, snapshot_id)` on target node.
4. Start/resume VM only after restore call succeeds.

## 4) Verification

Run:

```bash
./scripts/verify_tierb_execution_restore.sh --strict
```

The gate validates static wiring and executes continuity test `continuity_rebinds_session_after_primary_vmd_loss`, which asserts restore invocation on rebound node.

## 5) Pass Criteria

- Verifier exits `0`.
- Restore marker resolution code is present.
- Continuity rebind test confirms at least one restore call on secondary node.
