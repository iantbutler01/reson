<!-- @dive-file: Runbook for planned drain handoff, admission freeze, and ownership fencing behavior. -->
<!-- @dive-rel: Backed by scripts/verify_planned_drain_handoff.sh and real planned-drain integration tests. -->
<!-- @dive-rel: Documents operator procedure for safe node drain without duplicate side effects. -->
# Planned Drain Handoff And Fencing Runbook

Status: Locked  
Scope: Section 12 item `12.31` planned node drain with admission freeze and in-flight command fencing.

## 1) Objective

Drain a node without accepting new sessions while preserving correctness for accepted/in-flight commands.

## 2) Controls

- Admission freeze flag: node registry payload field `admission_frozen=true`.
- CLI switch: `vmd --admission-frozen`.
- Env switch: `RESON_SANDBOX_NODE_ADMISSION_FROZEN=true`.
- Ownership fencing: control commands carry `expected_fence` and are rejected on stale fence.

## 3) Procedure

1. Mark the target node as frozen (`admission_frozen=true`).
2. Confirm scheduler excludes frozen nodes for new session placement.
3. Keep existing sessions attachable while handoff progresses.
4. Enforce ownership-fence checks so stale in-flight mutators are rejected.
5. Rebind sessions to healthy nodes (`attach_session`) before decommissioning drained node.

## 4) Automated Verification

Run:

```bash
./scripts/verify_planned_drain_handoff.sh --strict
```

The verifier checks:

- frozen-node scheduling exclusion,
- continuity attach/rebind contract test,
- ownership-fence stale-transition rejection test.

## 5) Pass Criteria

- Verifier exits `0`.
- Frozen nodes are never selected for new admissions.
- Rebound attach succeeds after endpoint loss.
- Stale fence transitions are rejected deterministically.
