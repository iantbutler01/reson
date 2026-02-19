<!-- @dive-file: Runbook for tier_b_eligible execution-state fidelity classification and enforcement policy. -->
<!-- @dive-rel: Enforced by scripts/verify_execution_fidelity_policy.sh for checklist item 12.34. -->
<!-- @dive-rel: Implemented in crates/reson-sandbox/src/lib.rs metadata classifier and failover restore policy checks. -->
# Tier-B Execution Fidelity Policy Runbook

Status: Locked  
Scope: Section 12 item `12.34` execution-state fidelity classifier and policy.

## 1) Objective

Guarantee that every session is explicitly classified for Tier-B SLO accounting and that Tier-B-eligible failover requires disk+memory continuity markers.

## 2) Classification Contract

At session creation, facade must persist:

- `reson.tier_b_eligible`: `true` or `false`
- `reson.execution_fidelity_requirement`:
  - `disk+memory` when `tier_b_eligible=true`
  - `best-effort` when `tier_b_eligible=false`

Default when unset: `tier_b_eligible=true`.

## 3) Enforcement Contract

During failover rebinding:

- If `tier_b_eligible=true`, restore marker is mandatory (`execution_restore_snapshot_id/name` or fork snapshot fallback).
- Missing marker must fail with explicit policy error.
- If `tier_b_eligible=false`, missing marker does not violate policy.

## 4) Verification

Run:

```bash
./scripts/verify_execution_fidelity_policy.sh --strict
```

The gate executes:

- `tier_b_eligibility_classifier_sets_metadata_policy`
- `tier_b_eligible_failover_requires_restore_snapshot_marker`

## 5) Pass Criteria

- Gate exits `0`.
- Classification metadata is written deterministically.
- Tier-B eligible failover without restore marker fails explicitly.
