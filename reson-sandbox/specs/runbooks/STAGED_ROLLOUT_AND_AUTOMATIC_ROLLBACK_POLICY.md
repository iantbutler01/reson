# Staged Rollout And Automatic Rollback Policy

Status: Locked  
Scope: Distributed production rollouts for control gateways, workers, and node daemons.

## 1) Rollout Stages

Default staged progression:

1. `canary` (1%-5% traffic / limited nodes)
2. `stage_1` (10%-25%)
3. `stage_2` (50%)
4. `full` (100%)

Advancement requires:

- no unresolved critical alerts
- gateway failover drill pass
- SLO/error-budget evaluation pass for current stage window

## 2) Error-Budget Policy

- Evaluation window: 7-day projected burn.
- Rollout pause threshold: projected burn > 100% of budget.
- Automatic rollback threshold: projected burn >= configured rollback threshold (default 120%).
- Any stage crossing rollback threshold MUST trigger immediate rollback to previous stable version.

## 3) Automatic Rollback Contract

When rollback triggers:

- Mark rollout state `rollback_required=true`.
- Freeze stage advancement.
- Route traffic back to prior stable deployment revision.
- Emit machine-readable status artifact for incident/change review.

## 4) Automation Entry Point

- Script: `scripts/evaluate_rollout_policy.sh`
- Inputs:
  - threshold policy JSON (`specs/RESON_SANDBOX_SLO_THRESHOLDS.json`)
  - observed rollout metrics JSON (`RESON_SANDBOX_ROLLOUT_OBSERVED_FILE` in CI/nightly)
- Output:
  - pass/fail exit code
  - policy decision fields (`pause_rollout`, `rollback_required`)

## 5) Operational Notes

- Local strict runs may omit observed rollout file and only validate schema + policy wiring.
- Production CI/nightly MUST provide observed rollout metrics and enforce evaluator decision.
