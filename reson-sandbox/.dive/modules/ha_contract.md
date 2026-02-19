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
