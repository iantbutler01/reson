<!-- @dive-file: Runbook for Tier-B mid-command failover semantics with exactly-once dispatch validation. -->
<!-- @dive-rel: Enforced by scripts/verify_tierb_exactly_once.sh for checklist item 12.33. -->
<!-- @dive-rel: Backed by contract test inflight_exec_rebinds_on_rpc_loss_and_runs_exactly_once in crates/reson-sandbox/tests/facade_contract.rs. -->
# Tier-B Mid-Command Failover Exactly-Once Runbook

Status: Locked  
Scope: Section 12 item `12.33` Tier-B chaos suite for mid-command failover + exactly-once.

## 1) Objective

Validate that command dispatch during endpoint loss is recovered by session rebinding and that user command dispatch is not duplicated across primary/secondary failover.

## 2) Scenario

1. Create an active session on primary endpoint.
2. Mirror VM metadata onto secondary endpoint for continuity candidate selection.
3. Simulate in-flight RPC path loss by terminating the primary guest RPC endpoint (portproxy) while preserving secondary path.
4. Issue command through facade session handle.
5. Verify facade rebinds and command exits successfully.
6. Verify command invocation count across nodes equals exactly one.

## 3) Verification

Run:

```bash
./scripts/verify_tierb_exactly_once.sh --strict
```

The gate performs static wiring checks and executes:

- `cargo test -p reson-sandbox --test facade_contract inflight_exec_rebinds_on_rpc_loss_and_runs_exactly_once -- --nocapture`

## 4) Pass Criteria

- Gate exits `0`.
- Rebind helper path is present in facade (`rebind_session_endpoint` and rebind-candidate transport error classification).
- Contract test confirms:
  - command succeeds after failover
  - command dispatch count across nodes is exactly one
