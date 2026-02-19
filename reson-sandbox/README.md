# reson-sandbox

Sandbox runtime subproject for Reson.

This subproject is the extraction target for Rust VM sandbox runtime components currently implemented in OpenBracket (`vmd`, `portproxy`, and their proto contracts). It provides a stable dependency boundary for `reson-rust` via the high-level `reson-sandbox` facade crate.

## Current Status

Runtime code and facade crate are present with strict verifiers.

## Verification Entry Points

- `make verify` - Local verification with optional runtime gates
- `make verify-strict` - Enforces all gates (build/lint/test/proto/e2e/leaks/facade/fork)
- `make verify-e2e` - Runs strict verification with e2e enabled
- `make verify-fork` - Runs strict CoW fork gate directly
- `make verify-api` - Runs strict facade API gate directly
- `make verify-security` - Runs strict authn/authz + TLS security profile gate directly
- `make verify-slo` - Runs strict SLO threshold/instrumentation gate directly
- `make evaluate-rollout-policy` - Evaluates rollout pause/rollback policy from observed error-budget burn data
- `make verify-gateway` - Runs strict control-gateway failover gate directly
- `make drill-gateway-failover` - Runs the automated control-gateway failover drill and writes a report artifact
- `make verify-storage` - Runs strict storage-profile durability gate directly
- `make verify-admission` - Runs strict scheduler/admission-capacity gate directly
- `make verify-dr` - Runs strict DR backup/restore drill gate directly
- `make verify-security-ops` - Runs strict cert-rotation/revocation + secret-ops gate directly
- `make verify-envelope` - Runs strict envelope schema compatibility gate directly
- `make verify-fence` - Runs strict ownership-fence race/consistency gate directly
- `make verify-partition` - Runs strict network-partition fail-closed + bounded-grace gate directly

## Required Runtime Artifacts (for strict mode)

- `vmd/Cargo.toml`
- `portproxy/Cargo.toml`
- `crates/reson-sandbox/Cargo.toml`
- `proto/bracket/vmd/v1/vmd.proto`
- `proto/bracket/portproxy/v1/portproxy.proto`

## Notes

- Rust-only migration boundary: Python integrations and generated Python stubs are intentionally excluded.
- Strict mode is the CI enforcement target and includes fork CoW checks.
- Distributed control is enabled by default (feature `distributed-control`):
  - etcd stores node registry + durable `session_id -> node endpoint` bindings under `/reson-sandbox`.
  - NATS JetStream carries durable control commands/events with explicit-ack consumers and dead-letter replay tooling.
  - Facade API surface does not change (`Sandbox::new/connect/session/attach/list/fork` remain the same).
  - `vmd` can self-register with etcd lease heartbeats when `RESON_SANDBOX_ETCD_ENDPOINTS` is set.
  - Dead-letter replay utility: `scripts/replay_mq_dead_letters.sh <limit> [--nats-url ... --subject-prefix ...]`.
  - `SandboxConfig.control_gateway_endpoints` supports secondary control endpoints for failover without API-surface changes.
  - HA target contract and remaining scope are tracked in `specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md`.
- Security profile (production-facing):
  - `vmd` supports bearer-token auth via `--auth-token` / `RESON_SANDBOX_AUTH_TOKEN`.
  - Optional read-only token is supported via `--readonly-auth-token` / `RESON_SANDBOX_READONLY_AUTH_TOKEN`.
  - Optional TLS server profile is supported via `--tls-cert`, `--tls-key`, and `--tls-client-ca`.
  - Facade clients can set `SandboxConfig.auth_token` and `SandboxConfig.tls` for authenticated TLS endpoints.
- Admission control baseline:
  - `vmd` supports `--max-active-vms` (or `RESON_SANDBOX_MAX_ACTIVE_VMS`) to reject over-capacity session creates with explicit `resource_exhausted` errors.
- Storage profile and HA mode baseline:
  - `vmd` supports `--storage-profile {local-ephemeral|durable-shared}` (or `RESON_SANDBOX_STORAGE_PROFILE`).
  - `vmd --ha-mode` (or `RESON_SANDBOX_HA_MODE=1`) enforces durable-shared storage and requires both etcd node registry + NATS control bus configuration.
  - Continuity tier policy is enforced: distributed HA defaults to Tier-B (`RESON_SANDBOX_CONTINUITY_TIER=tier-b`), and Tier-A is permitted only with explicit degraded mode (`RESON_SANDBOX_DEGRADED_MODE=1`).
  - Node heartbeats publish `continuity_tier` and `degraded_mode`; distributed node selection defaults to `required_continuity_tier=tier-b` and excludes degraded Tier-A unless explicitly allowed in config.
  - Node heartbeats include `storage_profile`; distributed routing can require a profile via `DistributedControlConfig.required_storage_profile`.
  - Fork lineage metadata includes storage profile and fork durability hints (`reson.storage_profile`, `reson.fork_durability_class`, `reson.fork_restore_scope`).
- SLO profile:
  - Threshold schema is defined in `specs/RESON_SANDBOX_SLO_THRESHOLDS.json`.
  - Local strict verification enforces schema + instrumentation + evaluator tests.
  - CI/nightly can enforce observed SLO budgets by setting `RESON_SANDBOX_SLO_OBSERVED_FILE`.
  - CI/nightly can enforce staged rollout pause/rollback decisions via `RESON_SANDBOX_ROLLOUT_OBSERVED_FILE` and `scripts/evaluate_rollout_policy.sh`.
  - Rollout policy runbook: `specs/runbooks/STAGED_ROLLOUT_AND_AUTOMATIC_ROLLBACK_POLICY.md`.
- DR automation baseline:
  - `scripts/dr/backup_etcd.sh` captures etcd snapshots (live or dry-run).
  - `scripts/dr/restore_etcd.sh` restores snapshots into a new data dir (live or dry-run).
  - `scripts/dr/run_restore_drill.sh` runs a repeatable restore drill and emits `drill_report.txt`.
- Security operations baseline:
  - TLS bundle rotation automation: `scripts/security/rotate_tls_bundle.sh`.
  - Client cert revocation automation: `scripts/security/revoke_client_cert.sh`.
  - Operational runbook: `specs/runbooks/SECURITY_CERT_ROTATION_AND_SECRETS_PLAYBOOK.md`.
- Envelope compatibility baseline:
  - Command/event envelope schemas are locked in `specs/schemas/`.
  - Compatibility policy is documented in `specs/RESON_SANDBOX_ENVELOPE_COMPATIBILITY_POLICY.md`.
- Ownership fence baseline:
  - Distributed session-route ownership transitions use fence-token compare-and-swap semantics.
  - Control commands carry optional `expected_fence` and are rejected on stale ownership tokens.
- Network-partition fail-closed baseline:
  - `vmd` probes etcd quorum visibility and transitions into fail-closed mutation mode on sustained probe failures.
  - Mutating gRPC RPCs and MQ control commands are rejected while partitioned.
  - Local pre-existing data-plane streams are allowed only within a bounded grace window.
- Control-gateway HA baseline:
  - HA topology + drill runbook: `specs/runbooks/CONTROL_GATEWAY_HA_TOPOLOGY_AND_FAILOVER_DRILL.md`.
  - Automated drill script: `scripts/run_control_gateway_failover_drill.sh`.
