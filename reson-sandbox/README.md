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
- `make verify-gateway` - Runs strict control-gateway failover gate directly

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
- SLO profile:
  - Threshold schema is defined in `specs/RESON_SANDBOX_SLO_THRESHOLDS.json`.
  - Local strict verification enforces schema + instrumentation + evaluator tests.
  - CI/nightly can enforce observed SLO budgets by setting `RESON_SANDBOX_SLO_OBSERVED_FILE`.
