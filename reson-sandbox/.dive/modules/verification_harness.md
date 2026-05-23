<!-- @dive-file: Module-level metadata for strict verifier scripts, drill wrappers, and shared shell helpers. -->
<!-- @dive-rel: Complements .dive/modules/ha_contract.md by focusing on executable gate wiring rather than contract scope. -->
<!-- @dive-rel: Tracks script relationships used by scripts/verify_reson_sandbox.sh and scripts/verify_strict_real.sh. -->
# Verification Harness Module
Documents the shell-level verification and operational drill surface used to enforce sandbox readiness gates.

## Files
- `scripts/common.sh` - Shared verifier helpers (`log`, `warn`, `err`, runtime-source and command checks).
- `scripts/verify_*.sh` - Gate-specific contract checks and targeted runtime/integration assertions.
- `scripts/build_portproxy_guest_bins_docker.sh` - Guest `portproxy` artifact builder used by integration flows.
- `scripts/evaluate_rollout_policy.sh` - Rollout policy evaluator used by SLO verification.
- `scripts/replay_mq_dead_letters.sh` - DLQ replay utility for control-bus operations.
- `scripts/run_control_gateway_failover_drill.sh` - Control gateway failover drill and report emitter.
- `scripts/run_control_plane_gameday_drill.sh` - Control-plane outage/zonal game-day drill runner.
- `scripts/dr/backup_etcd.sh` - DR backup helper for etcd snapshots and manifests.
- `scripts/dr/restore_etcd.sh` - DR restore helper for etcd snapshots and manifests.
- `scripts/dr/run_restore_drill.sh` - DR drill orchestrator combining backup + restore scripts.
- `scripts/security/rotate_tls_bundle.sh` - TLS bundle rotation automation helper.
- `scripts/security/revoke_client_cert.sh` - Client certificate revocation automation helper.

## Relationships
- `scripts/verify_reson_sandbox.sh` -> `scripts/verify_*.sh`: orchestrates strict gate order and failure semantics.
- `scripts/verify_strict_real.sh` -> `scripts/verify_real_gate41.sh` ... `scripts/verify_real_gate48.sh`: enforces real integration gates after preflight.
- `scripts/verify_*.sh` -> `scripts/common.sh`: uses shared logging and runtime command guardrails.
- `scripts/verify_slo_profile.sh` -> `scripts/evaluate_rollout_policy.sh`: applies measured SLO data to rollout control decisions.
- `scripts/verify_operational_gameday.sh` -> `scripts/run_control_plane_gameday_drill.sh`: executes live outage drills and validates reports.
- `scripts/verify_control_gateway_failover.sh` -> `scripts/run_control_gateway_failover_drill.sh`: executes gateway failover drill and validates pass markers.
- `scripts/verify_dr_restore.sh` -> `scripts/dr/run_restore_drill.sh`: validates DR script chain and manifest outputs.
- `scripts/dr/run_restore_drill.sh` -> `scripts/dr/backup_etcd.sh`: produces source backup state used in restore rehearsal.
- `scripts/dr/run_restore_drill.sh` -> `scripts/dr/restore_etcd.sh`: validates restore command generation and artifacts.
- `scripts/verify_security_ops.sh` -> `scripts/security/rotate_tls_bundle.sh`: validates cert/key rotation automation.
- `scripts/verify_security_ops.sh` -> `scripts/security/revoke_client_cert.sh`: validates revocation ledger automation.
