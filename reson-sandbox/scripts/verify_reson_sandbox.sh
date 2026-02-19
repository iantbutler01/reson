#!/usr/bin/env bash
# @dive-file: Top-level strict verifier orchestrator for reson-sandbox runtime, facade, and distributed HA contract gates.
# @dive-rel: Enforces executable checklist closure from specs/RESON_SANDBOX_HA_DISTRIBUTED_CONTRACT.md.
# @dive-rel: Delegates gate-specific checks to scripts/verify_*.sh and records contract coverage ordering.
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/common.sh"

STRICT=0
WITH_E2E=0
WITH_LEAK_CHECK=0

usage() {
  cat <<'USAGE'
Usage: verify_reson_sandbox.sh [--strict] [--with-e2e] [--with-leak-check]
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
      ;;
    --with-e2e)
      WITH_E2E=1
      shift
      ;;
    --with-leak-check)
      WITH_LEAK_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

if [[ "$STRICT" -eq 1 ]]; then
  WITH_E2E=1
  WITH_LEAK_CHECK=1
fi

if [[ "$WITH_LEAK_CHECK" -eq 1 && "$WITH_E2E" -ne 1 ]]; then
  err "--with-leak-check requires --with-e2e"
  exit 2
fi

log "gate 0: repository contract checks"
for required_path in \
  "Cargo.toml" \
  "scripts/common.sh" \
  "scripts/verify_proto_clean.sh" \
  "scripts/verify_e2e_smoke.sh" \
  "scripts/verify_no_leaks.sh" \
  "scripts/verify_mq_control.sh" \
  "scripts/verify_reconcile.sh" \
  "scripts/verify_distributed_chaos.sh" \
  "scripts/verify_distributed_soak.sh" \
  "scripts/verify_security_profile.sh" \
  "scripts/verify_slo_profile.sh" \
  "scripts/evaluate_rollout_policy.sh" \
  "scripts/verify_outbox.sh" \
  "scripts/verify_control_gateway_failover.sh" \
  "scripts/run_control_gateway_failover_drill.sh" \
  "scripts/verify_storage_profile.sh" \
  "scripts/verify_admission_control.sh" \
  "scripts/verify_dr_restore.sh" \
  "scripts/verify_security_ops.sh" \
  "scripts/verify_envelope_compat.sh" \
  "scripts/verify_ownership_fence.sh" \
  "scripts/verify_network_partition.sh" \
  "scripts/verify_operational_gameday.sh" \
  "scripts/verify_tierb_continuity.sh" \
  "scripts/verify_planned_drain_handoff.sh" \
  "scripts/verify_tierb_execution_restore.sh" \
  "scripts/verify_tierb_exactly_once.sh" \
  "scripts/verify_execution_fidelity_policy.sh" \
  "scripts/verify_warm_pool_pipeline.sh" \
  "scripts/run_control_plane_gameday_drill.sh" \
  "scripts/dr/backup_etcd.sh" \
  "scripts/dr/restore_etcd.sh" \
  "scripts/dr/run_restore_drill.sh" \
  "scripts/security/rotate_tls_bundle.sh" \
  "scripts/security/revoke_client_cert.sh" \
  "scripts/replay_mq_dead_letters.sh" \
  "scripts/verify_api_facade.sh" \
  "scripts/verify_fork_cow.sh" \
  "specs/RESON_SANDBOX_MIGRATION_CHECKLIST.md" \
  "specs/runbooks/CONTROL_GATEWAY_HA_TOPOLOGY_AND_FAILOVER_DRILL.md" \
  "specs/runbooks/CONTROL_PLANE_GAMEDAY_ETCD_MQ_ZONE_FAILURE.md" \
  "specs/runbooks/PLANNED_DRAIN_HANDOFF_AND_FENCING.md" \
  "specs/runbooks/TIERB_EXECUTION_STATE_RESTORE.md" \
  "specs/runbooks/TIERB_MID_COMMAND_FAILOVER_EXACTLY_ONCE.md" \
  "specs/runbooks/TIERB_EXECUTION_FIDELITY_POLICY.md" \
  "specs/runbooks/WARM_POOL_AND_PREWARM_PIPELINE.md" \
  "specs/runbooks/SECURITY_CERT_ROTATION_AND_SECRETS_PLAYBOOK.md" \
  "specs/runbooks/STAGED_ROLLOUT_AND_AUTOMATIC_ROLLBACK_POLICY.md" \
  "deploy/gameday/docker-compose.control-plane.yml" \
  "specs/RESON_SANDBOX_ENVELOPE_COMPATIBILITY_POLICY.md" \
  "specs/schemas/control_command_envelope.v1.json" \
  "specs/schemas/control_event_envelope.v1.json" \
  "specs/RESON_SANDBOX_SLO_THRESHOLDS.json" \
  "crates/reson-sandbox/Cargo.toml"; do
  if [[ ! -f "$REPO_ROOT/$required_path" ]]; then
    err "missing required file: $required_path"
    exit 1
  fi
done

if have_runtime_sources; then
  require_cmd cargo

  log "gate 1: cargo check --workspace"
  (cd "$REPO_ROOT" && cargo check --workspace)

  log "gate 2: cargo clippy --workspace --all-targets --all-features -- -D warnings"
  (cd "$REPO_ROOT" && cargo clippy --workspace --all-targets --all-features -- -D warnings)

  log "gate 3: cargo test --workspace"
  (cd "$REPO_ROOT" && cargo test --workspace)
else
  if [[ "$STRICT" -eq 1 ]]; then
    err "runtime sources missing; strict mode requires vmd, portproxy, and proto inputs"
    exit 1
  fi
  warn "runtime sources missing; skipping gates 1-3"
fi

log "gate 4: proto contract"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_proto_clean.sh" --strict
else
  "$REPO_ROOT/scripts/verify_proto_clean.sh"
fi

if [[ "$WITH_E2E" -eq 1 ]]; then
  log "gate 5: e2e smoke"
  if [[ "$STRICT" -eq 1 ]]; then
    "$REPO_ROOT/scripts/verify_e2e_smoke.sh" --strict
  else
    "$REPO_ROOT/scripts/verify_e2e_smoke.sh"
  fi
else
  log "gate 5: e2e smoke skipped (enable with --with-e2e)"
fi

if [[ "$WITH_LEAK_CHECK" -eq 1 ]]; then
  log "gate 6: leak gate"
  smoke_cmd=("$REPO_ROOT/scripts/verify_e2e_smoke.sh")
  if [[ "$STRICT" -eq 1 ]]; then
    smoke_cmd+=("--strict")
  fi
  "$REPO_ROOT/scripts/verify_no_leaks.sh" -- "${smoke_cmd[@]}"
else
  log "gate 6: leak gate skipped (enable with --with-leak-check)"
fi

log "gate 7: facade API"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_api_facade.sh" --strict
else
  "$REPO_ROOT/scripts/verify_api_facade.sh"
fi

log "gate 8: fork CoW"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_fork_cow.sh" --strict
else
  "$REPO_ROOT/scripts/verify_fork_cow.sh"
fi

log "gate 9: mq control contract"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_mq_control.sh" --strict
else
  "$REPO_ROOT/scripts/verify_mq_control.sh"
fi

log "gate 10: reconciliation convergence"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_reconcile.sh" --strict
else
  "$REPO_ROOT/scripts/verify_reconcile.sh"
fi

log "gate 11: distributed chaos"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_distributed_chaos.sh" --strict
else
  "$REPO_ROOT/scripts/verify_distributed_chaos.sh"
fi

log "gate 12: distributed soak"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_distributed_soak.sh" --strict
else
  "$REPO_ROOT/scripts/verify_distributed_soak.sh"
fi

log "gate 13: security profile"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_security_profile.sh" --strict
else
  "$REPO_ROOT/scripts/verify_security_profile.sh"
fi

log "gate 14: slo profile"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_slo_profile.sh" --strict
else
  "$REPO_ROOT/scripts/verify_slo_profile.sh"
fi

log "gate 15: transactional outbox"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_outbox.sh" --strict
else
  "$REPO_ROOT/scripts/verify_outbox.sh"
fi

log "gate 16: control gateway failover"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_control_gateway_failover.sh" --strict
else
  "$REPO_ROOT/scripts/verify_control_gateway_failover.sh"
fi

log "gate 17: storage profile durability semantics"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_storage_profile.sh" --strict
else
  "$REPO_ROOT/scripts/verify_storage_profile.sh"
fi

log "gate 18: scheduler/admission control"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_admission_control.sh" --strict
else
  "$REPO_ROOT/scripts/verify_admission_control.sh"
fi

log "gate 19: disaster recovery restore drill"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_dr_restore.sh" --strict
else
  "$REPO_ROOT/scripts/verify_dr_restore.sh"
fi

log "gate 20: security operations"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_security_ops.sh" --strict
else
  "$REPO_ROOT/scripts/verify_security_ops.sh"
fi

log "gate 21: envelope compatibility"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_envelope_compat.sh" --strict
else
  "$REPO_ROOT/scripts/verify_envelope_compat.sh"
fi

log "gate 22: ownership fence tokens"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_ownership_fence.sh" --strict
else
  "$REPO_ROOT/scripts/verify_ownership_fence.sh"
fi

log "gate 23: network partition fail-closed"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_network_partition.sh" --strict
else
  "$REPO_ROOT/scripts/verify_network_partition.sh"
fi

log "gate 24: operational game-day automation"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_operational_gameday.sh" --strict
else
  "$REPO_ROOT/scripts/verify_operational_gameday.sh"
fi

log "gate 25: tier-b continuity orchestration"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_tierb_continuity.sh" --strict
else
  "$REPO_ROOT/scripts/verify_tierb_continuity.sh"
fi

log "gate 26: planned drain handoff + fencing"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_planned_drain_handoff.sh" --strict
else
  "$REPO_ROOT/scripts/verify_planned_drain_handoff.sh"
fi

log "gate 27: tier-b execution-state restore"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_tierb_execution_restore.sh" --strict
else
  "$REPO_ROOT/scripts/verify_tierb_execution_restore.sh"
fi

log "gate 33: tier-b mid-command failover exactly-once"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_tierb_exactly_once.sh" --strict
else
  "$REPO_ROOT/scripts/verify_tierb_exactly_once.sh"
fi

log "gate 35: tier-b execution-fidelity policy"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_execution_fidelity_policy.sh" --strict
else
  "$REPO_ROOT/scripts/verify_execution_fidelity_policy.sh"
fi

log "gate 36: warm-pool and prewarm pipeline"
if [[ "$STRICT" -eq 1 ]]; then
  "$REPO_ROOT/scripts/verify_warm_pool_pipeline.sh" --strict
else
  "$REPO_ROOT/scripts/verify_warm_pool_pipeline.sh"
fi

log "all enabled gates passed"
