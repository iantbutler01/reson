# @dive-file: Convenience entrypoints for strict verifier gates and operator drills.
# @dive-rel: Mirrors scripts/verify_reson_sandbox.sh gate coverage to enforce checklist completeness.
# @dive-rel: Surfaces per-gate commands used by CI and local contract validation loops.
SHELL := /bin/bash

.PHONY: verify verify-strict verify-e2e verify-e2e-leaks verify-api verify-fork verify-mq verify-reconcile verify-chaos verify-soak verify-security verify-slo evaluate-rollout-policy verify-outbox verify-gateway verify-storage verify-admission verify-dr verify-security-ops verify-envelope verify-fence verify-partition verify-gameday verify-continuity verify-drain verify-restore verify-exactly-once verify-fidelity verify-warm-pool integration-up integration-down integration-reset integration-artifacts drill-gateway-failover replay-dlq

verify:
	./scripts/verify_reson_sandbox.sh

verify-strict:
	./scripts/verify_reson_sandbox.sh --strict

verify-e2e:
	./scripts/verify_reson_sandbox.sh --strict --with-e2e

verify-e2e-leaks:
	./scripts/verify_reson_sandbox.sh --strict --with-e2e --with-leak-check

verify-api:
	./scripts/verify_api_facade.sh --strict

verify-fork:
	./scripts/verify_fork_cow.sh --strict

verify-mq:
	./scripts/verify_mq_control.sh --strict

verify-reconcile:
	./scripts/verify_reconcile.sh --strict

verify-chaos:
	./scripts/verify_distributed_chaos.sh --strict

verify-soak:
	./scripts/verify_distributed_soak.sh --strict

verify-security:
	./scripts/verify_security_profile.sh --strict

verify-slo:
	./scripts/verify_slo_profile.sh --strict

evaluate-rollout-policy:
	./scripts/evaluate_rollout_policy.sh

verify-outbox:
	./scripts/verify_outbox.sh --strict

verify-gateway:
	./scripts/verify_control_gateway_failover.sh --strict

verify-storage:
	./scripts/verify_storage_profile.sh --strict

verify-admission:
	./scripts/verify_admission_control.sh --strict

verify-dr:
	./scripts/verify_dr_restore.sh --strict

verify-security-ops:
	./scripts/verify_security_ops.sh --strict

verify-envelope:
	./scripts/verify_envelope_compat.sh --strict

verify-fence:
	./scripts/verify_ownership_fence.sh --strict

verify-partition:
	./scripts/verify_network_partition.sh --strict

verify-gameday:
	./scripts/verify_operational_gameday.sh --strict

verify-continuity:
	./scripts/verify_tierb_continuity.sh --strict

verify-drain:
	./scripts/verify_planned_drain_handoff.sh --strict

verify-restore:
	./scripts/verify_tierb_execution_restore.sh --strict

verify-exactly-once:
	./scripts/verify_tierb_exactly_once.sh --strict

verify-fidelity:
	./scripts/verify_execution_fidelity_policy.sh --strict

verify-warm-pool:
	./scripts/verify_warm_pool_pipeline.sh --strict

integration-up:
	./scripts/integration/up.sh --profile local-dev

integration-down:
	./scripts/integration/down.sh --profile local-dev

integration-reset:
	./scripts/integration/reset.sh --profile local-dev

integration-artifacts:
	./scripts/integration/collect-artifacts.sh --profile local-dev --reason make-target

drill-gateway-failover:
	./scripts/run_control_gateway_failover_drill.sh --strict

replay-dlq:
	./scripts/replay_mq_dead_letters.sh $(or $(LIMIT),100)
