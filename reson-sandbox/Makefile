SHELL := /bin/bash

.PHONY: verify verify-strict verify-e2e verify-e2e-leaks verify-api verify-fork verify-mq verify-reconcile verify-chaos verify-soak verify-security verify-slo verify-outbox verify-gateway replay-dlq

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

verify-outbox:
	./scripts/verify_outbox.sh --strict

verify-gateway:
	./scripts/verify_control_gateway_failover.sh --strict

replay-dlq:
	./scripts/replay_mq_dead_letters.sh $(or $(LIMIT),100)
