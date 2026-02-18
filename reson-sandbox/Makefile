SHELL := /bin/bash

.PHONY: verify verify-strict verify-e2e verify-e2e-leaks verify-api verify-fork

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
