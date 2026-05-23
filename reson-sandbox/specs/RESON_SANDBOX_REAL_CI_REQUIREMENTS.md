<!-- @dive-file: Runner capability and environment requirements for verify_strict_real.sh and real gates 41-48. -->
<!-- @dive-rel: Complements specs/RESON_SANDBOX_REAL_INTEGRATION_TEST_PLAN.md section 7.7 CI requirements. -->
<!-- @dive-rel: Referenced by scripts/verify_strict_real.sh and Makefile verify-strict-real target. -->
# Reson Sandbox Real CI Requirements

## Required Host Capabilities

| Capability | Required | Why |
|---|---|---|
| `docker` | yes | real integration harness control-plane (`etcd`/`nats`) lifecycle |
| `docker compose` (plugin or `docker-compose`) | yes | profile-based control-plane orchestration |
| Rust toolchain (`cargo`) | yes | build + run real integration tests |
| `jq`, `lsof`, `curl` | yes | harness diagnostics/readiness/leak checks |
| `/dev/kvm` (Linux KVM runners) | required for `ci-linux-kvm` | authoritative virtualization fidelity and Tier-B runtime behavior |
| Mac/Linux networking (localhost ports) | yes | harness uses profile-scoped host ports |

## Profile Guidance

- `local-dev`: fast local real-gate coverage, bounded L3.
- `ci-linux-kvm`: authoritative CI profile for release acceptance.
- `nightly-soak`: long-run churn/leak/convergence coverage.

## Canonical Commands

Local bounded real run:

```bash
./scripts/verify_strict_real.sh --profile local-dev
```

CI authoritative run:

```bash
./scripts/verify_strict_real.sh --profile ci-linux-kvm --mock-preflight-strict
```

Nightly soak posture:

```bash
./scripts/verify_strict_real.sh --profile nightly-soak --mock-preflight-strict
```

## CI Merge Policy

- `pull_request`: run a bounded subset suitable for iterative review on KVM runners:
  - `./scripts/verify_reson_sandbox.sh --strict`
  - `./scripts/verify_real_gate41.sh --profile ci-linux-kvm`
  - `./scripts/verify_real_gate43.sh --profile ci-linux-kvm`
  - `./scripts/integration/verify_real_failover.sh --profile ci-linux-kvm --distributed --selector distributed_stream_resume_from_checkpoint_is_forward_only_without_replay`
- `push` to `main`: run full authoritative real verification:
  - `./scripts/verify_strict_real.sh --profile ci-linux-kvm --mock-preflight-strict`
- Branch protection must require this workflow before merge so `main` cannot advance without CI evidence.

## Environment Notes

- `verify_strict_real.sh` runs fast mock preflight by default, then mandatory real gates `41-48`.
- For post-failure debugging, add `--keep-stack` so compose services remain available for inspection.
- Artifacts are emitted under `.integration-artifacts/<profile>/<timestamp>/`.
