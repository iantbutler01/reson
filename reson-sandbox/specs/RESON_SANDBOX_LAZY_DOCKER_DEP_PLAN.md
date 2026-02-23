# Docker Lazy Dependency Plan (vmd)

## Goal
Make Docker optional at `vmd` startup. Require Docker only when OCI->qcow2 fallback conversion is actually needed.

## Scope
- `vmd/src/config.rs`
- `vmd/src/state/manager.rs`
- `vmd` unit tests touching config normalize + docker platform/convert paths

## Changes
1. Remove eager Docker resolution in `Config::normalize()`.
- Delete unconditional `resolve_binary(&self.docker_bin, "docker")`.

2. Add lazy Docker availability check in manager.
- Helper checks whether `<docker_bin> --version` is runnable.

3. Update `resolve_docker_platform(...)`.
- If Docker available: keep current inspect path.
- If Docker unavailable:
  - with requested arch: accept requested arch and `linux/<arch>`
  - without requested arch: use host arch and `linux/<host_arch>`
- Log warning that manifest inspect was skipped due to missing Docker.

4. Gate conversion fallback with explicit error.
- Before `virt::run_d2vm(...)`, require Docker availability.
- If missing, return actionable error:
  - Docker required for local OCI->qcow2 fallback conversion
  - install Docker or provide prebuilt qcow2 for requested image/arch

## No Changes
- No facade API changes
- No proto changes
- No behavior changes when Docker is installed

## Tests
1. Config normalize succeeds without Docker present.
2. Platform resolution works without Docker:
- requested arch case
- host-arch default case
3. Conversion fallback without Docker returns explicit actionable error.
4. Existing conversion behavior unchanged when Docker exists.

## Acceptance Criteria
- `vmd` starts without Docker installed.
- Prebuilt image flow works unchanged.
- Missing Docker only blocks actual conversion fallback, with clear error.
- `cargo test -p vmd` passes.
