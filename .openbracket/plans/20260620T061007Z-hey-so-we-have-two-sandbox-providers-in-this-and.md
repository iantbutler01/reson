# Hey so we have two sandbox providers in this and I want to add a third Alright time to add

Created: 2026-06-20T06:10:07.559Z
Workspace: Chevalier

## Request

Hey so we have two sandbox providers in this and I want to add a third Alright time to add a 3rd provider: https://www.freestyle.sh/docs They gave me 10 grand in credit so yeah lol

## Plan

### Goal
Add Freestyle (https://www.freestyle.sh/docs) as a third sandbox provider alongside the two existing providers in this repository.

### Repository discoveries
- Primary sandbox abstraction is the Rust crate `sandbox/crates/sandbox` (`chevalier-sandbox`).
- Existing provider enum is in `sandbox/crates/sandbox/src/lib.rs`:
  - `SandboxProviderConfig::Chevalier` (default local/daemon-backed provider)
  - `SandboxProviderConfig::OpenComputer(OpenComputerBackendConfig)` (provider-managed HTTP/WebSocket backend)
- Existing OpenComputer provider transport lives in `sandbox/crates/sandbox/src/opencomputer.rs` and is private to the facade.
- Provider selection is programmatic through `SandboxConfig { provider: SandboxProviderConfig, ... }`; the TypeScript wrapper also maps provider config in `ts-sandbox/src/lib.rs`.
- OpenComputer env/config parsing is implemented by `OpenComputerBackendConfig::from_env()` in `sandbox/crates/sandbox/src/lib.rs` with env vars such as `OPENCOMPUTER_API_URL`, `OPENCOMPUTER_API_KEY`, `OPENCOMPUTER_TEMPLATE_ID`, `OPENCOMPUTER_CHECKPOINT_ID`, `OPENCOMPUTER_SANDBOX_TIMEOUT_SECS`, `OPENCOMPUTER_MOUNTS_JSON`, `OPENCOMPUTER_SHARED_MOUNTS_JSON`, and `OPENCOMPUTER_EGRESS_ALLOWLIST_JSON`.
- The existing provider branch is wired in `Sandbox::build_control_backend`, `Sandbox::session`, `Sandbox::attach_session`, and delete/alias helpers in `sandbox/crates/sandbox/src/lib.rs`.
- Tests/docs to update include:
  - `sandbox/crates/sandbox/tests/opencomputer_live.rs` as the live-provider test pattern.
  - Unit tests embedded in `sandbox/crates/sandbox/src/opencomputer.rs` as the mocked HTTP/provider-adapter pattern.
  - TypeScript wrapper provider parsing in `ts-sandbox/src/lib.rs` and docs in `ts-sandbox/README.md`.
  - Existing sandbox VFS/OpenComputer docs in `docs/sandbox-vfs.md` and design notes in `goals/opencomputer-sandbox-provider.md` may need Freestyle notes if VFS/shared mounts are supported.

### Implementation contract
1. Add a new provider config variant in `sandbox/crates/sandbox/src/lib.rs`:
   - `SandboxProviderConfig::Freestyle(FreestyleBackendConfig)`.
   - Add `FreestyleBackendConfig` near `OpenComputerBackendConfig` with `Deserialize`/`Serialize` if needed by TS bindings.
   - Suggested fields, subject to Freestyle docs: `api_url`, `api_key`, `template_id` or base image/source identifier, `timeout_secs`, optional default resources, env/secret/mount-related config if Freestyle supports them.
   - Add `FreestyleBackendConfig::from_env()` with `FREESTYLE_API_URL` (default from docs), `FREESTYLE_API_KEY`, `FREESTYLE_TEMPLATE_ID`/equivalent, `FREESTYLE_SANDBOX_TIMEOUT_SECS`, and any docs-confirmed Freestyle options.
2. Add `sandbox/crates/sandbox/src/freestyle.rs` modeled after `opencomputer.rs`:
   - Private `FreestyleControl` with `new`, `api_url`, `create_sandbox`, `get_sandbox`, `delete_sandbox`, file operations, exec, shell, preview URL, and any unsupported-operation errors matching facade expectations.
   - Keep response/data shapes isolated in this module; map all provider errors to `SandboxError`.
   - Use mocked unit tests in this file for URL normalization, auth headers, create/get/delete, exec/shell stream handling, preview URL generation, and unsupported features.
3. Wire Freestyle into the facade in `sandbox/crates/sandbox/src/lib.rs`:
   - Add `mod freestyle;`.
   - Extend the internal `ControlBackend` enum with `Freestyle(freestyle::FreestyleControl)`.
   - Generalize OpenComputer-specific alias helpers/names (`opencomputer_session_aliases`, `bind_opencomputer_session_alias`, etc.) to provider-managed aliases, or add Freestyle-specific equivalents if minimizing churn.
   - In `Sandbox::build_control_backend`, instantiate `FreestyleControl::new` when `SandboxProviderConfig::Freestyle` is selected.
   - In `Sandbox::session`, `attach_session`, deletion, `provider_preview_url`, file, exec, shell, and restore/checkpoint paths, dispatch both provider-managed variants. Prefer extracting shared provider-managed branches instead of copy/pasting OpenComputer logic.
   - Add SLO labels such as `session.create.freestyle` and `session.attach.freestyle`.
4. Update TypeScript bindings/wrapper in `ts-sandbox/src/lib.rs`:
   - Add Freestyle config parsing alongside the current OpenComputer parsing.
   - Accept a provider value/name like `"freestyle"` and map it to `SandboxProviderConfig::Freestyle(FreestyleBackendConfig::from_env() or explicit config)`.
   - Update `ts-sandbox/README.md` with a Freestyle example and env vars.
5. Add tests:
   - Rust unit tests in `sandbox/crates/sandbox/src/freestyle.rs` using mocked HTTP/WebSocket behavior, following `opencomputer.rs` patterns.
   - A live-provider test file such as `sandbox/crates/sandbox/tests/freestyle_live.rs`, gated behind `FREESTYLE_LIVE=1` and `FREESTYLE_API_KEY`, following `opencomputer_live.rs`.
   - Facade tests in `sandbox/crates/sandbox/tests/facade_contract.rs` or lib tests if aliasing/provider dispatch is refactored.
6. Update docs:
   - `ts-sandbox/README.md` with provider selection and env variables.
   - Add a short Rust example either to `sandbox/README.md` or a new docs section showing `SandboxProviderConfig::Freestyle(FreestyleBackendConfig::from_env()?)`.

### Data-shape notes for the coding agent
- Existing public shape:
  - `SandboxConfig { provider: SandboxProviderConfig, endpoint, auto_spawn, default_image, default_resources, ... }`.
  - `SessionOptions { session_id, name, image, resources, metadata, shared_mounts, egress_allowlist, ... }`.
  - Provider-created sessions should return `Session::new_with_backend(self.clone(), logical_session_id, provider_sandbox_id, control.api_url().to_string(), None, opts.shared_mounts)` or equivalent.
- Existing provider-managed metadata keys set for OpenComputer should also be set for Freestyle unless docs require otherwise:
  - `workspace_id`, `tenant_id`, `chevalier.requested_session_id`, `chevalier.name`, `chevalier.tier_b_eligible=false`, `chevalier.execution_fidelity_requirement=provider-managed`.

### Risks / open questions
- Need exact Freestyle API endpoints, auth header, request/response fields, websocket protocol, file API, preview URL behavior, and sandbox restore/fork semantics from https://www.freestyle.sh/docs.
- Freestyle may not support every operation the facade exposes (shared mounts, snapshots/checkpoints, list sessions, persistent IDs, VFS callback). Unsupported features should return `SandboxError::Unsupported` with clear messages.
- Current OpenComputer-specific alias naming may create technical debt if copied; prefer a small provider-managed abstraction or neutral naming.
- If Freestyle's model is project/git/app-based rather than raw VM/sandbox-based, mapping `SessionOptions.image/resources/shared_mounts` may require deliberate defaults or documented limitations.

### Ordered implementation tasks
1. Read Freestyle docs and fill in exact `FreestyleBackendConfig` fields, endpoint defaults, auth header, create/get/delete/exec/file/preview data shapes.
2. Add `FreestyleBackendConfig`, env parsing, and `SandboxProviderConfig::Freestyle` in `sandbox/crates/sandbox/src/lib.rs`.
3. Create `sandbox/crates/sandbox/src/freestyle.rs` implementing the provider transport and mocked tests.
4. Wire `FreestyleControl` into `ControlBackend` and provider-managed facade branches in `sandbox/crates/sandbox/src/lib.rs`; generalize alias helpers if possible.
5. Update `ts-sandbox/src/lib.rs` provider parsing and `ts-sandbox/README.md` examples.
6. Add live test `sandbox/crates/sandbox/tests/freestyle_live.rs` gated by env vars.
7. Update Rust docs/README with Freestyle env vars and a minimal config example.
8. Run verification gates.

### Verification gates
- `cargo test -p chevalier-sandbox` from `sandbox/` passes.
- Focused tests for `freestyle` module pass.
- `cargo test` or equivalent for `ts-sandbox` passes after wrapper updates.
- `FREESTYLE_LIVE=1 FREESTYLE_API_KEY=... cargo test -p chevalier-sandbox --test freestyle_live -- --ignored` (or the implemented live-test command) succeeds when credentials are available.
- Docs mention required Freestyle env vars, provider selection value, and unsupported operations/limitations.
