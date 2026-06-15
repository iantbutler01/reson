# Goal: OpenComputer As A Sandbox Backend

OpenComputer is just another sandbox provider. It must plug into the existing
`chevalier-sandbox` sandbox API instead of creating a parallel public API.

Non-negotiable direction:

- Do not add a separate downstream feature such as `sandbox-opencomputer`.
- Do not expose a parallel public `OpenComputerClient` / `OpenComputerSession`
  surface as the product-facing interface.
- Do not require consumers to change from the existing `Sandbox` / `Session`
  workflow to provider-specific types.
- The public shape should be config-level selection on the existing sandbox
  context, for example `SandboxConfig` choosing local/direct, distributed, or
  OpenComputer.
- Any OpenComputer REST implementation details should be private/internal
  plumbing behind the existing facade.
- Product repos should see this as a config swap, not an API migration.
- OtherYou is not part of this implementation pass except as historical
  reference for desired semantics.

Current correction needed:

- Remove any attempted public OpenComputer feature/export/API surface.
- Rework the implementation so existing `Sandbox::new`, `Sandbox::session`,
  `Sandbox::attach_session`, and `Session` operations branch through the
  configured backend.
- Do not assume OpenComputer lacks fork/snapshot or interactive shell support.
  Verify the API/SDK shape and implement those capabilities through the existing
  `Session::fork` / `Session::shell` surface when available.
- Only operations that are actually unavailable after verification should return
  the existing `SandboxError::Unsupported` from the existing methods.

Implementation note:

- OpenComputer fork/checkpoint support belongs behind `Session::fork`.
- OpenComputer websocket exec support also backs `Session::shell`, so callers
  should not see a separate interactive-shell API.
- OpenComputer also exposes a PTY endpoint. The SDK's own stateful shell helper
  is still backed by a long-running exec websocket, which is the current
  `Session::shell` mapping; use PTY later only if the facade grows terminal
  sizing/PTY-specific semantics.
- OpenComputer rclone/FUSE mounts are provider config behind the existing
  sandbox facade. Static provider mounts live in
  `OpenComputerBackendConfig.mounts`; Chevalier `SessionOptions.shared_mounts`
  require an explicit `OpenComputerBackendConfig.shared_mounts` mapping by
  mount tag, guest path, or backend profile so VFS mounts do not silently
  disappear under a provider swap.
- Unsupported responses should be narrow: currently sandbox listing is
  unavailable through the documented API, and OpenComputer preview URLs do not
  fit the existing local-port `ForwardHandle` return shape.
