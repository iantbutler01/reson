<!-- @dive-file: Migration boundary for moving OtherYou VM/VFS runtime wiring into chevalier-sandbox. -->
<!-- @dive-rel: Complements CHEVALIER_SANDBOX_HA_DISTRIBUTED_CONTRACT.md by defining the product-facing API boundary. -->
<!-- @dive-rel: Source inventory is OtherYou api/src/services/runtime and api/src/services/computer/chevalier.rs. -->
# Chevalier Sandbox OtherYou VFS/VM Migration

Status: Generic FUSE-server/VFS gateway extraction implemented and validated through OtherYou against corvidae VMD nodes, GCS-backed VFS, FUSE mounts, agent exec, pause/resume, and warm attach. Longer-term storage-engine extraction and any rclone-backed implementation remain deferred behind the same gateway contract.
Owner: Chevalier Sandbox
Intent: Make OtherYou consume a simple product-level public API for VM and VFS behavior, without owning vmd wiring, FUSE client/server transport, distributed placement machinery, CoreDNS/firewall details, or storage backend internals. The chevalier SDK surface is generic; product-specific names remain only inside product adapters and existing source-file references.

## Boundary Principle

OtherYou should describe product intent:

- which product entity owns the runtime
- which product scopes should be mounted
- which scopes are writable
- which ingress exposures are enabled
- which network policy applies
- when a surface should be created, forked, checkpointed, resumed, or retired

Chevalier should own runtime mechanics:

- vmd lifecycle and routing
- local, remote, and distributed control behavior
- shared mount validation
- FUSE/VFS protocol transport, including the generic HTTP server that backs vmd's FUSE client
- storage backend implementation
- checkpoint/restore provider contract
- public ingress proxy mechanics
- network enforcement mechanics

OtherYou should not need to build custom vmd mount lists, derive CoreDNS/firewall machinery, know the internal VFS HTTP route shape, or hand-maintain distributed runtime invariants.

Local-first is the default. Distributed control, guest DNS/proxy, CoreDNS, Envoy, firewall routing, and public ingress are explicit profile selections. A macOS local dev launch must be able to use a single vmd without CoreDNS or distributed control-plane processes.

## Adapter Placement Rule

Any adapter that translates OtherYou product concepts into chevalier SDK requests lives in OtherYou, not chevalier.

Chevalier may expose generic SDK primitives such as owner refs, mount intents, backend contracts, checkpoint policy, ingress policy, and network policy. Chevalier must not contain OtherYou-specific adapter types, product-specific naming, product path semantics, or product database assumptions.

OtherYou owns the product adapter that maps its product concepts, user ownership, product paths, and product persistence into those generic chevalier primitives.

Current adapter location:

- OtherYou builds generic chevalier mount intents in `api/src/services/computer/chevalier.rs`.
- OtherYou emits generic vmd metadata keys: `chevalier.network_policy` and `chevalier.public_ingresses`.
- OtherYou accepts generic chevalier VFS/ingress headers and keeps legacy product-header compatibility only inside OtherYou routes.
- OtherYou local launch scripts export `CHEVALIER_SANDBOX_VFS_INTERNAL_SERVICE_TOKEN` and only enable guest network/CoreDNS defaults when explicitly requested or configured.
- Resolved gateway gap: `api/src/routes/runtime_nymfs_internal.rs` now owns only the OtherYou backend adapter and legacy aliasing. The generic HTTP FUSE server behavior, request DTOs, range parsing, lease/fence handshake, and status mapping live behind the chevalier `vfs-server` feature.

No adapter in chevalier is named for, shaped around, or coupled to the OtherYou product. Product-specific compatibility shims remain in OtherYou only.

## Target Public API Shape

The migration target is a chevalier-owned facade with a product-level shape similar to:

```rust
let runtime = chevalier_sandbox::vm::Runtime::new(config);

let surface = runtime.ensure_surface(VmSurfaceSpec {
    owner: RuntimeOwnerRef::new("runtime-owner", product_entity_id),
    mounts: vec![
        VfsMountIntent::root_read_only("/workspace", "runtimefs"),
        VfsMountIntent::scoped(
            "/workspace/shared",
            "rfs-shared",
            "projects/current/shared",
            false,
        ),
        VfsMountIntent::scoped(
            "/workspace/task",
            "rfs-task",
            "projects/current/task",
            false,
        ),
    ],
    ingress: ingress_policy,
    network: network_policy,
    checkpoint: checkpoint_policy,
}).await?;
```

This is illustrative, not a final Rust API. The required shape is the important part: OtherYou passes product refs and intent, chevalier returns runtime surfaces and events.

The FUSE server side of the same API should have a product-adapter shape similar to:

```rust
let vfs_gateway = chevalier_sandbox_vfs_server::VfsGateway::new(
    VfsGatewayConfig {
        internal_service_token,
        body_limit_bytes: 64 * 1024 * 1024,
        legacy_route_prefixes: vec!["/v1/internal/nymfs/{owner_id}"],
    },
    OtherYouVfsBackend::new(app_state),
);

let app = app.merge(vfs_gateway.routes("/v1/internal/chevalier/vfs/{owner_id}"));
```

This is also illustrative. The required boundary is:

- chevalier owns the generic HTTP/FUSE server contract and route behavior
- OtherYou owns the backend implementation passed into that server
- legacy OtherYou paths may remain as delegated aliases during migration, but they do not define the generic contract

## Migration List

### 1) Public VM Runtime Facade

Move a stable VM runtime facade into chevalier.

Chevalier should own:

- create, attach, fork, exec, checkpoint, resume, discard
- stale VM detection
- shared mount contract comparison
- distributed control config
- provider-level VM target, surface, and checkpoint DTOs
- endpoint/routing rewrite behavior

OtherYou should keep:

- product authorization
- developer-mode gating
- task and conversation selection
- DB rows tying a product entity/task to a VM surface
- audit events and product status responses

Source evidence:

- OtherYou currently has the chevalier-specific provider in `api/src/services/computer/chevalier.rs`.
- OtherYou provider DTOs and trait live in `api/src/services/computer/mod.rs`.
- Chevalier already owns sandbox facade types in `sandbox/crates/sandbox/src/lib.rs`.

### 2) VFS Protocol and Router Contract

Move the VFS protocol contract into chevalier.

Chevalier should own:

- route names and methods
- query parameters
- write headers
- lease request/response DTOs
- stat/list/read DTOs
- range read semantics
- client error mapping
- FUSE operation names and surface-kind constants

Current generic chevalier contract names:

- VFS headers: `x-chevalier-vfs-component`, `x-chevalier-vfs-surface-kind`, `x-chevalier-vfs-operation`, `x-chevalier-vfs-resource-key`, `x-chevalier-vfs-lock-owner-token`
- VFS operation names: `vfs_write_through`, `vfs_setattr_size`, `vfs_mkdir`, `vfs_unlink`, `vfs_rmdir`, `vfs_rename`
- VFS surface kinds: `vm_shared_vfs`, `vm_workspace_vfs`
- Public ingress owner headers: `x-chevalier-owner-bearer`, `x-chevalier-owner-cookie`, `x-chevalier-ingress-token`
- Public ingress token query parameter: `chevalier_ingress_token`

OtherYou should provide callbacks for:

- authorize internal runtime request
- resolve product scope to logical path
- acquire/release product write lease
- perform read/write/delete/rename against configured storage
- emit product ledger/audit rows

Source evidence:

- OtherYou server routes live in `api/src/routes/runtime_nymfs_internal.rs`.
- Chevalier vmd client lives in `sandbox/vmd/src/fuse/client.rs`.
- Chevalier FUSE implementation lives in `sandbox/vmd/src/fuse/fs.rs`.

### 3) Generic FUSE Server / VFS Gateway

Move the generic HTTP server that backs vmd's FUSE client into chevalier.

This is distinct from moving durable storage mechanics. The server is the product-neutral gateway that accepts vmd FUSE requests, enforces the VFS protocol, parses ranges, acquires write leases, validates declared resource keys, maps errors to HTTP responses, and calls a product backend. The backend remains product-owned until chevalier also owns the durable storage metadata layer.

Preferred placement:

- `sandbox/crates/sandbox/src/vfs.rs` or equivalent for protocol constants, DTOs, and client/server-neutral contract types.
- A separate optional crate such as `sandbox/crates/sandbox-vfs-server`, or an opt-in `vfs-server` feature, for Axum/Tower server integration. The core SDK should not force Axum on consumers that only need the VM client.
- vmd's `RemoteVfsClient` should consume the same route constants and DTOs where practical, so the client and server do not drift.

Chevalier should own:

- generic route shape, method table, and path templates
- optional legacy route aliases configured by a product adapter, not hard-coded as product names
- `VfsPathQuery`, `VfsRenameQuery`, `VfsLeaseAcquireRequest`, `VfsLeaseReleaseRequest`, `VfsLeaseGrant`, `VfsDirEntry`, and `VfsMetadata` DTOs
- VFS header constants, operation constants, surface-kind constants, and default VM runtime component names
- Bearer/internal-service-token authentication for vmd-to-server calls
- request body limits and content-type behavior for raw file responses
- HTTP range parsing, `206 Partial Content`, `Content-Range`, and `Accept-Ranges` behavior
- read path handling for stat, list, full read, and range read
- write path handling for write, truncate/setattr-size, mkdir, unlink, rmdir, and rename
- lease acquire/release request handling
- declared-resource-key validation before mutating writes
- error-to-status mapping for not-found, conflict, unauthorized, bad-request, and internal failures
- scope-path normalization and endpoint joining shared with mount planning and the vmd FUSE client
- read-only mount enforcement at the gateway boundary
- contract tests using an in-memory backend that cover every route and every FUSE operation vmd can emit

OtherYou should keep:

- the backend implementation that maps owner IDs to product entities
- product authorization and product-visible access policy
- current conversation/current task alias semantics
- product resource-key format until a generic coordination namespace is introduced
- product write-zone policy checks
- product mutation ledger rows and audit events
- product DB projections, digest refreshes, and runtime chat/task bindings
- storage adapter implementations until durable VFS storage mechanics move into chevalier
- a compatibility alias for `/v1/internal/nymfs/{nym_id}` that delegates to the chevalier gateway while clients move to `/v1/internal/chevalier/vfs/{owner_id}`

The generic backend trait should expose callbacks equivalent to:

- `authorize_internal_request(owner, token)`
- `resolve_scope_path(owner, requested_path, mount_scope)`
- `stat(owner, resolved_path)`
- `list_dir(owner, resolved_path)`
- `read_all(owner, resolved_path)`
- `read_range(owner, resolved_path, offset, length)`
- `derive_write_scope(owner, resolved_path, operation)`
- `acquire_write_lease(owner, write_scope, mutation_count, component, run_id, reason)`
- `commit_write(owner, resolved_path, write_scope, lease, operation, bytes)`
- `commit_namespace_mutation(owner, from, to, write_scope, lease, operation)`
- `release_write_lease(owner, lease)`

The trait names are not prescribed. The invariant is that chevalier owns HTTP/FUSE correctness and OtherYou owns product semantics behind callbacks.

Cutover steps:

1. Add chevalier VFS protocol constants and DTOs that exactly match the current vmd client and OtherYou route behavior.
2. Add the generic VFS gateway with an in-memory backend test suite.
3. Update vmd `RemoteVfsClient` and FUSE tests to use the shared DTO/constant layer where feasible.
4. Replace OtherYou's direct `runtime_nymfs_internal.rs` handler logic with an `OtherYouVfsBackend` adapter that delegates to `FilesystemCoreService`, `NymFsService`, `CoordinationService`, and runtime chat repos.
5. Mount the generic route under `/v1/internal/chevalier/vfs/{owner_id}` and keep `/v1/internal/nymfs/{nym_id}` as a product compatibility alias.
6. Update `api/src/services/computer/chevalier.rs` to build VFS endpoints through a generic chevalier helper instead of hard-coding `/internal/nymfs/{nym_id}`.
7. Validate local direct mode, distributed mode, node failover, range reads, write flush, truncate, mkdir, unlink, rmdir, rename, lease conflict, and rewarm attach against OtherYou.

### 4) Rclone Evaluation

Rclone is a strong candidate for storage/VFS plumbing, but it is not a drop-in replacement for the current FUSE server boundary.

Relevant rclone capabilities:

- `rclone mount` can mount cloud storage systems through FUSE on Linux, macOS, FreeBSD, and Windows.
- `rclone serve webdav`, `serve http`, and `serve nfs` expose remotes over standard protocols.
- rclone's VFS layer has cache modes `off`, `minimal`, `writes`, and `full`; `writes` and `full` support normal filesystem write patterns better than the default `off`.
- rclone remote control can start, list, and stop serve instances dynamically.

What rclone could replace well:

- object-store read/write translation
- local disk VFS cache behavior
- read-ahead and sparse-file caching
- standard protocol serving for plain storage scopes
- some future durable storage adapter implementation behind the chevalier VFS backend trait

What rclone does not replace by itself:

- product write-zone policy
- per-action lease/fence consumption
- mutation ledger and audit row creation
- current conversation/current task alias resolution
- product task binding lookup
- generic chevalier VFS operation names and surface-kind headers
- declared-resource-key validation from the VM FUSE client
- endpoint/mount contract comparison used by the VM runtime facade

Rclone-specific risks for this migration:

- With VFS write caching, writes are flushed after close/write-back timing rather than at the exact product mutation boundary. That does not naturally match OtherYou's action-counted lease fence.
- rclone warns against overlapping remotes sharing the same VFS cache when cache mode is above `off`; distributed nodes would need explicit per-owner/per-node cache isolation.
- Standard WebDAV/NFS/FUSE clients will not carry chevalier's operation, surface-kind, resource-key, and lock-owner headers without an adapter.
- Using `rclone mount` directly would move the mount endpoint out of the existing vmd `RemoteVfsClient` protocol and would still require a sidecar or wrapper for product ledger/fence semantics.

Decision:

- Move the generic HTTP FUSE server into chevalier first.
- Keep rclone as an evaluated implementation option for a future `VfsStorageAdapter` or cache layer, not as the first replacement for the server contract.
- A rclone proof of concept is acceptable only if it sits behind the same chevalier VFS gateway and passes the same lease, ledger, range-read, write-flush, truncate, namespace-mutation, failover, and rewarm tests as the in-process backend.

Rclone references checked for this decision:

- `https://rclone.org/commands/rclone_mount/`
- `https://rclone.org/commands/rclone_serve_webdav/`
- `https://rclone.org/commands/rclone_serve_http/`
- `https://rclone.org/commands/rclone_serve_nfs/`
- `https://rclone.org/rc/`
- Source pass: rclone `v1.74.2`, especially `vfs/vfscommon/options.go`, `vfs/dir.go`, `vfs/read.go`, `vfs/write.go`, `vfs/read_write.go`, `vfs/vfscache/item.go`, `vfs/vfscache/downloaders/downloaders.go`, `vfs/vfscache/writeback/writeback.go`, and `cmd/mount/handle.go`.

### 5) Rclone-Informed VFS/FUSE Optimization Targets

Our current implementation is already optimized in the areas that matter most to OtherYou's workload: GCS metadata TTLs are split by lifecycle, layout paths get long-lived metadata caching, known-missing paths are negative-cached, pack bytes are LRU-cached, small files are cached in memory, directory listings are short-TTL cached, and writes are fenced through product leases before committing. Rclone still exposes useful optimization targets for the FUSE client and the future generic chevalier gateway.

2026-06-05 source follow-up:

- Rclone latest stable is v1.74.2. The relevant source paths are `vfs/vfs.md`, `vfs/vfscommon/options.go`, `vfs/read.go`, `vfs/read_write.go`, `vfs/vfscache/item.go`, `vfs/vfscache/downloaders/downloaders.go`, `fs/chunkedreader/sequential.go`, `fs/chunkedreader/parallel.go`, and `fs/asyncreader/asyncreader.go`.
- The useful rclone pattern is not the top-level provider shape; it is the local VFS cache machinery: sparse files, present-range tracking, short in-sequence read/write waits, reusable background downloaders, configurable memory/disk read-ahead, chunk growth, optional parallel chunk streams, handle grace caching, and explicit cache/writeback observability.
- Our server boundary should stay. Rclone does not naturally carry the chevalier VFS headers, declared resource keys, action-counted owner tokens, product surface kind, or mutation ledger semantics. Replacing the gateway with WebDAV/NFS/rclone mount would reintroduce adapter glue at the wrong boundary.
- Our highest-risk current gap versus rclone is large-file and random-write behavior in vmd FUSE: large reads ask the gateway for exactly the FUSE request size, and dirty write handles keep the full file image in memory. OtherYou's GCS adapter is already optimized behind the gateway, but vmd can still create avoidable request count and memory pressure.

High-confidence targets:

- Add range-aware large-file caching in vmd FUSE.
  - Rclone's cache metadata tracks which byte ranges are present and fills missing ranges with downloader workers plus read-ahead. Our vmd FUSE cache is currently whole-file only for files under `LARGE_FILE_BYTES`; larger files fall through to independent HTTP range reads. A chevalier `RemoteFuseRangeCache` should cache byte windows per open file or per path, track present ranges, and serve repeated/sequential reads without round-tripping to the gateway for each FUSE read.

- Add adaptive read-ahead and sequential-read coalescing.
  - Rclone waits briefly for near-sequential reads before treating them as seeks, then reuses or starts a downloader within a read window. vmd should recognize sequential FUSE reads, request larger backend ranges than the kernel asked for, and reuse that window. This targets desktop/browser asset reads and large generated artifacts without changing the backend contract.

- Add handle grace caching for clean read handles.
  - Rclone keeps clean cached handles/downloaders alive briefly after close, so immediate reopen patterns avoid re-opening remote readers. vmd currently drops open-handle state on release and only retains path-level whole-file cache for small files. A short clean-handle grace period can reuse range cache/download state across duplicate FUSE open/close cycles without delaying dirty writes.

- Preserve virtual directory entries for in-flight writes.
  - Rclone keeps dirty/uploading files visible in directory listings even before the remote listing reflects them. vmd already writes through on flush and invalidates parent caches, but the generic gateway should expose just-written files immediately from a pending/dirty overlay so guest `create -> readdir -> stat` sequences stay consistent even under backend listing lag.

- Split write strategies by workload.
  - Rclone uses stream-write handles for simple write-only cases and disk-backed read-write handles for random writes, append, truncate, and cache-backed read/write. vmd currently loads full file contents into a memory buffer before write/truncate. Keep the memory path for small files, but add a configurable disk-spooled handle for large or sparse/random writes so one guest write cannot allocate the whole file in vmd memory.

- Add explicit VFS cache observability.
  - Rclone exposes cache stats and writeback queue state through rc. Chevalier should expose counters for directory cache hits, small-file cache hits, large-range cache hits, backend range fetches, lease acquire latency, flush latency, dirty bytes, open handles, and eviction reason. This matters more than rclone replacement because optimization targets need production evidence.

Targets that are already mostly covered:

- Directory listing caching exists in vmd's `RemoteFuseCache` and OtherYou's GCS adapter.
- Small-file byte caching exists in vmd and OtherYou's GCS adapter.
- Pack-level caching and prefetch exist in OtherYou's GCS adapter.
- HTTP range reads already exist in the generic protocol and OtherYou route.
- Kernel writeback capability is already requested by writable vmd FUSE mounts.
- GCS metadata and pack lifecycle are already heavily optimized by the OtherYou adapter.

Targets to avoid or defer:

- Do not adopt rclone's delayed writeback as the default. OtherYou's product ledger and write lease consume the mutation at commit time; delaying upload after close would break the current action-counted fence unless the gateway also owns a durable pending-mutation journal.
- Do not replace the chevalier VFS protocol with WebDAV/NFS. Standard protocols do not carry chevalier's surface kind, operation, declared resource key, owner token, or product lease metadata.
- Do not use one shared cache root across distributed nodes. Rclone warns that cache eviction and in-use files are local concerns; distributed Chevalier nodes need per-node/per-owner cache isolation or explicit shared-cache coordination.
- Do not make rclone a hard dependency of the generic SDK. It can be a backend implementation behind the gateway once it passes the same protocol and product semantics.

Implementation order for the FUSE-server move:

1. Move constants/DTOs into chevalier, then make vmd and the gateway share them.
2. Add gateway contract tests for range reads and write/namespace mutations.
3. Add vmd range-cache instrumentation without changing behavior.
4. Implement adaptive read-ahead/range caching behind a config flag.
5. Add disk-spooled large-write handles behind a size threshold.
6. Add virtual-entry overlay support in the gateway backend contract.
7. Validate with OtherYou on corvidae using asset streaming, large file reads, random writes, truncate, mkdir/unlink/rmdir/rename, lease conflicts, failover, and rewarm.

### 6) Shared Mount Planning

Move shared mount planning into chevalier.

Chevalier should own:

- mapping mount intents to `SharedMount`
- mount tags
- guest paths
- backend profiles
- availability and continuity classes
- FUSE-backed versus host-path-backed behavior
- required shared mount profile validation
- stale mount-signature comparison

OtherYou should keep:

- deciding which conversation/task/skills scopes are active
- resolving current conversation and current task aliases
- product naming of runtime scopes

Source evidence:

- OtherYou builds product VFS shared mounts in `api/src/services/computer/chevalier.rs`.
- OtherYou has mount constants and backend contract in `api/src/services/runtime/nym_fs_mount_contract.rs`.
- Chevalier shared mount DTO/proto lives in `sandbox/crates/sandbox/src/lib.rs` and `sandbox/proto/bracket/vmd/v1/vmd.proto`.

### 7) Durable VFS Storage Mechanics

Move durable VFS storage mechanics into chevalier.

Chevalier should own:

- object-store abstraction
- GCS/object storage implementation
- pack format
- pack cache
- bulk pack writes
- range reads from pack slots
- metadata-backed directory listing strategy
- pack compaction and orphan sweeping
- storage backend lifecycle tests

OtherYou should keep:

- product path layout
- write-zone policy intent
- product/task mutation ledger rows
- product projections for chat, memory, context, and task artifacts
- product-specific SQL schema until a shared storage metadata backend exists

Source evidence:

- Adapter contract lives in `api/src/services/runtime/nym_fs_adapter.rs`.
- Pack format lives in `api/src/services/runtime/nym_fs_pack.rs`.
- Pack cache lives in `api/src/services/runtime/nym_fs_pack_cache.rs`.
- GCS-backed adapter lives in `api/src/services/runtime/gcs_nym_fs_adapter.rs`.
- Pack compactor lives in `api/src/services/runtime/nym_fs_pack_compactor_service.rs`.

### 8) Checkpoint and Restore Contract

Move provider-level checkpoint and restore contract types into chevalier.

Chevalier should own:

- VM checkpoint manifest schema
- restore policy classes
- required distributed restore fields
- filesystem reference hash shape
- provider snapshot identity
- resume validation and error classification

OtherYou should keep:

- product checkpoint rows
- task-surface attachment rules
- when checkpoints are requested
- product fallback policy for cold task reruns
- writing product-visible checkpoint manifests into the product VFS until storage metadata is generalized

Source evidence:

- OtherYou builds VM checkpoint manifests in `api/src/services/runtime/vm_runtime_service.rs`.
- OtherYou computes external filesystem checkpoint references in `api/src/services/runtime/nym_fs_checkpoint_reference.rs`.

### 9) Public Ingress and Network Runtime Metadata

Move provider-facing ingress and network metadata contracts into chevalier.

Chevalier should own:

- vmd metadata keys
- public ingress proxy behavior
- auth callback transport
- denied sandbox control ports
- guest-port forwarding semantics
- network enforcement metadata shape
- egress snapshot schema produced by the runtime

OtherYou should keep:

- hostname allocation
- user authorization
- ingress exposure DB rows
- product API DTOs
- product audit logs
- deployment-specific external ingress reconciliation, until chevalier offers a generic deployment plugin for it

Source evidence:

- OtherYou ingress service lives in `api/src/services/nym_vm_ingress_service.rs`.
- OtherYou network policy service lives in `api/src/services/nym_network_policy_service.rs`.
- Chevalier vmd public ingress proxy lives in `sandbox/vmd/src/public_ingress.rs`.
- Chevalier firewall and DNS-adjacent runtime enforcement lives in `sandbox/vmd/src/network/firewall.rs`.

## Keep In OtherYou

These are product-owned and should not migrate into chevalier:

- product entity user ownership and authorization
- conversation, task, and skill product models
- product path layout and content semantics
- write-zone policy intent
- mutation ledger and audit rows
- UI/API DTOs for product runtime surfaces
- task binding rows and runtime chat queries
- current conversation/current task alias selection
- product decisions about which scopes exist

## Completed First Interface Implementation Pass

This pass migrated the active OtherYou integration to the generic chevalier SDK boundary and validated the Linux runtime on corvidae. It did not complete the FUSE-server extraction; `api/src/routes/runtime_nymfs_internal.rs` still owns generic HTTP gateway behavior that must move into chevalier for the migration to be complete.

Chevalier changes:

- Added generic VM/VFS SDK helpers in `sandbox/crates/sandbox/src/vm.rs`.
- Exposed the generic module from `sandbox/crates/sandbox/src/lib.rs`.
- Kept vmd local-first: CoreDNS/Envoy are not required unless guest network/proxy services are explicitly configured.
- Fixed CLI disable flags so disabled node registry/control bus modes stay disabled.
- Made d2vm use the public converter image by default and require explicit `CHEVALIER_SANDBOX_D2VM_INCLUDE_BOOTSTRAP=true` for the legacy private bootstrap flag.
- Passed Docker API version through d2vm conversion via `CHEVALIER_SANDBOX_D2VM_DOCKER_API_VERSION` or `DOCKER_API_VERSION`.
- Added `CHEVALIER_SANDBOX_ENVOY_BASE_ID` so two test nodes on one Linux host can run Envoy without shared base-id socket conflicts.
- Fixed distributed ownership-fence defaults: when `CHEVALIER_SANDBOX_ETCD_PREFIX` is set and `CHEVALIER_SANDBOX_CONTROL_DEDUPE_PREFIX` is not, vmd now derives `<registry-prefix>/command-dedupe`, which makes control-bus fence reads use the same session-fence namespace as the SDK.
- Moved transient QMP and virtiofsd Unix sockets to `/tmp/chevalier-vmd/<vm-id>/...` by default, preserving VM data layout while avoiding Linux `sun_path` length failures from long data roots.

OtherYou changes:

- Kept the product adapter in `api/src/services/computer/chevalier.rs`.
- Added/kept `api/src/bin/computer_repro.rs` as an OtherYou-owned repro harness for direct, distributed, attach, keep, exec, and stream checks.
- Accepted generic `x-chevalier-*` VFS and ingress headers in OtherYou routes while preserving legacy compatibility inside OtherYou. This first-pass route implementation was a temporary OtherYou-owned gateway; the extraction pass below replaces it with a chevalier-owned gateway plus an OtherYou backend adapter.
- Updated local VMD scripts so network services are opt-in instead of required for macOS local development.
- Updated OVH VMD deployment env to set `CHEVALIER_SANDBOX_D2VM_INCLUDE_BOOTSTRAP=true`, preserving the current private guest bootstrap contract while chevalier defaults remain generic.

## Completed Gateway Extraction Pass

This pass moves the generic HTTP/FUSE gateway behavior into chevalier while keeping the product adapter in OtherYou.

Chevalier changes:

- Added `sandbox/crates/sandbox/src/vfs.rs` with generic VFS protocol headers, DTOs, endpoint helpers, scope-path joining, range parsing, gateway errors, and an opt-in `vfs-server` Axum route builder.
- Added `chevalier-sandbox` feature `vfs-server`, keeping Axum and async-trait out of the default VM client surface unless a consumer needs the HTTP gateway.
- Added `rust` feature `sandbox-vfs-server` so consumers can access the gateway through `chevalier_agentic::sandbox`.
- Updated vmd's FUSE client/cache/fs code to use the shared chevalier VFS DTOs, header constants, operation names, surface-kind constants, and scope path helper.
- Kept the vmd FUSE server/client generic: no OtherYou, product, or legacy `x-nymfs-*` names exist in the new chevalier VFS module or vmd FUSE protocol code.

OtherYou changes:

- Replaced direct gateway route handlers in `api/src/routes/runtime_nymfs_internal.rs` with an OtherYou-owned `OtherYouVfsBackend` adapter implementing `VfsGatewayBackend`.
- Mounted the generic chevalier route at `/v1/internal/chevalier/vfs/{owner_id}`.
- Kept `/v1/internal/nymfs/{nym_id}` as a compatibility alias using the same chevalier gateway route builder and the same OtherYou backend.
- Kept product path aliases, task/conversation scope derivation, product write leases, mutation ledger calls, auth, and ingress authorization in OtherYou.
- Updated `api/src/services/computer/chevalier.rs` to build new VFS endpoints through `chevalier_agentic::sandbox::vfs::owner_vfs_endpoint`, so new VM mounts target `/v1/internal/chevalier/vfs/{owner_id}` instead of hard-coding the legacy product route.
- Fixed the GCS-backed VFS adapter startup path so blocking cloud-client construction runs safely inside the Tokio runtime.
- Fixed current-task mount binding so `.runtime/current-task-workspace` resolves from actual task anchor entries, prefers the active task, and falls back to the latest nonterminal task anchor after pause/resume clears `active_task_id`.
- Classified corvidae's unsupported QEMU background-snapshot response and preserved `live-pause-resume` checkpoints by pausing the VM without inventing a snapshot identity when the host kernel cannot background-snapshot.

Local validation for this pass:

- `rustup run 1.90-aarch64-apple-darwin cargo test -p chevalier-sandbox --features vfs-server vfs::` covering endpoint helpers, range parsing, stat, list, full read, range read, write, setattr-size operation forwarding, mkdir, unlink, rmdir, rename, lease acquire/release, declared-resource-key mismatch, missing owner token, backend read-only rejection, and stale lease rejection propagation.
- `cargo test -p chevalier-sandbox vfs::`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p vmd fuse::cache::tests::`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p vmd fuse::fs::tests::`
- `rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path api/Cargo.toml`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml routes::runtime_nymfs_internal`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml services::computer::chevalier::tests::build_nymfs_shared_mounts_uses_shared_storage_contract_for_gcs`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml background_snapshot_unsupported_status_matches_corvidae_qmp_error`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml runtime_internal -- --nocapture`
- `git diff --check`
- `git -C /Users/crow/SoftwareProjects/OtherYou diff --check`

## Final Moved-Gateway Corvidae Validation

This validation used the moved generic chevalier VFS gateway in OtherYou, not the old product-owned gateway implementation.

Setup:

- Local OtherYou API served product routes on `127.0.0.1:3002`.
- Reverse SSH forwarded corvidae's service callback `127.0.0.1:3001` to the local API, so VMD FUSE callbacks hit `/v1/internal/chevalier/vfs/{owner_id}`.
- Corvidae VMD nodes ran the rebuilt chevalier image with GCS VFS support, host networking, KVM, FUSE, CoreDNS, Envoy, etcd, and NATS.
- Test bucket: `chevalier-otheryou-vfs-vm-migration-20260605-b0f78571`.
- Test prefix: `otheryou-corvidae-vfs-vm-migration`.
- Main moved-gateway owner: `363aef8c-dba2-4451-a0d0-07802d474b37`.
- Main task anchor: `f4e165e5-9c43-4541-9bd5-037525636cac`.
- Main VM: `d768b3d1-1b88-474e-a822-e33e52d372df` on corvidae node1.

OtherYou product API scenarios:

- First `/computer/connect` booted the VM through the product API in `7.055436s`; guest exec readiness completed in about `3.51s`.
- The VM mounted four `gcs-vfs-fuse` shared mounts: `/nym`, `/nym/vm/mounts/shared`, `/nym/vm/mounts/task`, and `/nym/vm/mounts/skills`.
- New mounts used the generic chevalier VFS endpoint shape: `/v1/internal/chevalier/vfs/{owner_id}`.
- Warm `/computer/connect` after the VM was already running returned in `0.276211s`.

OtherYou agent/MCP scenario:

- The agent-facing tool surface exposed `nym_computer_exec`, `nym_computer_pause_surface`, and `nym_computer_resume_surface`.
- `nym_computer_exec` ran through OtherYou's product runtime to the corvidae VM and wrote through FUSE:
  - command: `dd if=/dev/zero of=/nym/vm/mounts/task/agent-vfs-patched-dd.bin bs=16 count=1`
  - result: exit code `0`
- Generic VFS stat through `/v1/internal/chevalier/vfs/{owner_id}/stat?path=.runtime/current-task-workspace/agent-vfs-patched-dd.bin` returned a file with `size_bytes: 16`.
- VM-side stat after the write returned `regular file:16`.

Pause, rewarm, and resume scenario:

- `nym_computer_pause_surface` hit corvidae's QMP background-snapshot limitation: `Background-snapshot is not supported by host kernel`.
- The provider fallback produced checkpoint `d6baee96-29c8-4053-8392-3c04e9b573f6` with `snapshot_mode: live_pause_resume`, no synthetic snapshot id, and VM state `paused`.
- `nym_computer_resume_surface` resumed the same VM and returned state `running`.
- VM-side stat after resume still returned `regular file:16`, proving the current-task GCS/FUSE alias re-resolved after cooling.

Linux control-plane checks:

- Both corvidae VMD containers remained healthy after the moved-gateway run.
- Envoy admin for node1 and node2 reported `state=LIVE`.
- CoreDNS on node1 and node2 resolved external names from the Linux host.
- Earlier corvidae matrix runs on the same rebuilt VMD image covered distributed node1, distributed node2, node2-only with node1 stopped, direct node2 mode, stream forwarding, DNS/proxy egress, VM stop/start rewarm, and warm attach.

## Bugs Found And Fixed

- Distributed exec fence mismatch:
  - Symptom: OtherYou distributed exec reached vmd but the command went to the DLQ with `ownership_fence_conflict`.
  - Cause: SDK wrote `/chevalier-sandbox-otheryou/session_fences/...`; vmd read fences through the default `/sandbox/command-dedupe`-derived namespace.
  - Fix: derive the control dedupe prefix from `CHEVALIER_SANDBOX_ETCD_PREFIX` when an explicit control dedupe prefix is absent.

- Multi-node Envoy on one Linux host:
  - Symptom: the second node's Envoy could fail on shared base-id runtime sockets.
  - Fix: added `CHEVALIER_SANDBOX_ENVOY_BASE_ID` and pass `--base-id` to Envoy.

- Long QMP socket path:
  - Symptom: node2 failed to boot a VM with `UNIX socket path ... is too long` for `<long-data-dir>/<uuid>/qmp.sock`.
  - Fix: store transient QMP and virtiofsd sockets under `/tmp/chevalier-vmd/<vm-id>/...`, with runtime-dir ownership prepared for the configured QEMU uid/gid.

- OVH private d2vm bootstrap:
  - Symptom: public d2vm conversion booted but the current private guest contract lacked the portproxy bootstrap path.
  - Fix: chevalier defaults do not pass the private flag, and OtherYou OVH explicitly opts in with `CHEVALIER_SANDBOX_D2VM_INCLUDE_BOOTSTRAP=true`.

- Nested Docker data path:
  - Symptom during corvidae testing: d2vm wrote nested-Docker output to the host path while vmd expected the container path.
  - Fix in test deployment: use identity mounts for VMD data/staging paths, matching the OVH systemd template.

- GCS adapter initialization inside Tokio:
  - Symptom: the local OtherYou API could panic on startup when constructing the blocking GCS client inside the async runtime.
  - Fix: build the blocking GCS client through `tokio::task::block_in_place` when a runtime handle is present.

- Current-task alias selected non-task assistant rows:
  - Symptom: guest writes under `/nym/vm/mounts/task` failed with directory-shaped aliases and `EISDIR` behavior.
  - Cause: mount binding lookup could select later assistant entries such as approval requests instead of task anchor entries.
  - Fix: bind current-task aliases only from `task_placeholder` and `heartbeat_system` entries with task ids.

- Current-task alias after pause/resume:
  - Symptom: after pause/resume, `/nym/vm/mounts/task` could fall back to the placeholder directory because the conversation no longer had `active_task_id`.
  - Fix: prefer the active task when present, then fall back to the latest nonterminal task anchor for that owner.

- Corvidae background snapshot unsupported:
  - Symptom: `nym_computer_pause_surface` failed because the host kernel did not support QEMU background snapshots.
  - Fix: classify the QMP error and preserve the `live-pause-resume` policy by pausing the VM and recording a checkpoint without snapshot id/name.

## Spec Completion Criteria

- [x] The missing FUSE-server move is explicitly specified as a required migration item.
- [x] The spec separates generic HTTP/FUSE gateway behavior from product-owned backend callbacks.
- [x] The spec identifies concrete source files that currently own client, server, storage, and product adapter behavior.
- [x] The spec includes cutover steps for chevalier, vmd, and OtherYou.
- [x] The spec includes acceptance tests for read, write, range, truncate, namespace mutations, lease conflicts, distributed failover, and rewarm.
- [x] The spec evaluates keeping the chevalier gateway versus replacing it with rclone and records the decision.

## First Interface Pass Acceptance Criteria

- [x] OtherYou can request VM sessions through the generic chevalier SDK boundary without owning distributed control wiring.
- [x] OtherYou adapter code lives in OtherYou.
- [x] Chevalier generic SDK code has no OtherYou-specific adapter naming.
- [x] OtherYou accepts generic chevalier VFS/ingress headers and keeps legacy product compatibility inside OtherYou.
- [x] OtherYou local mode can be configured without CoreDNS, Envoy, firewall, or distributed machinery being required on macOS development hosts.
- [x] Distributed deployment remains available through chevalier configuration, with OtherYou selecting endpoints/profile rather than building control-plane machinery.
- [x] Linux CoreDNS, Envoy, TAP networking, public ingress, direct mode, distributed mode, node2-only mode, and rewarm attach were validated on corvidae.
- [x] Existing OtherYou product behavior remains product-owned: authorization, task binding, mutation ledger, checkpoint rows, ingress rows, and audit logs stay in OtherYou.

## Complete Migration Acceptance Criteria

The migration is not complete until these are true:

- [x] Chevalier owns the generic HTTP FUSE server/VFS gateway instead of OtherYou route code owning it.
- [x] vmd's `RemoteVfsClient` and the chevalier gateway share one protocol/DTO/constant source where practical.
- [x] OtherYou implements a product backend adapter for the chevalier gateway.
- [x] OtherYou's `/v1/internal/nymfs/{nym_id}` path is only a compatibility alias that delegates to the generic chevalier route.
- [x] OtherYou's chevalier computer provider builds VFS endpoints through a generic chevalier helper, not by hard-coding the product route shape.
- [x] Gateway contract tests cover stat, list, full read, range read, write flush, truncate/setattr-size, mkdir, unlink, rmdir, rename, lease acquire/release, declared-resource-key mismatch, stale lease token, and read-only mount rejection.
- [x] OtherYou integration tests prove generic-route and legacy-alias parity.
- [x] Corvidae validation proves the VM migration across local direct, distributed node1, distributed node2, node2-only, failover-style node removal, and rewarm scenarios, with a post-move gateway replay through OtherYou covering product API connect, agent exec, GCS-backed FUSE write/stat, pause/resume, CoreDNS, Envoy, and warm attach.
- [x] No rclone-backed proof of concept was selected for this migration. Any future rclone-backed implementation must sit behind the chevalier gateway and pass the same gateway contract tests before it can replace the in-process backend.

Longer-term extraction still allowed by this spec, but not required for the FUSE-server move:

- Move durable VFS object-store pack/cache/compaction internals from OtherYou into chevalier once chevalier owns a generic storage metadata backend.
- Move product-visible checkpoint manifest storage into chevalier once product checkpoint rows can reference a generic manifest identity.

## Non-Goals

- Do not move product models into chevalier.
- Do not make chevalier depend on OtherYou DB schemas.
- Do not require distributed mode for local development.
- Do not make OVH, GKE, or any deployment-specific infrastructure a hard dependency of the product API.
- Do not redesign product path semantics as part of the first extraction.

## Verification Matrix

Local tests run in this pass:

- `rustup run 1.90-aarch64-apple-darwin cargo fmt -p vmd`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p chevalier-sandbox vm:: -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p chevalier-sandbox session_create_forwards_shared_mounts_contract -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p vmd runtime_socket_path_stays_short_for_long_vm_data_roots -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p vmd config::tests::control_dedupe_prefix_defaults_to_registry_prefix -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p vmd build_qemu_args -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test -p vmd virt::tests::run_d2vm -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path api/Cargo.toml --bin computer_repro`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml services::computer::chevalier -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml routes::runtime_nymfs_internal -- --nocapture`
- `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml routes::vm_ingress -- --nocapture`

Corvidae Linux validation:

- Built `chevalier-vmd:otheryou-migration-test` from the updated chevalier sandbox with public d2vm, Envoy, and CoreDNS images.
- Ran OVH-shaped privileged Docker VMD nodes on `crow@corvidae` with host networking, KVM, FUSE, Docker socket, identity-mounted VM data paths, CoreDNS, Envoy, public ingress, etcd, and NATS.
- Verified node sockets:
  - node1: VMD `8052`, public ingress `8081`, CoreDNS `15053`, Envoy proxy `3128`, Envoy admin `9901`
  - node2: VMD `8053`, public ingress `8082`, CoreDNS `16053`, Envoy proxy `3228`, Envoy admin `9902`
- Verified node2-only DNS/proxy egress rules after restarting node2 alone.

Live OtherYou scenarios run against corvidae:

- Distributed node2-primary session:
  - `stage=session ok elapsed_ms=629`
  - `stage=exec ok elapsed_ms=3183`
  - stdout `distributed-node2-qmp-ok`
  - stream forward/TCP/HTTP probe passed
- Distributed kept session and rewarm:
  - keep session create `elapsed_ms=570`
  - initial exec `elapsed_ms=3044`
  - stop VM `real 5.928s`
  - start VM `real 0.427s`
  - attach after rewarm `elapsed_ms=126`
  - exec after rewarm `elapsed_ms=254`
  - HTTP stream probe after rewarm passed on first attempt
- Original distributed node1 path:
  - `stage=session ok elapsed_ms=578`
  - `stage=exec ok elapsed_ms=2836`
  - stdout `distributed-node1-original-path-ok`
  - stream forward/TCP/HTTP probe passed
- Node2-only distributed mode with node1 stopped:
  - `stage=session ok elapsed_ms=552`
  - `stage=exec ok elapsed_ms=3184`
  - stdout `distributed-node2-only-ok`
  - stream forward/TCP/HTTP probe passed
- Direct non-distributed node2 mode:
  - `stage=session ok elapsed_ms=3878`
  - `stage=exec ok elapsed_ms=93`
  - stdout `direct-node2-public-api-ok`
  - stream forward/TCP/HTTP probe passed

Operational notes from validation:

- Running two privileged VMD nodes on one Linux host shares iptables state. This is useful for test coverage, but real OVH nodes should run one node per host or own isolated network namespaces. When node1 was stopped, its cleanup removed shared chains; restarting node2 alone restored the expected node2 rules.
- The current private OtherYou guest image still depends on the legacy d2vm bootstrap flag. That dependency is now explicit in OtherYou OVH config instead of being hidden in chevalier defaults.
- Endpoint override validation showed that forwarding only static VMD ports is not enough for live guest readiness and stream checks, because VMD allocates dynamic host ports for guest portproxy. For laptop-to-corvidae validation, use direct corvidae VMD endpoints or forward the dynamic range intentionally.
