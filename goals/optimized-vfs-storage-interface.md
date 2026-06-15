# Goal: One Optimized VFS Storage Interface

Build one canonical, optimized VFS storage interface in a Chevalier VFS namespace/crate that every VFS consumer uses, whether the implementation is local/in-process or remote over the Chevalier VFS gateway.

This is the next migration stage after the generic VFS gateway/FUSE boundary. The current intermediate step proves the gateway shape works. The next step is to move the durable VFS storage mechanics into Chevalier without losing the performance and correctness properties already optimized in OtherYou.

Current-state assumption: the optimized durable VFS storage path still lives in OtherYou today. Chevalier currently owns the generic HTTP gateway, VMD/FUSE client behavior, VM lifecycle, and distributed control mechanics. This goal is not to replace that working storage path with a new design; it is to lift the proven storage primitives into a Chevalier VFS namespace behind one interface.

Namespace requirement: generic VFS storage must be a sibling Chevalier namespace, not a child of `chevalier-sandbox`. `chevalier-vfs` should own pack/manifest/storage primitives and optimized local/remote storage interfaces. `chevalier-sandbox` should remain a consumer/integration layer for VM, FUSE, and gateway behavior.

## Core Direction

`chevalier-vfs` should own generic VFS storage mechanics:

- file manifests and file versions
- pack layout and pack metadata
- zstd slot compression/decompression
- batch reads
- atomic bulk writes
- range reads
- precondition-aware writes
- content hashes
- cache warming and subtree prefetch
- compaction and garbage collection hooks
- backend adapters for local filesystem/object-store and GCS first, with room for future stores such as S3/R2

Packed storage requires a manifest/index layer because pack objects alone do not reveal the current logical file head or the slot coordinates for a path. That index must be an interface owned by `chevalier-vfs`, not an assumption that VFS always has a product database. Chevalier should ship a batteries-included Postgres-backed manifest/index implementation as the default serious path for new adopters, because Postgres is common, operationally familiar, and strong enough for correctness-sensitive compare-and-swap updates. That default Postgres implementation must be extracted from and reconciled toward OtherYou's current production VFS implementation, not independently reinvented. OtherYou's implementation is the source of truth for behavior, schema semantics, query shape, transaction boundaries, batching, and performance until parity is proven. Once lifted, OtherYou should consume the Chevalier implementation instead of continuing to own a bespoke copy. Local/dev Chevalier can also use filesystem metadata or sidecar manifests; gateway-backed Chevalier can delegate the index to the serving backend.

The default Postgres implementation should be a real implementation, not sample glue:

- migrations or schema DDL for scopes, entries, current manifests, file versions, pack records, and optional mutation/audit rows
- batched lookup methods matching the `chevalier-vfs` manifest/index trait
- compare-and-swap commit semantics for packed writes
- transactional pack + manifest + current-head updates
- query plans and data-shape decisions carried forward from the known-fast OtherYou path unless a measured change proves better
- examples/config that let a new Chevalier user start with local Postgres and object/file storage without writing their own index layer

This default must remain an implementation of the generic interface, not a dependency of the core abstraction. Users who do not want Postgres should be able to provide a sidecar, embedded, or gateway-backed index without changing storage semantics.

OtherYou should keep product policy and product meaning:

- user/nym ownership and auth
- logical path mapping
- task and conversation mount scopes
- coordination resource keys
- mutation ledger semantics
- product-specific sidecars such as digest files
- product-visible checkpoint/task/conversation rows

OtherYou should not keep generic VFS storage adapter mechanics after the migration is complete:

- no product-owned local/GCS storage adapter implementation as the canonical VFS path
- no product-owned pack building/extraction, zstd slot layout, pack cache, compaction, or object-store read/write batching
- no product-owned `read_many`/`read_range`/`write_many_atomic` storage implementation except as a thin policy call into `chevalier-vfs`
- no product-owned metadata batching or changed-only write planner except where it is genuinely product policy

OtherYou may keep a small product bridge that supplies account/nym identity, logical mount roots, authorization/lease decisions, mutation ledger recording, and product database/repository access to the generic `chevalier-vfs` implementation. That bridge should not reimplement storage mechanics.

## Interface Requirement

Expose one optimized storage trait/API in `chevalier-vfs` for:

- `stat`
- `stat_many` / `metadata_many`
- `list`
- `list_dir_with_metadata`
- `list_dir_filtered`
- `list_subtree_file_metadata`
- `read`
- `read_range`
- `read_many`
- `read_many_if_etag_mismatch` or equivalent content-bundle reads, if kept generic
- `write`
- `write_many_atomic`
- `write_many_if_changed_atomic`
- `mkdir`
- `delete_file`
- `delete_file_with_metadata`
- `rmdir`
- `rename`
- `rename_with_metadata`
- `prefetch_subtree`
- hash/precondition-aware writes
- storage cleanup/compaction primitives

This is a target storage interface, not merely the current Chevalier `VfsGatewayBackend` trait. The current gateway is per-file HTTP/FUSE plumbing in `chevalier-sandbox`; the target interface lives beside that layer and must also preserve the bulk/batch primitives that currently make OtherYou's storage path fast.

The important missing operations are not POSIX breadth. They are the operations OtherYou currently needs for efficient product behavior:

- subtree file metadata: list logical files under a prefix with manifest-backed hashes, sizes, versions, token counts when available, and enough pack/manifest state to drive later reads
- batched metadata by path: answer many `stat`/manifest/precondition questions in one storage call
- directory listing with metadata and filtering: avoid per-entry hash/stat loops in gateway-backed list paths
- changed-only atomic bulk writes: compare candidate content hashes against stored logical file hashes and only write changed files, without rereading existing file bytes in the caller
- metadata-returning delete/rename: return previous and next logical file metadata needed by product mutation ledgers without extra stat/hash calls
- generic stale-content bundle reads, only if they stay storage-shaped and do not absorb product-specific thread bundle semantics

Do not expand the core interface with helpers that are just caller convenience unless usage proves they need storage-level optimization. `append`, JSON/text wrappers, arbitrary copy, watch, chmod/chown, and product-specific bundle concepts are not part of this goal.

The interface branches internally by deployment mode:

- local/direct mode: in-process adapter, used for local dev and same-process consumers
- remote/gateway mode: client adapter speaking to the Chevalier VFS HTTP gateway exposed by sandbox/product integration

Local and remote must share semantics. They are execution modes under one interface, not separate product paths.

## Performance Requirements

Preserve or improve the old optimized OtherYou behavior:

- bulk reads stay batched
- manifest lookups stay batched
- subtree metadata lookups stay batched
- path metadata lookups stay batched
- bulk writes stay atomic
- changed-only bulk writes avoid reading existing content bytes when a manifest hash comparison is sufficient
- compressed pack range reads remain range-based where possible
- raw reads do not accidentally compute content hashes
- stat/list/freshness paths return stored manifest/object-sync hashes and do not unpack or read pack bytes just to answer metadata
- small-file cache warming still works
- subtree prefetch still reduces repeated pack fetches
- no per-file loops where the old path used batch operations
- no full-pack/object fetch where a slot/range fetch is sufficient

## Correctness Requirements

Preserve old VFS semantics:

- content hashes represent decoded logical file bytes, not whole pack objects
- pack reads use manifest slot coordinates and only decompress/verify the requested slot
- pack-object hashes may exist for pack integrity, debugging, or GC, but they do not replace per-file content hashes for freshness/conflict/cache semantics
- precondition/conflict behavior matches the old path
- lease/resource-key rejection happens before mutation
- write/delete/rename/mkdir/rmdir produce the same mutation/audit behavior once adapted through product policy
- digest sidecars and other product writes still land correctly
- legacy compatibility paths remain until explicitly removed
- local and remote implementations return the same observable errors for the same state

## Migration Shape

1. Extract the pack/manifest/zstd primitives from OtherYou into `chevalier-vfs` without changing behavior.
2. Extract the Postgres-backed manifest/index implementation from OtherYou into `chevalier-vfs`, preserving the production schema semantics, transaction behavior, batching, and query performance.
3. Add a local/direct `chevalier-vfs` implementation and prove it against the old OtherYou test corpus.
4. Add a remote/gateway `chevalier-vfs` implementation that uses the same interface.
5. Adapt OtherYou to provide product policy hooks and consume the lifted Chevalier implementation instead of owning generic storage mechanics.
6. Replace the old OtherYou adapter files/call paths with `chevalier-vfs` consumers once parity is proven.
7. Keep legacy OtherYou routes as compatibility aliases during cutover.
8. Remove duplicated storage logic only after parity is proven.

## Current Slice Status

As of the current branch state, `chevalier-vfs` owns the generic pack format, zstd slot extraction, pack cache, local storage backend, object-store storage backend, GCS object-store adapter, manifest/index trait, pack lifecycle trait, generic pack coalescing/orphan-sweep primitives, and a Postgres-backed index implementation shaped from the OtherYou production path.

OtherYou's metadata-backed GCS adapter now routes the main storage mechanics through `chevalier-vfs` for stat, directory listing with metadata/filtering, full reads, batch reads, range reads, atomic writes, changed-only writes, mkdir, delete, rename, and subtree prefetch. The adapter still keeps product policy and product caches: nym/path projection, legacy loose-object compatibility, protected roots, product mutation semantics, and TTL caches. The remaining single-slot GCS write helper now builds packs through `chevalier-vfs::manifest::build_pack_manifest` and uses the Chevalier pack record when inserting the product pack row, so the adapter no longer calls a product-local `PackBuilder`. The old `nym_fs_pack` compatibility re-export has no active callers and has been removed; product code imports Chevalier pack primitives directly where a bridge still needs slot metadata.

The no-DB GCS branch has been audited. It is legacy compatibility, not a valid durable packed-storage backend: without a manifest/index projection it can upload pack objects and seed fresh process caches, but later reads fall back to loose-object keys and cannot rediscover logical file heads or pack slots. Do not present object-store-only packed writes as a supported Chevalier default. The supported default path should be the lifted Postgres manifest/index implementation, with any future sidecar/embedded manifest implementation treated as a separate backend behind the same `chevalier-vfs` index trait.

Subtree prefetch now returns warmed small-file bytes from `chevalier-vfs`, so pack slot decompression for prefetch warming no longer lives in the product adapter. OtherYou copies that result into its existing TTL byte cache to preserve the old warm-read behavior.

Pack lifecycle is now represented in `chevalier-vfs` as a generic index boundary plus tested generic scope discovery, coalesce, and zero-reference sweep primitives. Coalescing preserves the old storage sequence: select small packs, fetch full source packs, extract zstd slots, rebuild a coalesced pack off the async reactor when Tokio is enabled, upload the new pack before the metadata pivot, then atomically insert the new pack, repoint manifests, and decrement old pack refcounts. Zero-reference sweeping preserves the old cleanup sequence: correct refcount drift, list old zero-reference packs, recount each candidate before deletion, delete the pack object, then delete pack rows for successfully deleted objects. OtherYou implements the lifecycle trait over its existing production repo methods, including the per-nym advisory lock for the compaction pivot. The OtherYou background compactor service now calls the generic candidate discovery, coalescing, and zero-reference sweep primitives through a narrow blob-store adapter while keeping its existing scheduling, per-nym retry loop, tick aggregation, logging envelope, and product config.

The remote/gateway side has also moved from per-file plumbing toward the optimized storage surface. `chevalier-sandbox` currently owns the HTTP/FUSE protocol route set through `VfsGatewayBackend`: filtered list, stat, batch metadata, batch reads, raw/ranged read, subtree metadata, subtree prefetch, single-file write, atomic write-many, namespace mutations, metadata-returning delete/rename, and leases. `metadata_many` preserves request order and missing-entry slots, and OtherYou overrides it to call `filesystem_core_service.metadata_many_async`, which can use the metadata-backed GCS/Postgres batch path. `read_many` does the same for file bodies: the route returns ordered optional byte entries, and OtherYou overrides it to call `filesystem_core_service.read_many_async`, preserving the existing cache-first and DB-backed packed-read behavior. `write_many` accepts one lease/resource-key-protected batch, rejects cross-resource-key writes, and delegates to a backend atomic write-many implementation. OtherYou overrides that route with its existing protected `nymfs_service.write_bulk_bytes` path, so remote batch writes keep the product mutation ledger, commit fence, GCS atomic pack write, retries, and digest refresh behavior. OtherYou also overrides filtered directory listing, subtree metadata, prefetch, delete-with-metadata, and rename-with-metadata with existing `filesystem_core_service`/`nymfs_service` paths so metadata-backed filtering, GCS/Postgres subtree lookups, delete preconditions, and mutation metadata remain server-side instead of falling back to per-file gateway loops. `chevalier-vfs` now has a feature-gated `GatewayVfsStorage` client that implements filtered directory metadata, subtree metadata, prefetch, the read side, lease-protected single write, precondition-aware write-many, changed-only write-many planning, mkdir, delete, rmdir, rename, delete preconditions, and metadata-returning delete/rename over those gateway routes. Focused gateway-client tests cover filter query propagation, scope stripping for subtree metadata, object-state preservation, prefetch warmed-byte mapping, changed-only multi-write skipping of unchanged files before the lease-protected write-many call, write precondition serialization, delete precondition header serialization, and delete/rename metadata responses without client-side stat calls. `chevalier-sandbox` server tests cover precondition deserialization/forwarding on write-many and namespace delete. VM/FUSE mounts can keep using the per-file gateway, and higher-level remote VFS consumers now have the main optimized read/write/list/subtree/prefetch/mutation primitives through the same `chevalier-vfs::OptimizedVfsStorage` interface.

Completion disposition for the current migration slice:

- The current no-DB / loose-object GCS paths are retained only as cutover compatibility. They are not the durable packed VFS backend, and the docs call this out explicitly. Object-store-only packed writes are not presented as supported durable storage.
- The active metadata-backed GCS path calls into `chevalier-vfs::ObjectBackedVfsStorage` for the canonical storage operations. The remaining OtherYou adapter code is a product bridge for path projection, protected-write policy, mutation recording, digest refresh, TTL cache bridging, and compatibility fallback behavior.
- The one remaining direct pack-slot extraction in OtherYou is in the `read_range_async` compatibility fallback after the primary `chevalier-vfs` range-read path misses. It is not the canonical metadata-backed path.
- The full local product validation from the migration goal has been completed with the gitignored local env overlay: GCS-backed VM/FUSE write, packed read, range read, lease rejection, protected cleanup, pause/resume, and warm reattach were validated against a fresh local account/Nym.

## Local Product Validation Evidence

2026-06-13 local validation used the documented OtherYou tmux/corvidae path with the GCS backend enabled and a fresh account/Nym created through the UI:

- Local account created through the app: `vfs.flow.20260613.2229@example.com`.
- Account approved in local Postgres by setting `users.approved_at`.
- Nym created through the first-use UI flow: `Fathom`, id `13285546-f051-49d7-b80e-6b354da90c6a`.
- A short runtime chat task was submitted against the clean local API after fixing the local env overlay so the generated API env exposed `ANTHROPIC_API_KEY`.
- Clean task `278ad6c2-cf32-4db1-8564-f22f9183e8ca`, run `290cb51f-96fc-4e61-8883-274099562d55`, completed and rendered the final durable assistant message `vfs-ok` in the UI.
- The task-created file landed at `conversations/c136f087-03eb-41a1-acf7-521f7d84d074/0008_assistant/vfs_validation.txt`.
- `nym_fs_entries` recorded the file as a GCS-backed VFS file with `size_bytes = 6`.
- `nym_fs_file_manifests` recorded a packed object slot for that file:
  - pack key under `nymfs/packs/13285546-f051-49d7-b80e-6b354da90c6a/...pack`
  - non-zero `pack_slot_offset`
  - `pack_slot_length = 66`
  - `pack_slot_compression = 1`
- Internal Chevalier VFS raw read returned `vfs-ok`.
- Internal Chevalier VFS ranged read `bytes=0-2` returned `vfs`.
- API logs for the completed run show `nym_fs.bulk.atomic_pack_write` and `batch_record_mutations` on the fresh Nym after the runtime task completed.
- Browser validation on the fresh account showed Nym `Fathom` rendering the durable top-level assistant message `vfs-ok` without requiring a page refresh.
- The task-created file has a packed manifest row reachable through `nym_fs_entries.current_version_id -> nym_fs_file_versions.manifest_id -> nym_fs_file_manifests.id`. Do not validate current packed state by joining `nym_fs_entries.content_hash` directly; that field can be null on the current migrated path.
- A no-LLM internal Chevalier VFS validation against the same Nym/conversation verified the protected shared mount path:
  - rejected component `local_vfs_validation` returned `409`
  - the rejected write left `nym_fs_entries` unchanged for that path
  - raw read for the rejected path returned `404`
  - allowed component `vm_runtime` with `vm_shared`/`write` headers wrote `shared-ok`
  - raw read returned `shared-ok`
  - ranged read `bytes=0-5` returned `shared`
  - the file had a GCS packed manifest slot under `nymfs/packs/13285546-f051-49d7-b80e-6b354da90c6a/...pack`
  - metadata-returning delete returned `200`
  - the deleted path had no DB entry and raw read returned `404`
- That validation found and fixed a real ordering bug in the GCS bulk path: protected-write policy was enforced after the storage write could already commit. `NymFsService::write_bulk_bytes_gcs_once` now resolves logical paths and enforces per-item protected-write policy before previous-hash lookup, commit fence, atomic pack write, mutation recording, or digest refresh.
- Local validation artifacts under `conversations/c136f087-03eb-41a1-acf7-521f7d84d074/shared/vfs-*` were deleted through the internal Chevalier VFS API after the check.
- A real LLM-backed developer-mode VM/FUSE validation then exercised the product path through model tool selection, not only direct API calls:
  - task `58ff5646-c5d6-460a-aacc-1d6a2bc5cda5`, run `4e647825-ccbb-4c2f-9762-78e178e12ed1`
  - the model selected `computer_exec` and wrote `llm-vm-vfs-ok` to `/nym/vm/mounts/shared/llm-vfs-validation.txt`
  - the file landed at `conversations/c136f087-03eb-41a1-acf7-521f7d84d074/shared/llm-vfs-validation.txt`
  - internal Chevalier VFS raw read returned `llm-vm-vfs-ok`
  - ranged read `bytes=0-5` returned `llm-vm`
  - the DB manifest chain recorded a GCS packed slot under `nymfs/packs/13285546-f051-49d7-b80e-6b354da90c6a/...pack` with `pack_slot_length = 73`, `pack_slot_compression = 1`, and a decoded logical content hash matching the file content
  - audit events show `runtime.mcp_tool_called` for `computer_exec` with `succeeded`, VM surface write lock acquisition/release, and `runtime.conscious_run_completed`
- A second real LLM-backed validation exercised VM pause/resume and warm reattach around the same mounted shared file:
  - task `e025f07f-a1d6-4320-846c-54e5380aabf0`, run `7c8ea912-f58b-43ef-be41-684df6fc02fd`
  - the model called `computer_exec`, `computer_pause_surface`, `computer_resume_surface`, then `computer_exec` again
  - the task and run both completed
  - the model wrote `llm-vm-resume-ok` to `/nym/vm/mounts/shared/llm-vm-resume-validation.txt`, paused the VM surface, resumed it, and read the same file after resume
  - internal Chevalier gateway raw read returned `llm-vm-resume-ok`; legacy `/internal/nymfs` raw read also returned `llm-vm-resume-ok`
  - ranged reads through both routes returned `llm-vm`
  - the file landed at `conversations/c136f087-03eb-41a1-acf7-521f7d84d074/shared/llm-vm-resume-validation.txt`
  - the DB manifest chain recorded a GCS packed slot under `nymfs/packs/13285546-f051-49d7-b80e-6b354da90c6a/...pack` with `pack_slot_length = 77`, `pack_slot_compression = 1`, and `size_bytes = 17`
  - `nym_surface_checkpoints` recorded a VM checkpoint for surface `43834e14-6987-476f-adff-ee30af56de71`
  - `nym_vm_surfaces` shows the surface in `running` state with `paused_at` and `resumed_at` populated for the validation window
  - audit events show `computer_pause_surface` and `computer_resume_surface` both succeeded between the two successful `computer_exec` calls
- The two LLM-created shared validation files were then cleaned up through the protected Chevalier VFS gateway delete path:
  - one lease was acquired and released per file for resource key `nym:13285546-f051-49d7-b80e-6b354da90c6a:conversation:c136f087-03eb-41a1-acf7-521f7d84d074:shared-overlay`
  - both delete calls returned `204`
  - raw reads for both paths now return `404`
  - `nym_fs_entries` has no remaining rows for either validation path
- Focused automated validation after the fix:
  - `rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path api/Cargo.toml`
  - `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml nym_fs -- --nocapture` passed: 14 passed, 6 ignored legacy loose-blob tests, remaining filtered out
  - `cargo test --manifest-path vfs/Cargo.toml` passed: 39 passed
  - `cargo test --manifest-path sandbox/crates/sandbox/Cargo.toml vfs -- --nocapture` passed: 4 passed, remaining filtered out
  - `git diff --check` passed in both OtherYou and Chevalier
- Feature-gated Chevalier VFS validation after the completion audit:
  - `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path vfs/Cargo.toml --features postgres` passed: 42 passed
  - `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path vfs/Cargo.toml --features gateway` passed: 45 passed
  - `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path vfs/Cargo.toml --features postgres,gateway,gcs,tracing` passed: 48 passed
  - `rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path vfs/Cargo.toml --example postgres_local_object_store --features postgres,tokio` passed
  - the `postgres_local_object_store` example was live-tested against the local Postgres server after applying `migrations/postgres/001_chevalier_vfs_index.sql`; it wrote and read `hello from chevalier-vfs`, then the temporary `chevalier_vfs_*` tables and local object directory were cleaned up
  - `rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path sandbox/crates/sandbox/Cargo.toml --test facade_contract control_gateway_failover_prefers_healthy_secondary_endpoint -- --nocapture` passed
- That feature audit found and fixed two real issues:
  - the Postgres index used an unstable `if let` chain in `delete_file_entry`, so the `postgres` feature did not compile on stable
  - the Postgres batch promote CTE built invalid SQL with manual separator/cast handling; it now uses `QueryBuilder::push_values`, matching the surrounding batch insert style and passing the live Postgres example
- Adopter-facing Chevalier VFS docs and a local Postgres/object-store example were added:
  - `vfs/README.md`
  - `vfs/examples/postgres_local_object_store.rs`
  - the README points to `migrations/postgres/001_chevalier_vfs_index.sql`, explains the object-store + manifest/index split, and shows local Postgres plus local-object-store setup

Local ops findings from that validation:

- `infra/env/api.env.example.local` had the Anthropic key under `NTHROPIC_API_KEY`; the local API expects `ANTHROPIC_API_KEY`. The local gitignored overlay now aliases `ANTHROPIC_API_KEY=${NTHROPIC_API_KEY}`.
- Stale local API listeners on ports `3001`/`3012` can make `ops/dev/run_local_tmux.sh up` appear to start while the new API fails to bind. Before product validation, confirm the listener belongs to the current tmux-managed API process.
- `api --prewarm-computer-only` can take long enough that readiness checks may fail briefly after `run_local_tmux.sh up`; wait for the serving `cargo run --bin api` phase and `/health`.

## Checked Assumptions

- At the start of this goal, the optimized storage mechanics were in OtherYou, not Chevalier. OtherYou owned the GCS adapter, manifests, file versions, pack rows, zstd pack building, batch reads, atomic bulk writes, range reads, subtree prefetch, and compaction. Current branch state has lifted the generic pack/cache/object-store/index/gateway/compaction pieces into `chevalier-vfs`, while OtherYou still owns the product bridge and compatibility cutover paths.
- Current Chevalier VFS code in `chevalier-sandbox` is a gateway/backend boundary for VM/FUSE access. It has per-file methods, range reads through `read_file(..., range)`, namespace mutations, and lease calls, but it is not yet the full optimized storage abstraction described here.
- The existing OtherYou `NymFsAdapter` surface already has the important storage primitives to preserve: `read_many_async`, `read_range_async`, `write_bulk_bytes_atomic_async`, `prefetch_subtree_async`, and write preconditions.
- The pack format is a generic storage primitive: it stores zstd/raw slots, slot offsets/lengths, and per-slot hashes of uncompressed logical file payloads. It has no product policy or database dependency and is a candidate to lift directly.
- DB-backed manifests are an OtherYou implementation choice for the current packed durable path, not a fundamental Chevalier VFS dependency. `chevalier-vfs` should require a manifest/index abstraction for packed storage while allowing DB-backed, local sidecar-backed, or gateway-delegated implementations. Chevalier should nevertheless provide a first-class Postgres-backed index implementation so a new project is not forced to invent one before using packed VFS storage. That Postgres implementation should be based on the production-tested OtherYou implementation and validated by direct parity, not designed as a fresh parallel interpretation.
- Product ownership, logical path mapping, protected write policy, resource-key/lease validation, mutation ledger rows, and product sidecars remain OtherYou policy. They should call into Chevalier storage rather than being absorbed into generic Chevalier storage semantics.
- Existing OtherYou adapter files are migration source material, not the desired final architecture. Their storage responsibilities should either move to `chevalier-vfs` or disappear behind a thin product bridge.

## Test Matrix

Required validation before prod:

- local direct VFS
- remote VFS gateway
- GCS-backed VFS
- VM FUSE reads and writes
- full reads
- range reads
- `read_many`
- `metadata_many`
- `list_dir_with_metadata`
- `list_subtree_file_metadata`
- atomic `write_many`
- changed-only atomic `write_many`
- mkdir/delete/rename/rmdir
- metadata-returning delete/rename
- conflict/precondition failures
- compressed pack reads
- zstd slot extraction
- compaction/GC safety
- subtree prefetch/cache warming
- pause/resume and warm attach
- distributed/HA gateway routing
- legacy route compatibility

## Validation And Cleanup Constraints

Carry forward the operational constraints from `chevalier/migration-goal.md`:

- use the gitignored local env overlay at `/Users/crow/SoftwareProjects/OtherYou/infra/env/api.env.example.local` for local API runs
- do not edit generated env files such as `.run/local-tmux/api.env`
- when API keys are needed for local validation, source the local env path without printing secret values
- prefer `computer_repro`, direct API calls, internal MCP/tool calls, and Rust tests over open-ended LLM-driven validation
- use at most one short Nym/chat/agent run only when product integration cannot be proven otherwise
- avoid long heartbeats, browsing loops, repeated model turns, or any validation pattern that spends unnecessary LLM money
- prove the GCS-backed VFS path cleanly before prod cutover: GCS backend enabled, VM/FUSE mount active, writes visible through canonical VFS/GCS-backed storage, range reads working, leases/resource keys enforced, and pause/resume preserving the VM-visible file
- use the documented local tmux/corvidae/VMD ops path for end-to-end runtime validation
- do not leave local test heartbeats enabled
- stop local tmux/runtime sessions started for validation
- clean temporary test Nyms/tasks/files only where safe, and never mutate `ian@...`/`iantbutler01@...` accounts unless explicitly directed
- leave Chevalier and OtherYou with only intentional code changes and commit the final work

## Definition Of Done

There is exactly one optimized VFS storage abstraction in `chevalier-vfs`. Local and remote are implementation choices under that abstraction. `chevalier-sandbox` consumes the abstraction for VM/FUSE/gateway integration. OtherYou consumes `chevalier-vfs` and no longer has product-owned VFS adapter files as the canonical storage implementation, but still owns product policy and product-specific persistence meaning.

Performance and correctness are proven against the old optimized OtherYou behavior before production deployment.

Chevalier also includes a supported default manifest/index implementation, with Postgres as the recommended batteries-included backend. That implementation is the lifted, generalized form of OtherYou's production-tested VFS index path. A new Chevalier adopter can run packed VFS storage with documented local Postgres setup, migrations/schema, and the same optimized read/write/index semantics without depending on OtherYou code, and OtherYou can consume the Chevalier version without losing the behavior that has already proven itself in production.
