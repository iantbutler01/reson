# OtherYou VFS/VMD Migration Validation Note

Date: 2026-06-13

Branches:

- `chevalier`: `wip/chevalier-vfs-vmd-migration`
- `OtherYou`: `ian/vmd-vfs-chevalier-migration`

Current commits at initial validation:

- `chevalier`: `812bbb6 Harden VMD mount bootstrap`
- `OtherYou`: `cbfb07f Wire OtherYou runtime VFS through Chevalier gateway`

## What Is Proven

- Both repos were clean after committing the migration work.
- OtherYou compiles against the sibling `chevalier` checkout:
  - `rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path api/Cargo.toml`
- Chevalier Rust compiles:
  - `cargo check --manifest-path rust/Cargo.toml`
- Chevalier VMD bootstrap now tolerates already-existing mountpoint directories and fails hard when mountpoint creation or mount retries fail.
- OtherYou now consumes the generic Chevalier VFS gateway while keeping the legacy `/internal/nymfs/{nym_id}` alias as compatibility.
- Product-owned behavior remains in OtherYou: auth, ownership, task binding, write scopes, leases, mutation ledger, pack metadata, and GCS storage semantics.
- Local API was run from generated local tmux env, sourced from `infra/env/api.env.example.local`.
- Local validation used the staging GCS bucket from local env:
  - `chevalier-otheryou-vfs-vm-migration-20260605-b0f78571`
- Corvidae staging VMD nodes were started through:
  - `ops/computer/remote_vmd_box.sh up --api-internal-base-url http://100.73.105.110:3001 --restart-vmd`
- A real corvidae VM mounted:
  - `/nym`
  - `/nym/vm/mounts/shared`
  - `/nym/vm/mounts/skills`
  - `/nym/vm/mounts/task`
- The `/nym/vm/mounts/task` path was validated through the live VM:
  - write from inside VM
  - stat/read from inside VM
  - durable DB metadata observed under the conversation task mount
  - delete from inside VM
- Direct generic VFS/API validation covered:
  - lease acquire/release
  - mkdir/write/stat/read
  - range read
  - rename/delete
  - resource-key mismatch rejected before mutation
- Browser smoke reached the authenticated local app and selected the validation Nym. Artifacts are ignored under OtherYou `.run/`:
  - `.run/ui-nym-selected.png`
  - `.run/ui-nym-selected-body.txt`
- Local cleanup performed:
  - local API stopped
  - isolated Chrome validation profile stopped
  - corvidae VMD containers stopped
  - validation tasks for `MigrationVfs053050` were completed
  - no validation heartbeat schedules existed/enabled
- Final staging GCS cleanup was completed with the local staging service account in isolated `.run/gcloud` config:
  - deleted the two validation digest sidecar pack objects
  - deleted the matching local DB `nym_fs_entries`, `nym_fs_file_versions`, `nym_fs_file_manifests`, and `nym_fs_packs` rows
  - deleted one additional zero-reference validation pack object and DB pack row

## Final Cleanup Verification

Validation artifact DB count:

```bash
docker exec -i infra-postgres-1 psql -U nym -d nym -Atc \
  "select count(*) from nym_fs_entries where nym_id = 'a072b89c-fb33-447a-915c-5db54877853c' and logical_path like '%migration-validation-corvidae-active-task%';"
```

Result: `0`.

Validation pack parity:

```bash
CLOUDSDK_CONFIG=.run/gcloud gcloud storage ls \
  gs://chevalier-otheryou-vfs-vm-migration-20260605-b0f78571/nymfs/packs/a072b89c-fb33-447a-915c-5db54877853c/

docker exec -i infra-postgres-1 psql -U nym -d nym -Atc \
  "select pack_key from nym_fs_packs where nym_id = 'a072b89c-fb33-447a-915c-5db54877853c' order by pack_key;"
```

Result: `gcs=12 db=12`, diff empty.

## 2026-06-13 Performance/Correctness Audit Addendum

User concern: the migrated VFS/VMD boundary must not reinvent the optimized OtherYou VFS storage path. The old split had working performance characteristics around batch reads/writes, manifests, pack slots, zstd, and compressed pack range reads.

Finding:

- Chevalier owns only the generic VFS HTTP gateway, VMD/FUSE client behavior, VM lifecycle, and distributed control mechanics.
- OtherYou still owns product semantics and optimized storage: auth, path aliasing, task/conversation write scopes, leases, mutation ledger, GCS adapter behavior, manifests, pack rows, zstd pack building, `read_many_async`, `write_bulk_bytes_atomic_async`, range reads, and subtree prefetch.
- One regression was found and fixed: generic `/file/raw` originally called the backend's normal `stat`, but OtherYou's normal `stat` intentionally computes a content hash. The old `file/raw` route only used `stat_async` for kind/size and then read bytes. Because VMD FUSE already stats before reading, this accidentally added a hash/metadata cost to raw reads.

Fix:

- Chevalier `VfsGatewayBackend` now has a `stat_for_raw_read(...)` hook that defaults to `stat(...)`.
- Generic raw reads call `stat_for_raw_read(...)`.
- OtherYou overrides that hook with the old lightweight behavior: resolve path, `filesystem_core_service.stat_async`, no `hash_file_if_present_async`, then raw/range read.
- `tree` and `stat` still return content hashes because those are correctness/conflict surfaces.

Validation after the fix:

```bash
cargo test -p chevalier-sandbox --features vfs-server vfs:: -- --nocapture
rustup run 1.90-aarch64-apple-darwin cargo test --manifest-path api/Cargo.toml routes::runtime_nymfs_internal -- --nocapture
cargo test -p chevalier-sandbox distributed:: -- --nocapture
cargo test -p chevalier-sandbox --test facade_contract control_gateway_failover_prefers_healthy_secondary_endpoint -- --nocapture
cargo test -p chevalier-sandbox --test facade_contract continuity_rebinds_session_after_primary_vmd_loss -- --nocapture
cargo check -p chevalier-sandbox --features vfs-server
rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path api/Cargo.toml
rustup run 1.90-aarch64-apple-darwin cargo check -p vmd
```

Results:

- Chevalier VFS tests passed, including `gateway_raw_read_uses_lightweight_stat_hook`.
- OtherYou internal route tests passed.
- Distributed unit tests passed.
- Facade control-gateway failover and continuity rebind contract tests passed after rerunning outside the filesystem/network sandbox so test listeners could bind.
- Compile checks passed for OtherYou API, `chevalier-sandbox`, and `vmd`.

Live GCS/VMD validation evidence:

- Local API ran with `NYM__RUNTIME__NYMFS_BACKEND=gcs` against bucket `chevalier-otheryou-vfs-vm-migration-20260605-b0f78571`.
- Direct generic VFS route validation covered lease acquire/release, write/stat/full raw read/range raw read/rename/delete, legacy alias compatibility, and resource-key mismatch rejection.
- A real corvidae VM mounted `/nym`, `/nym/vm/mounts/shared`, `/nym/vm/mounts/skills`, and `/nym/vm/mounts/task`.
- Writing `live-vfs-from-vm-20260613` inside `/nym/vm/mounts/task/live-vfs.txt` produced durable OtherYou/GCS VFS metadata and digest sidecars.
- The same file was stat/read through the generic `/v1/internal/chevalier/vfs/{owner_id}` API.
- VM pause/resume was exercised through VMD; after resume, guest attach/read still returned `live-vfs-from-vm-20260613`.
- Validation artifacts were deleted through the generic VFS route under a product write lease, workspace tree returned `[]`, the disposable VM was deleted, and remote VMD validation nodes were stopped.

Known limit of this addendum:

- The focused facade/distributed gates were rerun on the current branch. The full real-machinery distributed failover script was not rerun on corvidae during this addendum because corvidae's existing `~/SoftwareProjects/chevalier` checkout is an older layout without the current `chevalier-sandbox` tree/toolchain staged. The migration code touched only the generic VFS raw-read stat hook and OtherYou adapter override after the earlier real-machinery validation, so the fast HA/distributed regression gates are the evidence for this addendum.

## Prod Deploy Commands After Morning Verification

No prod mutation has been performed for this migration.

API build, preferably using corvidae to avoid Cloud Build cost:

```bash
cd /Users/crow/SoftwareProjects/OtherYou
./ops/gke/nym-api/build-api-image-corvidae.sh us-west1-docker.pkg.dev/grounded-being-490907-r4/chevalier/nym-api:<tag>
kubectl -n nym-prod set image deployment/nym-api api=us-west1-docker.pkg.dev/grounded-being-490907-r4/chevalier/nym-api:<tag>
kubectl -n nym-prod rollout status deployment/nym-api --timeout=300s
```

OVH VMD upgrade if a new VMD image is built:

```bash
cd /Users/crow/SoftwareProjects/OtherYou
ops/ovh-vmd/upgrade-vmd.sh --image us-west1-docker.pkg.dev/grounded-being-490907-r4/chevalier/vmd@sha256:<digest>
```

Frontend deploy after API rollout is healthy:

```bash
cd /Users/crow/SoftwareProjects/OtherYou/desktop
pnpm build
firebase deploy --only hosting
```

Post-deploy smoke checks:

```bash
curl -fsS https://api.usenym.com/health
kubectl -n nym-prod get pods -o wide
kubectl -n nym-prod rollout status deployment/nym-api --timeout=300s
curl -sI https://app.usenym.com | grep -i cache-control
```

Rollback:

```bash
kubectl -n nym-prod rollout undo deployment/nym-api
kubectl -n nym-prod rollout status deployment/nym-api --timeout=300s
cd /Users/crow/SoftwareProjects/OtherYou/desktop
firebase hosting:releases:list
firebase hosting:rollback
```
