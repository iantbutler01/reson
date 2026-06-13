# OtherYou VFS/VMD Migration Validation Note

Date: 2026-06-13

Branches:

- `reson`: `wip/reson-vfs-vmd-migration`
- `OtherYou`: `ian/vmd-vfs-reson-migration`

Current commits at initial validation:

- `reson`: `812bbb6 Harden VMD mount bootstrap`
- `OtherYou`: `cbfb07f Wire OtherYou runtime VFS through Reson gateway`

## What Is Proven

- Both repos were clean after committing the migration work.
- OtherYou compiles against the sibling `reson` checkout:
  - `rustup run 1.90-aarch64-apple-darwin cargo check --manifest-path api/Cargo.toml`
- Reson Rust compiles:
  - `cargo check --manifest-path reson-rust/Cargo.toml`
- Reson VMD bootstrap now tolerates already-existing mountpoint directories and fails hard when mountpoint creation or mount retries fail.
- OtherYou now consumes the generic Reson VFS gateway while keeping the legacy `/internal/nymfs/{nym_id}` alias as compatibility.
- Product-owned behavior remains in OtherYou: auth, ownership, task binding, write scopes, leases, mutation ledger, pack metadata, and GCS storage semantics.
- Local API was run from generated local tmux env, sourced from `infra/env/api.env.example.local`.
- Local validation used the staging GCS bucket from local env:
  - `reson-otheryou-vfs-vm-migration-20260605-b0f78571`
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
  gs://reson-otheryou-vfs-vm-migration-20260605-b0f78571/nymfs/packs/a072b89c-fb33-447a-915c-5db54877853c/

docker exec -i infra-postgres-1 psql -U nym -d nym -Atc \
  "select pack_key from nym_fs_packs where nym_id = 'a072b89c-fb33-447a-915c-5db54877853c' order by pack_key;"
```

Result: `gcs=12 db=12`, diff empty.

## Prod Deploy Commands After Morning Verification

No prod mutation has been performed for this migration.

API build, preferably using corvidae to avoid Cloud Build cost:

```bash
cd /Users/crow/SoftwareProjects/OtherYou
./ops/gke/nym-api/build-api-image-corvidae.sh us-west1-docker.pkg.dev/grounded-being-490907-r4/reson/nym-api:<tag>
kubectl -n nym-prod set image deployment/nym-api api=us-west1-docker.pkg.dev/grounded-being-490907-r4/reson/nym-api:<tag>
kubectl -n nym-prod rollout status deployment/nym-api --timeout=300s
```

OVH VMD upgrade if a new VMD image is built:

```bash
cd /Users/crow/SoftwareProjects/OtherYou
ops/ovh-vmd/upgrade-vmd.sh --image us-west1-docker.pkg.dev/grounded-being-490907-r4/reson/vmd@sha256:<digest>
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
