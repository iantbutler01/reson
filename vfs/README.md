# chevalier-vfs

`chevalier-vfs` is the optimized VFS storage layer shared by local/direct consumers and remote gateway consumers.

It owns generic storage mechanics:

- packed file layout and zstd slot extraction
- object-store backed reads and writes
- range reads and batch reads
- metadata and subtree lookups
- precondition-aware atomic writes
- changed-only writes
- subtree prefetch and small-file warming
- pack coalescing and zero-reference cleanup helpers
- a default Postgres manifest/index implementation

Product policy stays outside this crate. Ownership, authorization, leases, audit rows, task rows, conversation rows, and product-specific path projection should wrap this crate rather than being embedded in it.

## Storage Shape

Packed storage has two parts:

- an `ObjectStoreClient` for pack bytes
- a `VfsManifestIndex` for logical path heads and pack-slot coordinates

The object store alone is not enough for durable packed storage. A pack object can hold many slots, but the manifest/index tells the VFS which slot is the current `/some/path.md` version.

## Default Postgres Index

The default serious index is `PostgresVfsManifestIndex`, behind the `postgres` feature.

Apply the schema in:

```text
migrations/postgres/001_chevalier_vfs_index.sql
```

That creates:

- `chevalier_vfs_entries`
- `chevalier_vfs_packs`
- `chevalier_vfs_file_manifests`
- `chevalier_vfs_file_versions`
- `chevalier_vfs_mutations`

The schema is generic. Product tables should reference or wrap it rather than adding product ownership directly to these tables.

## Local Object Store With Postgres

For local development, use Postgres for the manifest/index and a filesystem directory as the object store:

```rust
use std::{path::PathBuf, sync::Arc};

use bytes::Bytes;
use chevalier_vfs::{
    index::VfsIndexScope,
    object_storage::{ObjectBackedVfsStorage, ObjectBackedVfsStorageConfig},
    object_store::LocalObjectStoreClient,
    postgres_index::PostgresVfsManifestIndex,
    OptimizedVfsStorage,
};
use sqlx::PgPool;

# async fn build() -> Result<(), Box<dyn std::error::Error>> {
let pool = PgPool::connect("postgres://user:pass@localhost/chevalier").await?;
let index = Arc::new(PostgresVfsManifestIndex::new(pool));
let store = Arc::new(LocalObjectStoreClient::new(PathBuf::from(".run/chevalier-vfs-objects"))?);

let vfs = ObjectBackedVfsStorage::new(
    ObjectBackedVfsStorageConfig::new(VfsIndexScope::new("dev-scope")),
    store,
    index,
);

vfs.write("notes/hello.txt", Bytes::from_static(b"hello"), None).await?;
let bytes = vfs.read("notes/hello.txt").await?;
assert_eq!(&bytes[..], b"hello");
# Ok(())
# }
```

There is also a compile-checked example:

```sh
cargo check --manifest-path vfs/Cargo.toml \
  --example postgres_local_object_store \
  --features postgres,tokio
```

At runtime it expects:

```sh
export CHEVALIER_VFS_DATABASE_URL='postgres://user:pass@localhost/chevalier'
export CHEVALIER_VFS_OBJECT_ROOT='.run/chevalier-vfs-objects'
export CHEVALIER_VFS_SCOPE='dev-scope'
```

## GCS Object Store

Enable the `gcs` feature and use `GcsObjectStoreClient` for object bytes. The manifest/index can still be Postgres, a product bridge, or a gateway-backed implementation.

```rust
use chevalier_vfs::gcs_object_store::{GcsObjectStoreClient, GcsObjectStoreConfig};

let store = GcsObjectStoreClient::new(GcsObjectStoreConfig {
    bucket: "my-vfs-bucket".to_string(),
    service_account_key_path: Some("service-account.json".into()),
    api_base_url: "https://storage.googleapis.com/storage/v1".to_string(),
    upload_base_url: "https://storage.googleapis.com/upload/storage/v1".to_string(),
    token_uri: "https://oauth2.googleapis.com/token".to_string(),
    timeout_seconds: 30,
})?;
```

## Gateway Mode

Enable the `gateway` feature and use `GatewayVfsStorage` when storage is served through a Chevalier VFS HTTP gateway. It implements the same `OptimizedVfsStorage` trait, so callers can switch between local/direct and remote/gateway mode without changing call sites.

Gateway mode preserves the optimized operations that matter for product performance:

- `metadata_many`
- `read_many`
- `list_subtree_file_metadata`
- `prefetch_subtree`
- `write_many_atomic`
- changed-only write planning
- metadata-returning delete and rename

## Compatibility Notes

Loose-object or no-index object-store paths are compatibility-only. They can be useful during migration, but they are not a complete durable packed VFS backend because they cannot rediscover logical file heads or pack-slot coordinates from object bytes alone.
