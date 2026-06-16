//! VFS storage bindings: `local` (filesystem) and `gateway` (HTTP) backends of
//! the engine's `OptimizedVfsStorage`. Returns metadata as JSON, bytes as
//! Buffer. (gcs / object-backed + manifest index are a follow-up.)

use std::path::PathBuf;
use std::sync::Arc;

use bytes::Bytes;
use chevalier_vfs::gateway::{GatewayVfsStorage, GatewayVfsStorageConfig};
use chevalier_vfs::local::LocalVfsStorage;
use chevalier_vfs::{
    OptimizedVfsStorage, VfsStorageEntryKind, VfsStorageError, VfsStorageMetadata,
    VfsStorageObjectState,
};
use napi::bindgen_prelude::{BigInt, Buffer};
use napi_derive::napi;

fn vfs_err(e: VfsStorageError) -> napi::Error {
    napi::Error::new(napi::Status::GenericFailure, format!("VFS: {e}"))
}

fn to_json<T: serde::Serialize>(v: T) -> napi::Result<serde_json::Value> {
    serde_json::to_value(v)
        .map_err(|e| napi::Error::new(napi::Status::GenericFailure, format!("serialize: {e}")))
}

/// Pack-slot location for an object-backed file.
#[napi(object)]
pub struct VfsObjectState {
    /// `bigint` — exact size, lossless above 2^53.
    pub size_bytes: BigInt,
    pub pack_key: String,
    pub pack_slot_offset: BigInt,
    pub pack_slot_length: BigInt,
    pub pack_slot_compression: i32,
}

impl From<VfsStorageObjectState> for VfsObjectState {
    fn from(o: VfsStorageObjectState) -> Self {
        Self {
            size_bytes: BigInt::from(o.size_bytes),
            pack_key: o.pack_key,
            pack_slot_offset: BigInt::from(o.pack_slot_offset),
            pack_slot_length: BigInt::from(o.pack_slot_length),
            pack_slot_compression: o.pack_slot_compression as i32,
        }
    }
}

/// File/dir metadata. `sizeBytes` is a `bigint` so it never loses precision at
/// the FFI boundary (JS `number` only holds integers up to 2^53).
#[napi(object)]
pub struct VfsMetadata {
    pub path: String,
    /// `"File"` or `"Directory"`.
    pub kind: String,
    pub size_bytes: BigInt,
    pub content_hash: Option<String>,
    pub token_count: Option<i32>,
    pub version: Option<String>,
    /// RFC 3339 timestamp.
    pub updated_at: Option<String>,
    pub object_state: Option<VfsObjectState>,
}

impl From<VfsStorageMetadata> for VfsMetadata {
    fn from(m: VfsStorageMetadata) -> Self {
        Self {
            path: m.path,
            kind: match m.kind {
                VfsStorageEntryKind::File => "File",
                VfsStorageEntryKind::Directory => "Directory",
            }
            .to_string(),
            size_bytes: BigInt::from(m.size_bytes),
            content_hash: m.content_hash,
            token_count: m.token_count,
            version: m.version,
            updated_at: m.updated_at.map(|d| d.to_rfc3339()),
            object_state: m.object_state.map(VfsObjectState::from),
        }
    }
}

/// Options for the HTTP gateway backend.
#[napi(object)]
pub struct GatewayOptions {
    pub endpoint: String,
    pub auth_token: Option<String>,
    pub scope_path: Option<String>,
    pub component: Option<String>,
    pub mutation_reason: Option<String>,
}

/// A virtual filesystem. Construct via `VfsStorage.local(root)` or
/// `VfsStorage.gateway(opts)`.
#[napi]
pub struct VfsStorage {
    inner: Arc<dyn OptimizedVfsStorage>,
}

#[napi]
impl VfsStorage {
    /// Filesystem-backed storage rooted at `root`.
    #[napi(factory)]
    pub fn local(root: String) -> VfsStorage {
        VfsStorage {
            inner: Arc::new(LocalVfsStorage::new(PathBuf::from(root))),
        }
    }

    /// HTTP gateway-backed storage.
    #[napi(factory)]
    pub fn gateway(options: GatewayOptions) -> VfsStorage {
        let mut cfg = GatewayVfsStorageConfig::new(options.endpoint);
        if let Some(t) = options.auth_token {
            cfg = cfg.with_auth_token(t);
        }
        if let Some(s) = options.scope_path {
            cfg = cfg.with_scope_path(s);
        }
        if let Some(c) = options.component {
            cfg = cfg.with_component(c);
        }
        if let Some(r) = options.mutation_reason {
            cfg = cfg.with_mutation_reason(r);
        }
        VfsStorage {
            inner: Arc::new(GatewayVfsStorage::new(cfg)),
        }
    }

    /// Read a file's bytes.
    #[napi]
    pub async fn read(&self, path: String) -> napi::Result<Buffer> {
        let b = self.inner.read(&path).await.map_err(vfs_err)?;
        Ok(Buffer::from(b.to_vec()))
    }

    /// Write a file; returns the write result (JSON: content hash, changed, …).
    #[napi]
    pub async fn write(&self, path: String, data: Buffer) -> napi::Result<serde_json::Value> {
        let r = self
            .inner
            .write(&path, Bytes::from(data.to_vec()), None)
            .await
            .map_err(vfs_err)?;
        to_json(r)
    }

    /// Stat a path; returns typed metadata (`sizeBytes` is a `bigint`) or null.
    #[napi]
    pub async fn stat(&self, path: String) -> napi::Result<Option<VfsMetadata>> {
        Ok(self
            .inner
            .stat(&path)
            .await
            .map_err(vfs_err)?
            .map(VfsMetadata::from))
    }

    /// List a directory's entries with typed metadata.
    #[napi]
    pub async fn list_dir(&self, path: String) -> napi::Result<Vec<VfsMetadata>> {
        let items = self
            .inner
            .list_dir_with_metadata(&path, Default::default())
            .await
            .map_err(vfs_err)?;
        Ok(items.into_iter().map(VfsMetadata::from).collect())
    }

    /// Create a directory.
    #[napi]
    pub async fn mkdir(&self, path: String) -> napi::Result<()> {
        self.inner.mkdir(&path).await.map_err(vfs_err)
    }

    /// Delete a file; returns the delete result (JSON).
    #[napi]
    pub async fn remove(&self, path: String) -> napi::Result<serde_json::Value> {
        let r = self
            .inner
            .delete_file_with_metadata(&path, None)
            .await
            .map_err(vfs_err)?;
        to_json(r)
    }

    /// Remove an (empty) directory.
    #[napi]
    pub async fn rmdir(&self, path: String) -> napi::Result<()> {
        self.inner.rmdir(&path).await.map_err(vfs_err)
    }

    /// Rename/move a file; returns the rename result (JSON).
    #[napi]
    pub async fn rename(&self, from: String, to: String) -> napi::Result<serde_json::Value> {
        let r = self
            .inner
            .rename_with_metadata(&from, &to)
            .await
            .map_err(vfs_err)?;
        to_json(r)
    }
}
