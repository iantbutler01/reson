// @dive-file: Optimized VFS storage primitives shared by local and gateway-backed consumers.
// @dive-rel: Owns generic storage semantics such as logical metadata, batch reads/writes,
// @dive-rel: preconditions, subtree prefetch, and pack-format helpers without product policy.
// @dive-rel: Complements vfs.rs, which remains the HTTP/FUSE gateway protocol boundary.

use bytes::Bytes;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub mod compaction;
#[cfg(feature = "gateway")]
pub mod gateway;
#[cfg(feature = "gcs")]
pub mod gcs_object_store;
pub mod index;
pub mod local;
pub mod manifest;
pub mod object_storage;
pub mod object_store;
pub mod pack;
pub mod pack_cache;
#[cfg(feature = "postgres")]
pub mod postgres_index;

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum VfsStorageError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("forbidden: {0}")]
    Forbidden(String),
    #[error("conflict: {0}")]
    Conflict(String),
    #[error("internal error: {0}")]
    Internal(String),
}

pub type VfsStorageResult<T> = std::result::Result<T, VfsStorageError>;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum VfsStorageEntryKind {
    File,
    Directory,
}

impl VfsStorageEntryKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::File => "file",
            Self::Directory => "directory",
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageObjectState {
    pub size_bytes: u64,
    pub pack_key: String,
    pub pack_slot_offset: i64,
    pub pack_slot_length: i64,
    pub pack_slot_compression: i16,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageMetadata {
    pub path: String,
    pub kind: VfsStorageEntryKind,
    pub size_bytes: u64,
    pub content_hash: Option<String>,
    pub token_count: Option<i32>,
    pub version: Option<String>,
    pub updated_at: Option<DateTime<Utc>>,
    pub object_state: Option<VfsStorageObjectState>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageWritePrecondition {
    pub fingerprint: Option<String>,
    pub secondary_fingerprint: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageMetadataFields {
    pub include_object_state: bool,
    pub include_token_count: bool,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageDirListFilter {
    pub name_like: Option<String>,
    pub name_not_like: Option<String>,
    pub entry_kind: Option<VfsStorageEntryKind>,
    pub limit: Option<i64>,
    pub order: Option<VfsStorageDirListOrder>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum VfsStorageDirListOrder {
    KindThenName,
    NameAsc,
    NameDesc,
    UpdatedDesc,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageSubtreeOptions {
    pub include_object_state: bool,
    pub include_token_count: bool,
    pub limit: Option<i64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageReadRange {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageWrite {
    pub path: String,
    pub bytes: Bytes,
    pub token_count: Option<i32>,
    pub precondition: Option<VfsStorageWritePrecondition>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageWriteResult {
    pub path: String,
    pub content_hash: String,
    pub previous_hash: Option<String>,
    pub changed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageReadIfChanged {
    pub path: String,
    pub known_content_hash: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageReadIfChangedResult {
    pub path: String,
    pub content_hash: Option<String>,
    pub bytes: Option<Bytes>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageDeleteResult {
    pub previous: Option<VfsStorageMetadata>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStorageRenameResult {
    pub previous: Option<VfsStorageMetadata>,
    pub current: Option<VfsStorageMetadata>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStoragePrefetchOptions {
    pub include_small_file_bytes: bool,
    pub max_entries: Option<i64>,
    pub max_pack_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsStoragePrefetchResult {
    pub warmed_file_bytes: Vec<(String, Bytes)>,
}

#[async_trait::async_trait]
pub trait OptimizedVfsStorage: Send + Sync {
    fn backend_name(&self) -> &'static str;

    async fn stat(&self, path: &str) -> VfsStorageResult<Option<VfsStorageMetadata>>;

    async fn metadata_many(
        &self,
        paths: &[String],
        fields: VfsStorageMetadataFields,
    ) -> VfsStorageResult<Vec<Option<VfsStorageMetadata>>>;

    async fn list_dir_with_metadata(
        &self,
        path: &str,
        filter: VfsStorageDirListFilter,
    ) -> VfsStorageResult<Vec<VfsStorageMetadata>>;

    async fn list_subtree_file_metadata(
        &self,
        prefix: &str,
        options: VfsStorageSubtreeOptions,
    ) -> VfsStorageResult<Vec<VfsStorageMetadata>>;

    async fn read(&self, path: &str) -> VfsStorageResult<Bytes>;

    async fn read_range(&self, path: &str, range: VfsStorageReadRange) -> VfsStorageResult<Bytes>;

    async fn read_many(&self, paths: &[String]) -> VfsStorageResult<Vec<(String, Bytes)>>;

    async fn read_many_if_etag_mismatch(
        &self,
        requests: &[VfsStorageReadIfChanged],
    ) -> VfsStorageResult<Vec<VfsStorageReadIfChangedResult>>;

    async fn write(
        &self,
        path: &str,
        bytes: Bytes,
        precondition: Option<VfsStorageWritePrecondition>,
    ) -> VfsStorageResult<VfsStorageWriteResult>;

    async fn write_many_atomic(
        &self,
        writes: Vec<VfsStorageWrite>,
    ) -> VfsStorageResult<Vec<VfsStorageWriteResult>>;

    async fn write_many_if_changed_atomic(
        &self,
        writes: Vec<VfsStorageWrite>,
    ) -> VfsStorageResult<Vec<VfsStorageWriteResult>>;

    async fn mkdir(&self, path: &str) -> VfsStorageResult<()>;

    async fn delete_file_with_metadata(
        &self,
        path: &str,
        precondition: Option<VfsStorageWritePrecondition>,
    ) -> VfsStorageResult<VfsStorageDeleteResult>;

    async fn rmdir(&self, path: &str) -> VfsStorageResult<()>;

    async fn rename_with_metadata(
        &self,
        from: &str,
        to: &str,
    ) -> VfsStorageResult<VfsStorageRenameResult>;

    async fn prefetch_subtree(
        &self,
        prefix: &str,
        options: VfsStoragePrefetchOptions,
    ) -> VfsStorageResult<VfsStoragePrefetchResult>;
}
