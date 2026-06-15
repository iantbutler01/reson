// @dive-file: Local filesystem implementation of the optimized VFS storage trait.
// @dive-rel: Provides the direct/dev backend for chevalier-vfs without product policy or VM concerns.
// @dive-rel: Mirrors the old local nymfs adapter semantics while exposing batch-oriented calls.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use bytes::Bytes;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::{
    OptimizedVfsStorage, VfsStorageDeleteResult, VfsStorageDirListFilter, VfsStorageDirListOrder,
    VfsStorageEntryKind, VfsStorageError, VfsStorageMetadata, VfsStorageMetadataFields,
    VfsStorageObjectState, VfsStoragePrefetchOptions, VfsStoragePrefetchResult,
    VfsStorageReadIfChanged, VfsStorageReadIfChangedResult, VfsStorageReadRange,
    VfsStorageRenameResult, VfsStorageResult, VfsStorageSubtreeOptions, VfsStorageWrite,
    VfsStorageWritePrecondition, VfsStorageWriteResult,
    pack::{SlotCompression, hex_hash},
};

#[derive(Clone, Debug)]
pub struct LocalVfsStorage {
    root: PathBuf,
}

impl LocalVfsStorage {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn abs_path(&self, logical_path: &str) -> VfsStorageResult<PathBuf> {
        let logical_path = logical_path.trim_matches('/');
        let mut out = self.root.clone();
        if logical_path.is_empty() {
            return Ok(out);
        }
        for component in Path::new(logical_path).components() {
            match component {
                Component::Normal(part) => out.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(VfsStorageError::BadRequest(format!(
                        "invalid vfs path: {logical_path}"
                    )));
                }
            }
        }
        Ok(out)
    }

    fn logical_path_for(&self, abs_path: &Path) -> VfsStorageResult<String> {
        let rel = abs_path.strip_prefix(&self.root).map_err(|err| {
            VfsStorageError::Internal(format!("local path escaped vfs root: {err}"))
        })?;
        Ok(rel
            .components()
            .filter_map(|component| match component {
                Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("/"))
    }

    fn metadata_for_abs(&self, abs_path: &Path) -> VfsStorageResult<Option<VfsStorageMetadata>> {
        let metadata = match fs::metadata(abs_path) {
            Ok(metadata) => metadata,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => return Err(VfsStorageError::Internal(err.to_string())),
        };
        let kind = if metadata.is_dir() {
            VfsStorageEntryKind::Directory
        } else {
            VfsStorageEntryKind::File
        };
        let content_hash = if metadata.is_file() {
            hash_file_if_present(abs_path)?
        } else {
            None
        };
        let object_state = metadata.is_file().then(|| VfsStorageObjectState {
            size_bytes: metadata.len(),
            pack_key: format!("local://{}", abs_path.display()),
            pack_slot_offset: 0,
            pack_slot_length: metadata.len() as i64,
            pack_slot_compression: SlotCompression::Raw.as_db_smallint(),
        });
        Ok(Some(VfsStorageMetadata {
            path: self.logical_path_for(abs_path)?,
            kind,
            size_bytes: metadata.len(),
            content_hash,
            token_count: None,
            version: None,
            updated_at: modified_at(&metadata),
            object_state,
        }))
    }

    fn metadata_for_path(&self, path: &str) -> VfsStorageResult<Option<VfsStorageMetadata>> {
        let abs_path = self.abs_path(path)?;
        self.metadata_for_abs(&abs_path)
    }

    fn write_precondition(&self, path: &str) -> VfsStorageResult<VfsStorageWritePrecondition> {
        let abs_path = self.abs_path(path)?;
        Ok(VfsStorageWritePrecondition {
            fingerprint: Some(
                hash_file_if_present(&abs_path)?.unwrap_or_else(|| "absent".to_string()),
            ),
            secondary_fingerprint: None,
        })
    }

    fn assert_precondition(
        &self,
        path: &str,
        precondition: Option<&VfsStorageWritePrecondition>,
    ) -> VfsStorageResult<()> {
        let Some(precondition) = precondition else {
            return Ok(());
        };
        let expected = precondition.fingerprint.as_deref().unwrap_or("absent");
        let actual = self
            .write_precondition(path)?
            .fingerprint
            .unwrap_or_else(|| "absent".to_string());
        if actual == expected {
            Ok(())
        } else {
            Err(VfsStorageError::Conflict(format!(
                "local vfs write precondition failed for {path}"
            )))
        }
    }
}

#[async_trait::async_trait]
impl OptimizedVfsStorage for LocalVfsStorage {
    fn backend_name(&self) -> &'static str {
        "local"
    }

    async fn stat(&self, path: &str) -> VfsStorageResult<Option<VfsStorageMetadata>> {
        self.metadata_for_path(path)
    }

    async fn metadata_many(
        &self,
        paths: &[String],
        _fields: VfsStorageMetadataFields,
    ) -> VfsStorageResult<Vec<Option<VfsStorageMetadata>>> {
        paths
            .iter()
            .map(|path| self.metadata_for_path(path))
            .collect()
    }

    async fn list_dir_with_metadata(
        &self,
        path: &str,
        filter: VfsStorageDirListFilter,
    ) -> VfsStorageResult<Vec<VfsStorageMetadata>> {
        let abs_path = self.abs_path(path)?;
        let read_dir = fs::read_dir(&abs_path).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => VfsStorageError::NotFound(path.to_string()),
            _ => VfsStorageError::Internal(err.to_string()),
        })?;
        let mut entries = Vec::new();
        for entry in read_dir {
            let entry = entry.map_err(|err| VfsStorageError::Internal(err.to_string()))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            if !filter_name(&name, &filter) {
                continue;
            }
            let Some(metadata) = self.metadata_for_abs(&entry.path())? else {
                continue;
            };
            if let Some(kind) = filter.entry_kind
                && metadata.kind != kind
            {
                continue;
            }
            entries.push(metadata);
        }
        sort_entries(&mut entries, filter.order);
        if let Some(limit) = filter.limit {
            entries.truncate(limit.max(0) as usize);
        }
        Ok(entries)
    }

    async fn list_subtree_file_metadata(
        &self,
        prefix: &str,
        options: VfsStorageSubtreeOptions,
    ) -> VfsStorageResult<Vec<VfsStorageMetadata>> {
        let root = self.abs_path(prefix)?;
        if !root.exists() {
            return Ok(Vec::new());
        }
        let mut stack = vec![root];
        let mut out = Vec::new();
        while let Some(path) = stack.pop() {
            let metadata = match fs::metadata(&path) {
                Ok(metadata) => metadata,
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
                Err(err) => return Err(VfsStorageError::Internal(err.to_string())),
            };
            if metadata.is_dir() {
                for entry in
                    fs::read_dir(&path).map_err(|err| VfsStorageError::Internal(err.to_string()))?
                {
                    stack.push(
                        entry
                            .map_err(|err| VfsStorageError::Internal(err.to_string()))?
                            .path(),
                    );
                }
                continue;
            }
            if metadata.is_file()
                && let Some(file_metadata) = self.metadata_for_abs(&path)?
            {
                out.push(file_metadata);
            }
            if let Some(limit) = options.limit
                && out.len() >= limit.max(0) as usize
            {
                break;
            }
        }
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    }

    async fn read(&self, path: &str) -> VfsStorageResult<Bytes> {
        read_file(&self.abs_path(path)?).map(Bytes::from)
    }

    async fn read_range(&self, path: &str, range: VfsStorageReadRange) -> VfsStorageResult<Bytes> {
        let mut file = fs::File::open(self.abs_path(path)?).map_err(|err| match err.kind() {
            std::io::ErrorKind::NotFound => VfsStorageError::NotFound(path.to_string()),
            _ => VfsStorageError::Internal(err.to_string()),
        })?;
        file.seek(SeekFrom::Start(range.offset))
            .map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        let mut bytes = Vec::with_capacity(range.length as usize);
        file.take(range.length)
            .read_to_end(&mut bytes)
            .map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        Ok(Bytes::from(bytes))
    }

    async fn read_many(&self, paths: &[String]) -> VfsStorageResult<Vec<(String, Bytes)>> {
        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            match self.read(path).await {
                Ok(bytes) => out.push((path.clone(), bytes)),
                Err(VfsStorageError::NotFound(_)) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(out)
    }

    async fn read_many_if_etag_mismatch(
        &self,
        requests: &[VfsStorageReadIfChanged],
    ) -> VfsStorageResult<Vec<VfsStorageReadIfChangedResult>> {
        let mut out = Vec::with_capacity(requests.len());
        for request in requests {
            let metadata = self.metadata_for_path(&request.path)?;
            let Some(metadata) = metadata else {
                out.push(VfsStorageReadIfChangedResult {
                    path: request.path.clone(),
                    content_hash: None,
                    bytes: None,
                });
                continue;
            };
            let hash = metadata.content_hash.clone();
            if hash == request.known_content_hash {
                out.push(VfsStorageReadIfChangedResult {
                    path: request.path.clone(),
                    content_hash: hash,
                    bytes: None,
                });
                continue;
            }
            out.push(VfsStorageReadIfChangedResult {
                path: request.path.clone(),
                content_hash: hash,
                bytes: Some(self.read(&request.path).await?),
            });
        }
        Ok(out)
    }

    async fn write(
        &self,
        path: &str,
        bytes: Bytes,
        precondition: Option<VfsStorageWritePrecondition>,
    ) -> VfsStorageResult<VfsStorageWriteResult> {
        let result = self
            .write_many_atomic(vec![VfsStorageWrite {
                path: path.to_string(),
                bytes,
                token_count: None,
                precondition,
            }])
            .await?;
        result
            .into_iter()
            .next()
            .ok_or_else(|| VfsStorageError::Internal("write returned no result".to_string()))
    }

    async fn write_many_atomic(
        &self,
        writes: Vec<VfsStorageWrite>,
    ) -> VfsStorageResult<Vec<VfsStorageWriteResult>> {
        for write in &writes {
            self.assert_precondition(&write.path, write.precondition.as_ref())?;
        }
        install_writes(self, writes)
    }

    async fn write_many_if_changed_atomic(
        &self,
        writes: Vec<VfsStorageWrite>,
    ) -> VfsStorageResult<Vec<VfsStorageWriteResult>> {
        for write in &writes {
            self.assert_precondition(&write.path, write.precondition.as_ref())?;
        }
        let mut changed = Vec::new();
        let mut unchanged = Vec::new();
        for write in writes {
            let previous_hash = self
                .metadata_for_path(&write.path)?
                .and_then(|metadata| metadata.content_hash);
            let next_hash = hex_hash(&write.bytes);
            if previous_hash.as_deref() == Some(next_hash.as_str()) {
                unchanged.push(VfsStorageWriteResult {
                    path: write.path,
                    content_hash: next_hash,
                    previous_hash,
                    changed: false,
                });
            } else {
                changed.push(write);
            }
        }
        let mut out = install_writes(self, changed)?;
        out.extend(unchanged);
        out.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(out)
    }

    async fn mkdir(&self, path: &str) -> VfsStorageResult<()> {
        fs::create_dir_all(self.abs_path(path)?)
            .map_err(|err| VfsStorageError::Internal(err.to_string()))
    }

    async fn delete_file_with_metadata(
        &self,
        path: &str,
        precondition: Option<VfsStorageWritePrecondition>,
    ) -> VfsStorageResult<VfsStorageDeleteResult> {
        self.assert_precondition(path, precondition.as_ref())?;
        let previous = self.metadata_for_path(path)?;
        if matches!(
            previous.as_ref().map(|metadata| metadata.kind),
            Some(VfsStorageEntryKind::Directory)
        ) {
            return Err(VfsStorageError::BadRequest(format!(
                "vfs path {path} is not a file"
            )));
        }
        match fs::remove_file(self.abs_path(path)?) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(VfsStorageError::Internal(err.to_string())),
        }
        Ok(VfsStorageDeleteResult { previous })
    }

    async fn rmdir(&self, path: &str) -> VfsStorageResult<()> {
        let Some(metadata) = self.metadata_for_path(path)? else {
            return Ok(());
        };
        if metadata.kind != VfsStorageEntryKind::Directory {
            return Err(VfsStorageError::BadRequest(format!(
                "vfs path {path} is not a directory"
            )));
        }
        match fs::remove_dir(self.abs_path(path)?) {
            Ok(()) => Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::DirectoryNotEmpty => Err(
                VfsStorageError::Conflict(format!("vfs directory {path} is not empty")),
            ),
            Err(err) => Err(VfsStorageError::Internal(err.to_string())),
        }
    }

    async fn rename_with_metadata(
        &self,
        from: &str,
        to: &str,
    ) -> VfsStorageResult<VfsStorageRenameResult> {
        let previous = self.metadata_for_path(from)?;
        let Some(_) = previous else {
            return Err(VfsStorageError::NotFound(from.to_string()));
        };
        let to_abs = self.abs_path(to)?;
        if let Some(parent) = to_abs.parent() {
            fs::create_dir_all(parent).map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        }
        fs::rename(self.abs_path(from)?, &to_abs)
            .map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        let current = self.metadata_for_abs(&to_abs)?;
        Ok(VfsStorageRenameResult { previous, current })
    }

    async fn prefetch_subtree(
        &self,
        _prefix: &str,
        _options: VfsStoragePrefetchOptions,
    ) -> VfsStorageResult<VfsStoragePrefetchResult> {
        Ok(VfsStoragePrefetchResult::default())
    }
}

fn install_writes(
    storage: &LocalVfsStorage,
    writes: Vec<VfsStorageWrite>,
) -> VfsStorageResult<Vec<VfsStorageWriteResult>> {
    let mut seen = HashSet::new();
    let mut staged = Vec::with_capacity(writes.len());
    for write in writes {
        if !seen.insert(write.path.clone()) {
            return Err(VfsStorageError::BadRequest(format!(
                "duplicate vfs write path: {}",
                write.path
            )));
        }
        let abs_path = storage.abs_path(&write.path)?;
        let previous_hash = storage
            .metadata_for_abs(&abs_path)?
            .and_then(|metadata| metadata.content_hash);
        let content_hash = hex_hash(&write.bytes);
        if let Some(parent) = abs_path.parent() {
            fs::create_dir_all(parent).map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        }
        let tmp_path = abs_path.with_file_name(format!(
            ".{}.{}.tmp",
            abs_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("vfs"),
            Uuid::new_v4()
        ));
        fs::write(&tmp_path, &write.bytes)
            .map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        staged.push((write.path, abs_path, tmp_path, content_hash, previous_hash));
    }

    let mut results = Vec::with_capacity(staged.len());
    for (path, abs_path, tmp_path, content_hash, previous_hash) in staged {
        fs::rename(&tmp_path, &abs_path)
            .map_err(|err| VfsStorageError::Internal(err.to_string()))?;
        results.push(VfsStorageWriteResult {
            path,
            content_hash,
            previous_hash,
            changed: true,
        });
    }
    Ok(results)
}

fn read_file(path: &Path) -> VfsStorageResult<Vec<u8>> {
    match fs::read(path) {
        Ok(bytes) => Ok(bytes),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Err(VfsStorageError::NotFound(path.display().to_string()))
        }
        Err(err) => Err(VfsStorageError::Internal(err.to_string())),
    }
}

fn hash_file_if_present(path: &Path) -> VfsStorageResult<Option<String>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(hex_hash(&bytes))),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(err) => Err(VfsStorageError::Internal(err.to_string())),
    }
}

fn modified_at(metadata: &fs::Metadata) -> Option<DateTime<Utc>> {
    metadata.modified().ok().map(DateTime::<Utc>::from)
}

fn filter_name(name: &str, filter: &VfsStorageDirListFilter) -> bool {
    if let Some(pattern) = filter.name_like.as_deref()
        && !sql_like_match(pattern, name)
    {
        return false;
    }
    if let Some(pattern) = filter.name_not_like.as_deref()
        && sql_like_match(pattern, name)
    {
        return false;
    }
    true
}

fn sort_entries(entries: &mut [VfsStorageMetadata], order: Option<VfsStorageDirListOrder>) {
    match order.unwrap_or(VfsStorageDirListOrder::KindThenName) {
        VfsStorageDirListOrder::KindThenName => {
            entries.sort_by(|a, b| kind_order(a.kind, b.kind).then_with(|| a.path.cmp(&b.path)))
        }
        VfsStorageDirListOrder::NameAsc => entries.sort_by(|a, b| a.path.cmp(&b.path)),
        VfsStorageDirListOrder::NameDesc => entries.sort_by(|a, b| b.path.cmp(&a.path)),
        VfsStorageDirListOrder::UpdatedDesc => entries.sort_by(|a, b| {
            b.updated_at
                .cmp(&a.updated_at)
                .then_with(|| a.path.cmp(&b.path))
        }),
    }
}

fn kind_order(a: VfsStorageEntryKind, b: VfsStorageEntryKind) -> Ordering {
    match (a, b) {
        (VfsStorageEntryKind::Directory, VfsStorageEntryKind::File) => Ordering::Less,
        (VfsStorageEntryKind::File, VfsStorageEntryKind::Directory) => Ordering::Greater,
        _ => Ordering::Equal,
    }
}

fn sql_like_match(pattern: &str, name: &str) -> bool {
    fn match_inner(pat: &[u8], s: &[u8]) -> bool {
        let mut pi = 0;
        let mut si = 0;
        let mut star = None;
        let mut star_si = 0;
        while si < s.len() {
            if pi < pat.len() {
                match pat[pi] {
                    b'%' => {
                        star = Some(pi);
                        star_si = si;
                        pi += 1;
                        continue;
                    }
                    b'_' => {
                        pi += 1;
                        si += 1;
                        continue;
                    }
                    c if c == s[si] => {
                        pi += 1;
                        si += 1;
                        continue;
                    }
                    _ => {}
                }
            }
            if let Some(sp) = star {
                pi = sp + 1;
                star_si += 1;
                si = star_si;
            } else {
                return false;
            }
        }
        while pi < pat.len() && pat[pi] == b'%' {
            pi += 1;
        }
        pi == pat.len()
    }
    match_inner(pattern.as_bytes(), name.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn local_storage_round_trips_batch_and_range_reads() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = LocalVfsStorage::new(dir.path());
        storage
            .write_many_atomic(vec![
                VfsStorageWrite {
                    path: "a/one.txt".to_string(),
                    bytes: Bytes::from_static(b"abcdef"),
                    token_count: None,
                    precondition: None,
                },
                VfsStorageWrite {
                    path: "a/two.txt".to_string(),
                    bytes: Bytes::from_static(b"ghijkl"),
                    token_count: None,
                    precondition: None,
                },
            ])
            .await
            .expect("write_many");

        let range = storage
            .read_range(
                "a/one.txt",
                VfsStorageReadRange {
                    offset: 2,
                    length: 3,
                },
            )
            .await
            .expect("range");
        assert_eq!(&range[..], b"cde");

        let many = storage
            .read_many(&[
                "a/one.txt".to_string(),
                "missing.txt".to_string(),
                "a/two.txt".to_string(),
            ])
            .await
            .expect("read_many");
        assert_eq!(many.len(), 2);
        assert_eq!(&many[0].1[..], b"abcdef");
        assert_eq!(&many[1].1[..], b"ghijkl");
    }

    #[tokio::test]
    async fn changed_only_write_skips_identical_content() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = LocalVfsStorage::new(dir.path());
        storage
            .write("note.txt", Bytes::from_static(b"same"), None)
            .await
            .expect("initial write");

        let results = storage
            .write_many_if_changed_atomic(vec![
                VfsStorageWrite {
                    path: "note.txt".to_string(),
                    bytes: Bytes::from_static(b"same"),
                    token_count: None,
                    precondition: None,
                },
                VfsStorageWrite {
                    path: "other.txt".to_string(),
                    bytes: Bytes::from_static(b"new"),
                    token_count: None,
                    precondition: None,
                },
            ])
            .await
            .expect("changed write");

        let by_path: HashMap<_, _> = results
            .into_iter()
            .map(|result| (result.path.clone(), result))
            .collect();
        assert!(!by_path["note.txt"].changed);
        assert!(by_path["other.txt"].changed);
    }

    #[tokio::test]
    async fn local_storage_lists_metadata_and_subtree_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = LocalVfsStorage::new(dir.path());
        storage.mkdir("root/child").await.expect("mkdir");
        storage
            .write("root/a.txt", Bytes::from_static(b"a"), None)
            .await
            .expect("write a");
        storage
            .write("root/child/b.md", Bytes::from_static(b"b"), None)
            .await
            .expect("write b");

        let listed = storage
            .list_dir_with_metadata(
                "root",
                VfsStorageDirListFilter {
                    name_not_like: Some("%.digest-%".to_string()),
                    ..Default::default()
                },
            )
            .await
            .expect("list dir");
        assert_eq!(listed.len(), 2);
        assert_eq!(listed[0].kind, VfsStorageEntryKind::Directory);

        let subtree = storage
            .list_subtree_file_metadata("root", VfsStorageSubtreeOptions::default())
            .await
            .expect("subtree");
        assert_eq!(
            subtree
                .into_iter()
                .map(|entry| entry.path)
                .collect::<Vec<_>>(),
            vec!["root/a.txt".to_string(), "root/child/b.md".to_string()]
        );
    }

    #[tokio::test]
    async fn local_storage_enforces_write_preconditions() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = LocalVfsStorage::new(dir.path());
        let first = storage
            .write("guarded.txt", Bytes::from_static(b"first"), None)
            .await
            .expect("initial write");
        let precondition = VfsStorageWritePrecondition {
            fingerprint: Some(first.content_hash),
            secondary_fingerprint: None,
        };
        storage
            .write("guarded.txt", Bytes::from_static(b"second"), None)
            .await
            .expect("racing write");
        let err = storage
            .write(
                "guarded.txt",
                Bytes::from_static(b"third"),
                Some(precondition),
            )
            .await
            .expect_err("stale precondition");
        assert!(matches!(err, VfsStorageError::Conflict(_)));
    }

    #[tokio::test]
    async fn local_storage_rejects_parent_escape() {
        let dir = tempfile::tempdir().expect("tempdir");
        let storage = LocalVfsStorage::new(dir.path());
        let err = storage
            .read("../outside.txt")
            .await
            .expect_err("escape rejected");
        assert!(matches!(err, VfsStorageError::BadRequest(_)));
    }
}
