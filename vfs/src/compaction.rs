// @dive-file: Generic pack coalescing and cleanup primitives for optimized VFS storage.
// @dive-rel: Lifts the production OtherYou pack-compaction flow into Chevalier while leaving
// @dive-rel: product scheduling, leases, and process lifecycle decisions to callers.

use std::{
    collections::{BTreeSet, HashMap},
    sync::Arc,
};

use async_trait::async_trait;
use futures::{StreamExt as _, stream};
use uuid::Uuid;

use crate::{
    VfsStorageError, VfsStorageResult,
    index::{VfsIndexScope, VfsManifestRepoint, VfsPackLifecycleIndex},
    manifest::VfsPackRecord,
    object_store::{ObjectDeleteCondition, ObjectStoreClient, ObjectWriteCondition},
    pack::{BuiltPack, PackBuilder, SlotCompression, extract_slot},
};

pub const PACK_TARGET_BYTES: i64 = 16 * 1024 * 1024;
pub const PACK_TARGET_SLOTS: i32 = 8_192;
pub const MIN_SMALL_PACKS_TO_COALESCE: i64 = 2;
pub const MAX_PACKS_PER_COALESCE_PASS: i64 = 5_000;

#[derive(Debug, Clone)]
pub struct VfsPackCompactorConfig {
    pub pack_key_prefix: String,
    pub max_total_bytes: i64,
    pub max_total_slots: i32,
    pub min_small_packs: i64,
    pub max_packs_per_pass: i64,
    pub fetch_parallelism: usize,
    pub delete_parallelism: usize,
    pub orphan_sweeper_min_age_seconds: i64,
}

impl Default for VfsPackCompactorConfig {
    fn default() -> Self {
        Self {
            pack_key_prefix: "packs".to_string(),
            max_total_bytes: PACK_TARGET_BYTES,
            max_total_slots: PACK_TARGET_SLOTS,
            min_small_packs: MIN_SMALL_PACKS_TO_COALESCE,
            max_packs_per_pass: MAX_PACKS_PER_COALESCE_PASS,
            fetch_parallelism: 1024,
            delete_parallelism: 256,
            orphan_sweeper_min_age_seconds: 300,
        }
    }
}

#[derive(Clone)]
pub struct VfsPackCompactor {
    index: Arc<dyn VfsPackLifecycleIndex>,
    blobs: Arc<dyn VfsPackBlobStore>,
    cfg: VfsPackCompactorConfig,
}

impl VfsPackCompactor {
    pub fn new(
        index: Arc<dyn VfsPackLifecycleIndex>,
        blobs: Arc<dyn VfsPackBlobStore>,
        cfg: VfsPackCompactorConfig,
    ) -> Self {
        Self { index, blobs, cfg }
    }

    pub async fn list_candidate_scopes(&self, limit: i64) -> VfsStorageResult<Vec<VfsIndexScope>> {
        self.index
            .list_scopes_with_small_packs(
                self.cfg.max_total_bytes,
                self.cfg.max_total_slots,
                self.cfg.min_small_packs,
                limit,
            )
            .await
    }

    pub async fn coalesce_scope_batch(
        &self,
        scope: &VfsIndexScope,
    ) -> VfsStorageResult<VfsPackCoalesceStats> {
        let mut small_packs = self
            .index
            .list_small_packs_for_scope(
                scope,
                self.cfg.max_total_bytes,
                self.cfg.max_total_slots,
                self.cfg.max_packs_per_pass,
            )
            .await?;

        if (small_packs.len() as i64) < self.cfg.min_small_packs {
            return Ok(VfsPackCoalesceStats::default());
        }

        let mut cumulative_bytes: i64 = 0;
        let mut keep_count = 0usize;
        for pack in &small_packs {
            let next = cumulative_bytes.saturating_add(pack.total_bytes);
            if keep_count >= self.cfg.min_small_packs as usize && next > self.cfg.max_total_bytes {
                break;
            }
            cumulative_bytes = next;
            keep_count += 1;
        }
        small_packs.truncate(keep_count);

        let old_pack_keys = small_packs
            .iter()
            .map(|pack| pack.pack_key.clone())
            .collect::<Vec<_>>();
        let manifests = self
            .index
            .list_file_manifest_records_by_pack_keys(scope, &old_pack_keys)
            .await?;
        if manifests.is_empty() {
            return Ok(VfsPackCoalesceStats::default());
        }

        let unique_pack_keys = old_pack_keys
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let fetched_packs = stream::iter(unique_pack_keys)
            .map(|pack_key| {
                let blobs = self.blobs.clone();
                async move {
                    let bytes = blobs.fetch_pack_bytes(&pack_key).await?;
                    Ok::<_, VfsStorageError>((pack_key, bytes))
                }
            })
            .buffer_unordered(self.cfg.fetch_parallelism.max(1))
            .collect::<Vec<_>>()
            .await;

        let mut pack_bytes_by_key = HashMap::new();
        for fetched in fetched_packs {
            let (pack_key, bytes) = fetched?;
            if let Some(bytes) = bytes {
                pack_bytes_by_key.insert(pack_key, bytes);
            }
        }

        let mut missing_source_slots = 0_u32;
        let mut slot_bytes_by_manifest = Vec::with_capacity(manifests.len());
        for manifest_record in manifests {
            let Some(pack_bytes) =
                pack_bytes_by_key.get(&manifest_record.manifest.pack_slot.pack_key)
            else {
                #[cfg(feature = "tracing")]
                tracing::warn!(
                    pack_key = %manifest_record.manifest.pack_slot.pack_key,
                    logical_path = %manifest_record.manifest.logical_path,
                    "vfs_pack_compactor: source pack missing in object store; skipping manifest"
                );
                missing_source_slots += 1;
                continue;
            };
            let slot = &manifest_record.manifest.pack_slot;
            let extracted = extract_slot(
                pack_bytes.as_slice(),
                slot.pack_slot_offset as u64,
                slot.pack_slot_length as u64,
            )?;
            slot_bytes_by_manifest.push((manifest_record, extracted.bytes));
        }
        if slot_bytes_by_manifest.is_empty() {
            return Ok(VfsPackCoalesceStats::default());
        }

        let path_and_bytes = slot_bytes_by_manifest
            .iter()
            .map(|(manifest_record, bytes)| {
                (manifest_record.manifest.logical_path.clone(), bytes.clone())
            })
            .collect::<Vec<_>>();
        let built = build_coalesced_pack(path_and_bytes).await?;
        let new_pack_key = self.blobs.build_pack_key(scope);
        self.blobs
            .put_pack_bytes(&new_pack_key, &built.pack_bytes)
            .await?;

        let migrated_count = i32::try_from(slot_bytes_by_manifest.len()).map_err(|_| {
            VfsStorageError::Internal("vfs compaction migrated slot count exceeds i32".to_string())
        })?;
        let new_pack = VfsPackRecord {
            pack_key: new_pack_key.clone(),
            total_slot_count: migrated_count,
            reference_count: migrated_count,
            total_bytes: built.pack_bytes.len() as i64,
            compacted_from_pack_keys: Some(old_pack_keys),
        };
        let repoints = slot_bytes_by_manifest
            .iter()
            .enumerate()
            .map(|(i, (manifest_record, _))| {
                let slot = &built.slots[i];
                VfsManifestRepoint {
                    manifest_id: manifest_record.id.clone(),
                    new_pack_key: new_pack_key.clone(),
                    new_pack_slot_offset: slot.pack_slot_offset as i64,
                    new_pack_slot_length: slot.pack_slot_length as i64,
                    new_pack_slot_compression: slot.compression.as_db_smallint(),
                }
            })
            .collect::<Vec<_>>();
        let mut old_refcount_decrements = HashMap::<String, i32>::new();
        for (manifest_record, _) in &slot_bytes_by_manifest {
            *old_refcount_decrements
                .entry(manifest_record.manifest.pack_slot.pack_key.clone())
                .or_insert(0) += 1;
        }
        let old_refcount_decrements = old_refcount_decrements.into_iter().collect::<Vec<_>>();
        self.index
            .apply_pack_compaction(scope, new_pack, &repoints, &old_refcount_decrements)
            .await?;

        Ok(VfsPackCoalesceStats {
            packs_coalesced: old_refcount_decrements.len() as u32,
            slots_migrated: migrated_count as u32,
            missing_source_slots,
        })
    }

    pub async fn sweep_zero_reference_packs(
        &self,
        limit: i64,
    ) -> VfsStorageResult<VfsPackSweepStats> {
        let mut stats = VfsPackSweepStats {
            refcounts_corrected: self.index.correct_pack_refcount_drift().await? as u32,
            ..Default::default()
        };
        let now = chrono::Utc::now();
        let candidates = self
            .index
            .list_zero_reference_packs(limit)
            .await?
            .into_iter()
            .filter(|pack| {
                (now - pack.updated_at).num_seconds() >= self.cfg.orphan_sweeper_min_age_seconds
            })
            .collect::<Vec<_>>();

        let delete_results = stream::iter(candidates)
            .map(|pack| {
                let index = self.index.clone();
                let blobs = self.blobs.clone();
                async move {
                    let result: VfsStorageResult<bool> = async {
                        let live_count = index
                            .recount_pack_reference_count(&pack.scope, &pack.pack.pack_key)
                            .await?;
                        if live_count > 0 {
                            return Ok(false);
                        }
                        blobs.delete_pack(&pack.pack.pack_key).await?;
                        Ok(true)
                    }
                    .await;
                    (pack, result)
                }
            })
            .buffer_unordered(self.cfg.delete_parallelism.max(1))
            .collect::<Vec<_>>()
            .await;

        let mut deletable = Vec::new();
        for (pack, result) in delete_results {
            match result {
                Ok(true) => deletable.push((pack.scope, pack.pack.pack_key)),
                Ok(false) => stats.refcounts_corrected += 1,
                Err(err) => {
                    #[cfg(feature = "tracing")]
                    tracing::warn!(
                        pack_key = %pack.pack.pack_key,
                        error = %err,
                        "vfs_pack_compactor: pack delete failed; will retry next sweep"
                    );
                    #[cfg(not(feature = "tracing"))]
                    let _ = err;
                    stats.delete_failures += 1;
                }
            }
        }
        stats.packs_deleted = deletable.len() as u32;
        self.index.delete_pack_records(&deletable).await?;
        Ok(stats)
    }
}

#[async_trait]
pub trait VfsPackBlobStore: Send + Sync {
    async fn fetch_pack_bytes(&self, pack_key: &str) -> VfsStorageResult<Option<Arc<Vec<u8>>>>;
    async fn put_pack_bytes(&self, pack_key: &str, bytes: &[u8]) -> VfsStorageResult<()>;
    async fn delete_pack(&self, pack_key: &str) -> VfsStorageResult<()>;
    fn build_pack_key(&self, scope: &VfsIndexScope) -> String;
}

#[derive(Clone)]
pub struct ObjectStorePackBlobStore {
    store: Arc<dyn ObjectStoreClient>,
    pack_key_prefix: String,
}

impl ObjectStorePackBlobStore {
    pub fn new(store: Arc<dyn ObjectStoreClient>, pack_key_prefix: impl Into<String>) -> Self {
        Self {
            store,
            pack_key_prefix: pack_key_prefix.into(),
        }
    }
}

#[async_trait]
impl VfsPackBlobStore for ObjectStorePackBlobStore {
    async fn fetch_pack_bytes(&self, pack_key: &str) -> VfsStorageResult<Option<Arc<Vec<u8>>>> {
        self.store
            .get_object_async(pack_key)
            .await
            .map(|bytes| bytes.map(Arc::new))
    }

    async fn put_pack_bytes(&self, pack_key: &str, bytes: &[u8]) -> VfsStorageResult<()> {
        self.store
            .put_object_async(
                pack_key,
                bytes,
                ObjectWriteCondition {
                    if_absent: true,
                    ..Default::default()
                },
            )
            .await
    }

    async fn delete_pack(&self, pack_key: &str) -> VfsStorageResult<()> {
        self.store
            .delete_object_async(pack_key, ObjectDeleteCondition::default())
            .await
    }

    fn build_pack_key(&self, scope: &VfsIndexScope) -> String {
        let prefix = self.pack_key_prefix.trim_matches('/');
        let scope = sanitize_scope_for_key(&scope.key);
        let pack_id = Uuid::new_v4().simple();
        if prefix.is_empty() {
            format!("{scope}/{pack_id}.pack")
        } else {
            format!("{prefix}/{scope}/{pack_id}.pack")
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VfsPackCoalesceStats {
    pub packs_coalesced: u32,
    pub slots_migrated: u32,
    pub missing_source_slots: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VfsPackSweepStats {
    pub packs_deleted: u32,
    pub refcounts_corrected: u32,
    pub delete_failures: u32,
}

fn sanitize_scope_for_key(scope: &str) -> String {
    scope
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(feature = "tokio")]
async fn build_coalesced_pack(inputs: Vec<(String, Vec<u8>)>) -> VfsStorageResult<BuiltPack> {
    tokio::task::spawn_blocking(move || build_coalesced_pack_sync(&inputs))
        .await
        .map_err(|err| VfsStorageError::Internal(format!("vfs pack builder join failed: {err}")))?
}

#[cfg(not(feature = "tokio"))]
async fn build_coalesced_pack(inputs: Vec<(String, Vec<u8>)>) -> VfsStorageResult<BuiltPack> {
    build_coalesced_pack_sync(&inputs)
}

fn build_coalesced_pack_sync(inputs: &[(String, Vec<u8>)]) -> VfsStorageResult<BuiltPack> {
    let mut builder = PackBuilder::new();
    for (logical_path, bytes) in inputs {
        builder.add(logical_path, bytes, SlotCompression::Zstd)?;
    }
    builder.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    use crate::{
        index::{VfsFileManifestRecord, VfsPackRecordWithScope},
        manifest::{VfsPackInput, build_pack_manifest},
    };

    #[test]
    fn object_store_blob_key_uses_sanitized_scope() {
        assert_eq!(sanitize_scope_for_key("nym:one/two"), "nym_one_two");
    }

    #[tokio::test]
    async fn coalesce_scope_batch_repoints_manifests_to_merged_pack() {
        let scope = VfsIndexScope::new("scope-one");
        let first = build_pack_manifest(
            "packs/scope-one/first.pack",
            &[VfsPackInput {
                logical_path: "a.md",
                bytes: b"alpha",
                compression: SlotCompression::Zstd,
                token_count: Some(1),
            }],
        )
        .expect("first pack");
        let second = build_pack_manifest(
            "packs/scope-one/second.pack",
            &[VfsPackInput {
                logical_path: "b.md",
                bytes: b"beta",
                compression: SlotCompression::Zstd,
                token_count: Some(1),
            }],
        )
        .expect("second pack");
        let index = Arc::new(MemoryLifecycleIndex::new(
            scope.clone(),
            vec![first.pack_record.clone(), second.pack_record.clone()],
            vec![
                VfsFileManifestRecord {
                    id: "manifest-a".to_string(),
                    scope: scope.clone(),
                    manifest: first.file_manifests[0].clone(),
                },
                VfsFileManifestRecord {
                    id: "manifest-b".to_string(),
                    scope: scope.clone(),
                    manifest: second.file_manifests[0].clone(),
                },
            ],
        ));
        let blobs = Arc::new(MemoryBlobStore::new(
            "packs/scope-one/merged.pack",
            vec![
                (
                    first.pack_record.pack_key.clone(),
                    first.pack.pack_bytes.clone(),
                ),
                (
                    second.pack_record.pack_key.clone(),
                    second.pack.pack_bytes.clone(),
                ),
            ],
        ));
        let compactor = VfsPackCompactor::new(
            index.clone(),
            blobs.clone(),
            VfsPackCompactorConfig {
                fetch_parallelism: 2,
                ..Default::default()
            },
        );

        let stats = compactor
            .coalesce_scope_batch(&scope)
            .await
            .expect("coalesce");

        assert_eq!(
            stats,
            VfsPackCoalesceStats {
                packs_coalesced: 2,
                slots_migrated: 2,
                missing_source_slots: 0,
            }
        );
        let inner = index.inner.lock().expect("index lock");
        assert!(inner.applied);
        assert_eq!(inner.packs[0].reference_count, 0);
        assert_eq!(inner.packs[1].reference_count, 0);
        assert!(inner.manifests.iter().all(|manifest| {
            manifest.manifest.pack_slot.pack_key == "packs/scope-one/merged.pack"
        }));
        assert!(
            blobs
                .inner
                .lock()
                .expect("blob lock")
                .contains_key("packs/scope-one/merged.pack")
        );
    }

    #[tokio::test]
    async fn sweep_zero_reference_packs_deletes_only_old_dead_packs() {
        let scope = VfsIndexScope::new("scope-one");
        let index = Arc::new(MemoryLifecycleIndex::new(
            scope.clone(),
            Vec::new(),
            Vec::new(),
        ));
        index.set_refcount_drift(2);
        let old_dead = VfsPackRecordWithScope {
            scope: scope.clone(),
            updated_at: chrono::Utc::now() - chrono::Duration::seconds(600),
            pack: VfsPackRecord {
                pack_key: "packs/old-dead.pack".to_string(),
                total_slot_count: 0,
                reference_count: 0,
                total_bytes: 0,
                compacted_from_pack_keys: None,
            },
        };
        let old_live = VfsPackRecordWithScope {
            scope: scope.clone(),
            updated_at: chrono::Utc::now() - chrono::Duration::seconds(600),
            pack: VfsPackRecord {
                pack_key: "packs/old-live.pack".to_string(),
                total_slot_count: 0,
                reference_count: 0,
                total_bytes: 0,
                compacted_from_pack_keys: None,
            },
        };
        let fresh_dead = VfsPackRecordWithScope {
            scope: scope.clone(),
            updated_at: chrono::Utc::now(),
            pack: VfsPackRecord {
                pack_key: "packs/fresh-dead.pack".to_string(),
                total_slot_count: 0,
                reference_count: 0,
                total_bytes: 0,
                compacted_from_pack_keys: None,
            },
        };
        index.add_zero_reference_pack(old_dead, 0);
        index.add_zero_reference_pack(old_live, 1);
        index.add_zero_reference_pack(fresh_dead, 0);
        let blobs = Arc::new(MemoryBlobStore::new(
            "unused.pack",
            vec![
                ("packs/old-dead.pack".to_string(), b"old-dead".to_vec()),
                ("packs/old-live.pack".to_string(), b"old-live".to_vec()),
                ("packs/fresh-dead.pack".to_string(), b"fresh-dead".to_vec()),
            ],
        ));
        let compactor = VfsPackCompactor::new(
            index.clone(),
            blobs.clone(),
            VfsPackCompactorConfig {
                orphan_sweeper_min_age_seconds: 300,
                delete_parallelism: 2,
                ..Default::default()
            },
        );

        let stats = compactor
            .sweep_zero_reference_packs(10)
            .await
            .expect("sweep");

        assert_eq!(
            stats,
            VfsPackSweepStats {
                packs_deleted: 1,
                refcounts_corrected: 3,
                delete_failures: 0,
            }
        );
        let blob_keys = blobs.inner.lock().expect("blob lock");
        assert!(!blob_keys.contains_key("packs/old-dead.pack"));
        assert!(blob_keys.contains_key("packs/old-live.pack"));
        assert!(blob_keys.contains_key("packs/fresh-dead.pack"));
        drop(blob_keys);
        assert_eq!(
            index.inner.lock().expect("index lock").deleted_records,
            vec![(scope, "packs/old-dead.pack".to_string())]
        );
    }

    struct MemoryLifecycleIndex {
        scope: VfsIndexScope,
        inner: Mutex<MemoryLifecycleInner>,
    }

    struct MemoryLifecycleInner {
        packs: Vec<VfsPackRecord>,
        manifests: Vec<VfsFileManifestRecord>,
        applied: bool,
        zero_reference_packs: Vec<VfsPackRecordWithScope>,
        deleted_records: Vec<(VfsIndexScope, String)>,
        refcount_drift: u64,
        recounts: HashMap<String, i32>,
    }

    impl MemoryLifecycleIndex {
        fn new(
            scope: VfsIndexScope,
            packs: Vec<VfsPackRecord>,
            manifests: Vec<VfsFileManifestRecord>,
        ) -> Self {
            Self {
                scope,
                inner: Mutex::new(MemoryLifecycleInner {
                    packs,
                    manifests,
                    applied: false,
                    zero_reference_packs: Vec::new(),
                    deleted_records: Vec::new(),
                    refcount_drift: 0,
                    recounts: HashMap::new(),
                }),
            }
        }

        fn add_zero_reference_pack(&self, pack: VfsPackRecordWithScope, live_count: i32) {
            let mut inner = self.inner.lock().expect("index lock");
            inner
                .recounts
                .insert(pack.pack.pack_key.clone(), live_count);
            inner.zero_reference_packs.push(pack);
        }

        fn set_refcount_drift(&self, count: u64) {
            self.inner.lock().expect("index lock").refcount_drift = count;
        }
    }

    #[async_trait]
    impl VfsPackLifecycleIndex for MemoryLifecycleIndex {
        async fn list_scopes_with_small_packs(
            &self,
            _max_total_bytes: i64,
            _max_total_slots: i32,
            _min_small_packs: i64,
            _limit: i64,
        ) -> VfsStorageResult<Vec<VfsIndexScope>> {
            Ok(vec![self.scope.clone()])
        }

        async fn list_small_packs_for_scope(
            &self,
            scope: &VfsIndexScope,
            _max_total_bytes: i64,
            _max_total_slots: i32,
            _limit: i64,
        ) -> VfsStorageResult<Vec<VfsPackRecord>> {
            assert_eq!(scope, &self.scope);
            Ok(self.inner.lock().expect("index lock").packs.clone())
        }

        async fn list_file_manifest_records_by_pack_keys(
            &self,
            scope: &VfsIndexScope,
            pack_keys: &[String],
        ) -> VfsStorageResult<Vec<VfsFileManifestRecord>> {
            assert_eq!(scope, &self.scope);
            Ok(self
                .inner
                .lock()
                .expect("index lock")
                .manifests
                .iter()
                .filter(|manifest| pack_keys.contains(&manifest.manifest.pack_slot.pack_key))
                .cloned()
                .collect())
        }

        async fn apply_pack_compaction(
            &self,
            scope: &VfsIndexScope,
            new_pack: VfsPackRecord,
            repoints: &[VfsManifestRepoint],
            old_pack_refcount_decrements: &[(String, i32)],
        ) -> VfsStorageResult<()> {
            assert_eq!(scope, &self.scope);
            let mut inner = self.inner.lock().expect("index lock");
            for repoint in repoints {
                let manifest = inner
                    .manifests
                    .iter_mut()
                    .find(|manifest| manifest.id == repoint.manifest_id)
                    .expect("manifest repoint target");
                manifest.manifest.pack_slot.pack_key = repoint.new_pack_key.clone();
                manifest.manifest.pack_slot.pack_slot_offset = repoint.new_pack_slot_offset;
                manifest.manifest.pack_slot.pack_slot_length = repoint.new_pack_slot_length;
                manifest.manifest.pack_slot.pack_slot_compression =
                    repoint.new_pack_slot_compression;
            }
            for (pack_key, decrement) in old_pack_refcount_decrements {
                let pack = inner
                    .packs
                    .iter_mut()
                    .find(|pack| &pack.pack_key == pack_key)
                    .expect("old pack");
                pack.reference_count -= *decrement;
            }
            inner.packs.push(new_pack);
            inner.applied = true;
            Ok(())
        }

        async fn correct_pack_refcount_drift(&self) -> VfsStorageResult<u64> {
            Ok(self.inner.lock().expect("index lock").refcount_drift)
        }

        async fn list_zero_reference_packs(
            &self,
            _limit: i64,
        ) -> VfsStorageResult<Vec<VfsPackRecordWithScope>> {
            Ok(self
                .inner
                .lock()
                .expect("index lock")
                .zero_reference_packs
                .clone())
        }

        async fn recount_pack_reference_count(
            &self,
            _scope: &VfsIndexScope,
            pack_key: &str,
        ) -> VfsStorageResult<i32> {
            Ok(*self
                .inner
                .lock()
                .expect("index lock")
                .recounts
                .get(pack_key)
                .unwrap_or(&0))
        }

        async fn delete_pack_records(
            &self,
            packs: &[(VfsIndexScope, String)],
        ) -> VfsStorageResult<()> {
            self.inner
                .lock()
                .expect("index lock")
                .deleted_records
                .extend_from_slice(packs);
            Ok(())
        }
    }

    struct MemoryBlobStore {
        next_pack_key: String,
        inner: Mutex<HashMap<String, Arc<Vec<u8>>>>,
    }

    impl MemoryBlobStore {
        fn new(next_pack_key: impl Into<String>, packs: Vec<(String, Vec<u8>)>) -> Self {
            Self {
                next_pack_key: next_pack_key.into(),
                inner: Mutex::new(
                    packs
                        .into_iter()
                        .map(|(key, bytes)| (key, Arc::new(bytes)))
                        .collect(),
                ),
            }
        }
    }

    #[async_trait]
    impl VfsPackBlobStore for MemoryBlobStore {
        async fn fetch_pack_bytes(&self, pack_key: &str) -> VfsStorageResult<Option<Arc<Vec<u8>>>> {
            Ok(self.inner.lock().expect("blob lock").get(pack_key).cloned())
        }

        async fn put_pack_bytes(&self, pack_key: &str, bytes: &[u8]) -> VfsStorageResult<()> {
            self.inner
                .lock()
                .expect("blob lock")
                .insert(pack_key.to_string(), Arc::new(bytes.to_vec()));
            Ok(())
        }

        async fn delete_pack(&self, pack_key: &str) -> VfsStorageResult<()> {
            self.inner.lock().expect("blob lock").remove(pack_key);
            Ok(())
        }

        fn build_pack_key(&self, _scope: &VfsIndexScope) -> String {
            self.next_pack_key.clone()
        }
    }
}
