// @dive-file: Batteries-included Postgres implementation of the VFS manifest/index boundary.
// @dive-rel: Provides a default durable index for packed storage without coupling core VFS
// @dive-rel: semantics to any product database schema.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use sqlx::{PgPool, Postgres, QueryBuilder, Transaction};
use uuid::Uuid;

use crate::{
    VfsStorageDirListFilter, VfsStorageDirListOrder, VfsStorageEntryKind, VfsStorageError,
    VfsStorageResult,
    index::{
        VfsFileManifestRecord, VfsIndexEntry, VfsIndexEntryWithManifest, VfsIndexScope,
        VfsManifestIndex, VfsManifestRepoint, VfsPackLifecycleIndex, VfsPackRecordWithScope,
        VfsPackedCommit, VfsPackedCommitResult,
    },
    manifest::{VfsFileManifest, VfsPackRecord, VfsPackSlotRef},
};

#[derive(Clone)]
pub struct PostgresVfsManifestIndex {
    pool: PgPool,
}

impl PostgresVfsManifestIndex {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &PgPool {
        &self.pool
    }
}

#[derive(Debug, sqlx::FromRow)]
struct FileManifestRow {
    logical_path: String,
    content_hash: String,
    logical_size_bytes: i64,
    pack_key: String,
    pack_slot_offset: i64,
    pack_slot_length: i64,
    pack_slot_compression: i16,
    token_count: Option<i32>,
}

impl From<FileManifestRow> for VfsFileManifest {
    fn from(row: FileManifestRow) -> Self {
        Self {
            logical_path: row.logical_path,
            content_hash: row.content_hash,
            logical_size_bytes: row.logical_size_bytes,
            pack_slot: VfsPackSlotRef {
                pack_key: row.pack_key,
                pack_slot_offset: row.pack_slot_offset,
                pack_slot_length: row.pack_slot_length,
                pack_slot_compression: row.pack_slot_compression,
            },
            token_count: row.token_count,
        }
    }
}

#[derive(Debug, sqlx::FromRow)]
struct EntryWithManifestRow {
    logical_path: String,
    parent_logical_path: String,
    entry_name: String,
    entry_kind: String,
    size_bytes: i64,
    content_hash: Option<String>,
    current_version_id: Option<String>,
    updated_at: DateTime<Utc>,
    manifest_logical_path: Option<String>,
    manifest_content_hash: Option<String>,
    manifest_logical_size_bytes: Option<i64>,
    manifest_pack_key: Option<String>,
    manifest_pack_slot_offset: Option<i64>,
    manifest_pack_slot_length: Option<i64>,
    manifest_pack_slot_compression: Option<i16>,
    manifest_token_count: Option<i32>,
}

impl TryFrom<EntryWithManifestRow> for VfsIndexEntryWithManifest {
    type Error = VfsStorageError;

    fn try_from(row: EntryWithManifestRow) -> Result<Self, Self::Error> {
        let kind = entry_kind_from_db(&row.entry_kind)?;
        let manifest = match (
            row.manifest_logical_path,
            row.manifest_content_hash,
            row.manifest_logical_size_bytes,
            row.manifest_pack_key,
            row.manifest_pack_slot_offset,
            row.manifest_pack_slot_length,
            row.manifest_pack_slot_compression,
        ) {
            (
                Some(logical_path),
                Some(content_hash),
                Some(logical_size_bytes),
                Some(pack_key),
                Some(pack_slot_offset),
                Some(pack_slot_length),
                Some(pack_slot_compression),
            ) => Some(VfsFileManifest {
                logical_path,
                content_hash,
                logical_size_bytes,
                pack_slot: VfsPackSlotRef {
                    pack_key,
                    pack_slot_offset,
                    pack_slot_length,
                    pack_slot_compression,
                },
                token_count: row.manifest_token_count,
            }),
            _ => None,
        };
        Ok(Self {
            entry: VfsIndexEntry {
                logical_path: row.logical_path,
                parent_logical_path: row.parent_logical_path,
                entry_name: row.entry_name,
                kind,
                size_bytes: row.size_bytes,
                content_hash: row.content_hash,
                current_version: row.current_version_id,
                updated_at: Some(row.updated_at),
            },
            manifest,
        })
    }
}

#[derive(Debug, sqlx::FromRow)]
struct PackRecordRow {
    pack_key: String,
    total_slot_count: i32,
    reference_count: i32,
    total_bytes: i64,
    compacted_from_pack_keys: Option<Vec<String>>,
}

impl From<PackRecordRow> for VfsPackRecord {
    fn from(row: PackRecordRow) -> Self {
        Self {
            pack_key: row.pack_key,
            total_slot_count: row.total_slot_count,
            reference_count: row.reference_count,
            total_bytes: row.total_bytes,
            compacted_from_pack_keys: row.compacted_from_pack_keys,
        }
    }
}

#[derive(Debug, sqlx::FromRow)]
struct PackRecordWithScopeRow {
    scope_key: String,
    pack_key: String,
    total_slot_count: i32,
    reference_count: i32,
    total_bytes: i64,
    compacted_from_pack_keys: Option<Vec<String>>,
    updated_at: DateTime<Utc>,
}

impl From<PackRecordWithScopeRow> for VfsPackRecordWithScope {
    fn from(row: PackRecordWithScopeRow) -> Self {
        Self {
            scope: VfsIndexScope::new(row.scope_key),
            updated_at: row.updated_at,
            pack: VfsPackRecord {
                pack_key: row.pack_key,
                total_slot_count: row.total_slot_count,
                reference_count: row.reference_count,
                total_bytes: row.total_bytes,
                compacted_from_pack_keys: row.compacted_from_pack_keys,
            },
        }
    }
}

#[derive(Debug, sqlx::FromRow)]
struct FileManifestRecordRow {
    id: String,
    scope_key: String,
    logical_path: String,
    content_hash: String,
    logical_size_bytes: i64,
    pack_key: String,
    pack_slot_offset: i64,
    pack_slot_length: i64,
    pack_slot_compression: i16,
    token_count: Option<i32>,
}

impl From<FileManifestRecordRow> for VfsFileManifestRecord {
    fn from(row: FileManifestRecordRow) -> Self {
        Self {
            id: row.id,
            scope: VfsIndexScope::new(row.scope_key),
            manifest: VfsFileManifest {
                logical_path: row.logical_path,
                content_hash: row.content_hash,
                logical_size_bytes: row.logical_size_bytes,
                pack_slot: VfsPackSlotRef {
                    pack_key: row.pack_key,
                    pack_slot_offset: row.pack_slot_offset,
                    pack_slot_length: row.pack_slot_length,
                    pack_slot_compression: row.pack_slot_compression,
                },
                token_count: row.token_count,
            },
        }
    }
}

#[async_trait]
impl VfsManifestIndex for PostgresVfsManifestIndex {
    async fn get_current_file_manifest(
        &self,
        scope: &VfsIndexScope,
        logical_path: &str,
    ) -> VfsStorageResult<Option<VfsFileManifest>> {
        sqlx::query_as::<_, FileManifestRow>(
            r#"
            SELECT
                m.logical_path,
                m.content_hash,
                m.logical_size_bytes,
                m.pack_key,
                m.pack_slot_offset,
                m.pack_slot_length,
                m.pack_slot_compression,
                m.token_count
            FROM reson_vfs_entries e
            JOIN reson_vfs_file_versions v
              ON v.scope_key = e.scope_key
             AND v.logical_path = e.logical_path
             AND v.version_id = e.current_version_id
            JOIN reson_vfs_file_manifests m
              ON m.scope_key = v.scope_key
             AND m.id = v.manifest_id
            WHERE e.scope_key = $1
              AND e.logical_path = $2
              AND e.entry_kind = 'file'
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .fetch_optional(&self.pool)
        .await
        .map(|row| row.map(Into::into))
        .map_err(internal)
    }

    async fn list_current_file_manifests_by_paths(
        &self,
        scope: &VfsIndexScope,
        logical_paths: &[String],
    ) -> VfsStorageResult<Vec<VfsFileManifest>> {
        if logical_paths.is_empty() {
            return Ok(Vec::new());
        }
        sqlx::query_as::<_, FileManifestRow>(
            r#"
            SELECT
                m.logical_path,
                m.content_hash,
                m.logical_size_bytes,
                m.pack_key,
                m.pack_slot_offset,
                m.pack_slot_length,
                m.pack_slot_compression,
                m.token_count
            FROM reson_vfs_entries e
            JOIN reson_vfs_file_versions v
              ON v.scope_key = e.scope_key
             AND v.logical_path = e.logical_path
             AND v.version_id = e.current_version_id
            JOIN reson_vfs_file_manifests m
              ON m.scope_key = v.scope_key
             AND m.id = v.manifest_id
            WHERE e.scope_key = $1
              AND e.logical_path = ANY($2::text[])
              AND e.entry_kind = 'file'
            ORDER BY e.logical_path
            "#,
        )
        .bind(&scope.key)
        .bind(logical_paths)
        .fetch_all(&self.pool)
        .await
        .map(|rows| rows.into_iter().map(Into::into).collect())
        .map_err(internal)
    }

    async fn list_current_file_manifests_in_subtree(
        &self,
        scope: &VfsIndexScope,
        logical_path_prefix: &str,
        limit: Option<i64>,
    ) -> VfsStorageResult<Vec<VfsFileManifest>> {
        sqlx::query_as::<_, FileManifestRow>(
            r#"
            SELECT
                m.logical_path,
                m.content_hash,
                m.logical_size_bytes,
                m.pack_key,
                m.pack_slot_offset,
                m.pack_slot_length,
                m.pack_slot_compression,
                m.token_count
            FROM reson_vfs_entries e
            JOIN reson_vfs_file_versions v
              ON v.scope_key = e.scope_key
             AND v.logical_path = e.logical_path
             AND v.version_id = e.current_version_id
            JOIN reson_vfs_file_manifests m
              ON m.scope_key = v.scope_key
             AND m.id = v.manifest_id
            WHERE e.scope_key = $1
              AND e.entry_kind = 'file'
              AND ($2 = '' OR e.logical_path = $2 OR e.logical_path LIKE ($2 || '/%'))
            ORDER BY e.logical_path
            LIMIT $3
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path_prefix)
        .bind(limit.unwrap_or(i64::MAX))
        .fetch_all(&self.pool)
        .await
        .map(|rows| rows.into_iter().map(Into::into).collect())
        .map_err(internal)
    }

    async fn get_entry_with_manifest(
        &self,
        scope: &VfsIndexScope,
        logical_path: &str,
    ) -> VfsStorageResult<Option<VfsIndexEntryWithManifest>> {
        sqlx::query_as::<_, EntryWithManifestRow>(ENTRY_WITH_MANIFEST_SQL)
            .bind(&scope.key)
            .bind(logical_path)
            .fetch_optional(&self.pool)
            .await
            .map_err(internal)?
            .map(TryInto::try_into)
            .transpose()
    }

    async fn list_entries_with_manifest_by_paths(
        &self,
        scope: &VfsIndexScope,
        logical_paths: &[String],
    ) -> VfsStorageResult<Vec<VfsIndexEntryWithManifest>> {
        if logical_paths.is_empty() {
            return Ok(Vec::new());
        }
        sqlx::query_as::<_, EntryWithManifestRow>(ENTRIES_WITH_MANIFEST_BY_PATHS_SQL)
            .bind(&scope.key)
            .bind(logical_paths)
            .fetch_all(&self.pool)
            .await
            .map_err(internal)?
            .into_iter()
            .map(TryInto::try_into)
            .collect()
    }

    async fn list_dir_with_manifest_attrs(
        &self,
        scope: &VfsIndexScope,
        parent_logical_path: &str,
        filter: VfsStorageDirListFilter,
    ) -> VfsStorageResult<Vec<VfsIndexEntryWithManifest>> {
        let mut qb = QueryBuilder::<Postgres>::new(
            r#"
            SELECT
                e.logical_path,
                e.parent_logical_path,
                e.entry_name,
                e.entry_kind,
                e.size_bytes,
                e.content_hash,
                e.current_version_id,
                e.updated_at,
                m.logical_path AS manifest_logical_path,
                m.content_hash AS manifest_content_hash,
                m.logical_size_bytes AS manifest_logical_size_bytes,
                m.pack_key AS manifest_pack_key,
                m.pack_slot_offset AS manifest_pack_slot_offset,
                m.pack_slot_length AS manifest_pack_slot_length,
                m.pack_slot_compression AS manifest_pack_slot_compression,
                m.token_count AS manifest_token_count
            FROM reson_vfs_entries e
            LEFT JOIN reson_vfs_file_versions v
              ON v.scope_key = e.scope_key
             AND v.logical_path = e.logical_path
             AND v.version_id = e.current_version_id
            LEFT JOIN reson_vfs_file_manifests m
              ON m.scope_key = v.scope_key
             AND m.id = v.manifest_id
            WHERE e.scope_key = "#,
        );
        qb.push_bind(&scope.key);
        qb.push(" AND e.parent_logical_path = ");
        qb.push_bind(parent_logical_path);
        if let Some(pattern) = filter.name_like {
            qb.push(" AND e.entry_name LIKE ");
            qb.push_bind(pattern);
        }
        if let Some(pattern) = filter.name_not_like {
            qb.push(" AND e.entry_name NOT LIKE ");
            qb.push_bind(pattern);
        }
        if let Some(kind) = filter.entry_kind {
            qb.push(" AND e.entry_kind = ");
            qb.push_bind(kind.as_str());
        }
        qb.push(
            match filter.order.unwrap_or(VfsStorageDirListOrder::KindThenName) {
                VfsStorageDirListOrder::KindThenName => {
                    " ORDER BY e.entry_kind DESC, e.entry_name ASC"
                }
                VfsStorageDirListOrder::NameAsc => " ORDER BY e.entry_name ASC",
                VfsStorageDirListOrder::NameDesc => " ORDER BY e.entry_name DESC",
                VfsStorageDirListOrder::UpdatedDesc => " ORDER BY e.updated_at DESC",
            },
        );
        if let Some(limit) = filter.limit {
            qb.push(" LIMIT ");
            qb.push_bind(limit);
        }
        qb.build_query_as::<EntryWithManifestRow>()
            .fetch_all(&self.pool)
            .await
            .map_err(internal)?
            .into_iter()
            .map(TryInto::try_into)
            .collect()
    }

    async fn commit_packed_files(
        &self,
        scope: &VfsIndexScope,
        commit: VfsPackedCommit,
    ) -> VfsStorageResult<VfsPackedCommitResult> {
        if commit.files.is_empty() {
            return Ok(VfsPackedCommitResult {
                committed_paths: Vec::new(),
            });
        }
        let mut tx = self.pool.begin().await.map_err(internal)?;
        let now = Utc::now();
        let entry_rows = commit
            .files
            .iter()
            .map(|file| EntryBatchRow {
                id: Uuid::new_v4().to_string(),
                logical_path: file.logical_path.clone(),
                parent_logical_path: file.parent_logical_path.clone(),
                entry_name: file.entry_name.clone(),
                size_bytes: file.manifest.logical_size_bytes,
                content_hash: Some(file.manifest.content_hash.clone()),
                storage_backend: "object_store".to_string(),
                updated_at: now,
            })
            .collect::<Vec<_>>();
        upsert_entries_batch(&mut tx, scope, &entry_rows).await?;
        insert_pack_record(&mut tx, scope, &commit).await?;

        let manifest_rows = commit
            .files
            .iter()
            .map(|file| ManifestBatchRow {
                id: Uuid::new_v4().to_string(),
                manifest: file.manifest.clone(),
            })
            .collect::<Vec<_>>();
        insert_manifests_batch(&mut tx, scope, &manifest_rows).await?;

        let promote_items = commit
            .files
            .iter()
            .zip(manifest_rows.iter())
            .map(|(file, manifest_row)| PromoteManifestVersionItem {
                version_id: Uuid::new_v4().to_string(),
                manifest_id: manifest_row.id.clone(),
                logical_path: file.logical_path.as_str(),
                content_hash: file.manifest.content_hash.as_str(),
                logical_size_bytes: file.manifest.logical_size_bytes,
                expected_current_version: file.expected_current_version.as_deref(),
            })
            .collect::<Vec<_>>();
        let committed_paths =
            batch_promote_manifest_versions(&mut tx, scope, &promote_items).await?;
        if committed_paths.len() != promote_items.len() {
            let committed = committed_paths
                .iter()
                .map(String::as_str)
                .collect::<std::collections::HashSet<_>>();
            let conflict = promote_items
                .iter()
                .find(|item| !committed.contains(item.logical_path))
                .map(|item| item.logical_path)
                .unwrap_or("");
            return Err(VfsStorageError::Conflict(format!(
                "vfs write precondition failed for {conflict}"
            )));
        }

        tx.commit().await.map_err(internal)?;
        Ok(VfsPackedCommitResult { committed_paths })
    }

    async fn create_directory(
        &self,
        scope: &VfsIndexScope,
        logical_path: &str,
        parent_logical_path: &str,
        entry_name: &str,
    ) -> VfsStorageResult<()> {
        let result = sqlx::query(
            r#"
            INSERT INTO reson_vfs_entries (
                id,
                scope_key,
                logical_path,
                parent_logical_path,
                entry_name,
                entry_kind,
                size_bytes,
                content_hash,
                storage_backend,
                current_version_id,
                materialization_generation,
                updated_at
            )
            VALUES ($1, $2, $3, $4, $5, 'directory', 0, NULL, 'object_store', NULL, 0, now())
            ON CONFLICT (scope_key, logical_path) DO UPDATE
            SET
                parent_logical_path = EXCLUDED.parent_logical_path,
                entry_name = EXCLUDED.entry_name,
                entry_kind = 'directory',
                size_bytes = 0,
                content_hash = NULL,
                storage_backend = EXCLUDED.storage_backend,
                current_version_id = NULL,
                updated_at = now()
            WHERE reson_vfs_entries.entry_kind = 'directory'
            "#,
        )
        .bind(Uuid::new_v4().to_string())
        .bind(&scope.key)
        .bind(logical_path)
        .bind(parent_logical_path)
        .bind(entry_name)
        .execute(&self.pool)
        .await
        .map_err(internal)?;
        if result.rows_affected() == 0 {
            return Err(VfsStorageError::Conflict(format!(
                "vfs file already exists at directory path: {logical_path}"
            )));
        }
        Ok(())
    }

    async fn delete_file_entry(
        &self,
        scope: &VfsIndexScope,
        logical_path: &str,
        expected_current_version: Option<&str>,
    ) -> VfsStorageResult<Option<VfsIndexEntryWithManifest>> {
        let Some(previous) = self.get_entry_with_manifest(scope, logical_path).await? else {
            return Ok(None);
        };
        if previous.entry.kind != VfsStorageEntryKind::File {
            return Err(VfsStorageError::BadRequest(format!(
                "vfs path {logical_path} is not a file"
            )));
        };
        if let Some(expected_current_version) = expected_current_version {
            if previous.entry.current_version.as_deref() != Some(expected_current_version) {
                return Err(VfsStorageError::Conflict(format!(
                    "vfs write precondition failed for {logical_path}"
                )));
            }
        }
        let mut tx = self.pool.begin().await.map_err(internal)?;
        let pack_keys = sqlx::query_scalar::<_, String>(
            r#"
            SELECT DISTINCT pack_key
            FROM reson_vfs_file_manifests
            WHERE scope_key = $1
              AND logical_path = $2
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .fetch_all(&mut *tx)
        .await
        .map_err(internal)?;
        sqlx::query(
            r#"
            DELETE FROM reson_vfs_file_versions
            WHERE scope_key = $1
              AND logical_path = $2
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        sqlx::query(
            r#"
            DELETE FROM reson_vfs_file_manifests
            WHERE scope_key = $1
              AND logical_path = $2
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        sqlx::query(
            r#"
            DELETE FROM reson_vfs_entries
            WHERE scope_key = $1
              AND logical_path = $2
              AND entry_kind = 'file'
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        for pack_key in pack_keys {
            update_pack_reference_count(&mut tx, scope, &pack_key).await?;
        }
        tx.commit().await.map_err(internal)?;
        Ok(Some(previous))
    }

    async fn remove_empty_directory(
        &self,
        scope: &VfsIndexScope,
        logical_path: &str,
    ) -> VfsStorageResult<()> {
        let mut tx = self.pool.begin().await.map_err(internal)?;
        let source_kind = sqlx::query_scalar::<_, String>(
            r#"
            SELECT entry_kind
            FROM reson_vfs_entries
            WHERE scope_key = $1
              AND logical_path = $2
            FOR UPDATE
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .fetch_optional(&mut *tx)
        .await
        .map_err(internal)?;
        let Some(source_kind) = source_kind else {
            return Ok(());
        };
        if source_kind != "directory" {
            return Err(VfsStorageError::BadRequest(format!(
                "vfs path {logical_path} is not a directory"
            )));
        }
        let has_child = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
                SELECT 1
                FROM reson_vfs_entries
                WHERE scope_key = $1
                  AND parent_logical_path = $2
                LIMIT 1
            )
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .fetch_one(&mut *tx)
        .await
        .map_err(internal)?;
        if has_child {
            return Err(VfsStorageError::Conflict(format!(
                "vfs directory {logical_path} is not empty"
            )));
        }
        sqlx::query(
            r#"
            DELETE FROM reson_vfs_entries
            WHERE scope_key = $1
              AND logical_path = $2
              AND entry_kind = 'directory'
            "#,
        )
        .bind(&scope.key)
        .bind(logical_path)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)?;
        Ok(())
    }

    async fn rename_file_entry(
        &self,
        scope: &VfsIndexScope,
        from_logical_path: &str,
        to_logical_path: &str,
        to_parent_logical_path: &str,
        to_entry_name: &str,
    ) -> VfsStorageResult<(VfsIndexEntryWithManifest, VfsIndexEntryWithManifest)> {
        let previous = self
            .get_entry_with_manifest(scope, from_logical_path)
            .await?
            .ok_or_else(|| VfsStorageError::NotFound(from_logical_path.to_string()))?;
        if previous.entry.kind != VfsStorageEntryKind::File {
            return Err(VfsStorageError::BadRequest(format!(
                "vfs path {from_logical_path} is not a file"
            )));
        }
        let mut tx = self.pool.begin().await.map_err(internal)?;
        let destination_exists = sqlx::query_scalar::<_, bool>(
            r#"
            SELECT EXISTS (
                SELECT 1 FROM reson_vfs_entries
                WHERE scope_key = $1 AND logical_path = $2
                UNION ALL
                SELECT 1 FROM reson_vfs_file_versions
                WHERE scope_key = $1 AND logical_path = $2
                UNION ALL
                SELECT 1 FROM reson_vfs_file_manifests
                WHERE scope_key = $1 AND logical_path = $2
            )
            "#,
        )
        .bind(&scope.key)
        .bind(to_logical_path)
        .fetch_one(&mut *tx)
        .await
        .map_err(internal)?;
        if destination_exists {
            return Err(VfsStorageError::Conflict(format!(
                "vfs destination already exists: {to_logical_path}"
            )));
        }
        sqlx::query(
            r#"
            UPDATE reson_vfs_entries
            SET
                logical_path = $3,
                parent_logical_path = $4,
                entry_name = $5,
                materialization_generation = materialization_generation + 1,
                updated_at = now()
            WHERE scope_key = $1
              AND logical_path = $2
              AND entry_kind = 'file'
            "#,
        )
        .bind(&scope.key)
        .bind(from_logical_path)
        .bind(to_logical_path)
        .bind(to_parent_logical_path)
        .bind(to_entry_name)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        sqlx::query(
            r#"
            UPDATE reson_vfs_file_manifests
            SET logical_path = $3
            WHERE scope_key = $1
              AND logical_path = $2
            "#,
        )
        .bind(&scope.key)
        .bind(from_logical_path)
        .bind(to_logical_path)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        sqlx::query(
            r#"
            UPDATE reson_vfs_file_versions
            SET logical_path = $3
            WHERE scope_key = $1
              AND logical_path = $2
            "#,
        )
        .bind(&scope.key)
        .bind(from_logical_path)
        .bind(to_logical_path)
        .execute(&mut *tx)
        .await
        .map_err(internal)?;
        tx.commit().await.map_err(internal)?;
        let current = self
            .get_entry_with_manifest(scope, to_logical_path)
            .await?
            .ok_or_else(|| {
                VfsStorageError::Internal(format!(
                    "renamed vfs entry missing after rename: {to_logical_path}"
                ))
            })?;
        Ok((previous, current))
    }
}

#[async_trait]
impl VfsPackLifecycleIndex for PostgresVfsManifestIndex {
    async fn list_scopes_with_small_packs(
        &self,
        max_total_bytes: i64,
        max_total_slots: i32,
        min_small_packs: i64,
        limit: i64,
    ) -> VfsStorageResult<Vec<VfsIndexScope>> {
        sqlx::query_scalar::<_, String>(
            r#"
            SELECT scope_key
            FROM reson_vfs_packs
            WHERE total_bytes < $1
              AND total_slot_count < $2
              AND reference_count > 0
            GROUP BY scope_key
            HAVING COUNT(*) >= $3
            ORDER BY COUNT(*) DESC
            LIMIT $4
            "#,
        )
        .bind(max_total_bytes)
        .bind(max_total_slots)
        .bind(min_small_packs)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map(|scopes| scopes.into_iter().map(VfsIndexScope::new).collect())
        .map_err(internal)
    }

    async fn list_small_packs_for_scope(
        &self,
        scope: &VfsIndexScope,
        max_total_bytes: i64,
        max_total_slots: i32,
        limit: i64,
    ) -> VfsStorageResult<Vec<VfsPackRecord>> {
        sqlx::query_as::<_, PackRecordRow>(
            r#"
            SELECT
                pack_key,
                total_slot_count,
                reference_count,
                total_bytes,
                compacted_from_pack_keys
            FROM reson_vfs_packs
            WHERE scope_key = $1
              AND total_bytes < $2
              AND total_slot_count < $3
              AND reference_count > 0
            ORDER BY created_at ASC
            LIMIT $4
            "#,
        )
        .bind(&scope.key)
        .bind(max_total_bytes)
        .bind(max_total_slots)
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map(|rows| rows.into_iter().map(Into::into).collect())
        .map_err(internal)
    }

    async fn list_file_manifest_records_by_pack_keys(
        &self,
        scope: &VfsIndexScope,
        pack_keys: &[String],
    ) -> VfsStorageResult<Vec<VfsFileManifestRecord>> {
        if pack_keys.is_empty() {
            return Ok(Vec::new());
        }
        sqlx::query_as::<_, FileManifestRecordRow>(
            r#"
            SELECT
                id,
                scope_key,
                logical_path,
                content_hash,
                logical_size_bytes,
                pack_key,
                pack_slot_offset,
                pack_slot_length,
                pack_slot_compression,
                token_count
            FROM reson_vfs_file_manifests
            WHERE scope_key = $1
              AND pack_key = ANY($2::text[])
            "#,
        )
        .bind(&scope.key)
        .bind(pack_keys)
        .fetch_all(&self.pool)
        .await
        .map(|rows| rows.into_iter().map(Into::into).collect())
        .map_err(internal)
    }

    async fn apply_pack_compaction(
        &self,
        scope: &VfsIndexScope,
        new_pack: VfsPackRecord,
        repoints: &[VfsManifestRepoint],
        old_pack_refcount_decrements: &[(String, i32)],
    ) -> VfsStorageResult<()> {
        let mut tx = self.pool.begin().await.map_err(internal)?;
        insert_pack_record_direct(&mut tx, scope, &new_pack).await?;
        repoint_manifests_to_new_pack_batch(&mut tx, scope, repoints).await?;
        for (pack_key, decrement) in old_pack_refcount_decrements {
            bump_pack_reference_count(&mut tx, scope, pack_key, -*decrement).await?;
        }
        tx.commit().await.map_err(internal)?;
        Ok(())
    }

    async fn correct_pack_refcount_drift(&self) -> VfsStorageResult<u64> {
        sqlx::query(
            r#"
            UPDATE reson_vfs_packs p
            SET reference_count = sub.live_count,
                updated_at = now()
            FROM (
                SELECT
                    p2.scope_key,
                    p2.pack_key,
                    COALESCE(
                        (
                            SELECT count(*)
                            FROM reson_vfs_file_manifests m
                            WHERE m.scope_key = p2.scope_key
                              AND m.pack_key = p2.pack_key
                        ),
                        0
                    )::int AS live_count
                FROM reson_vfs_packs p2
                WHERE p2.reference_count != COALESCE(
                    (
                        SELECT count(*)
                        FROM reson_vfs_file_manifests m
                        WHERE m.scope_key = p2.scope_key
                          AND m.pack_key = p2.pack_key
                    ),
                    0
                )
            ) sub
            WHERE p.scope_key = sub.scope_key
              AND p.pack_key = sub.pack_key
            "#,
        )
        .execute(&self.pool)
        .await
        .map(|result| result.rows_affected())
        .map_err(internal)
    }

    async fn list_zero_reference_packs(
        &self,
        limit: i64,
    ) -> VfsStorageResult<Vec<VfsPackRecordWithScope>> {
        sqlx::query_as::<_, PackRecordWithScopeRow>(
            r#"
            SELECT
                scope_key,
                pack_key,
                total_slot_count,
                reference_count,
                total_bytes,
                compacted_from_pack_keys,
                updated_at
            FROM reson_vfs_packs
            WHERE reference_count <= 0
            ORDER BY updated_at ASC
            LIMIT $1
            "#,
        )
        .bind(limit)
        .fetch_all(&self.pool)
        .await
        .map(|rows| rows.into_iter().map(Into::into).collect())
        .map_err(internal)
    }

    async fn recount_pack_reference_count(
        &self,
        scope: &VfsIndexScope,
        pack_key: &str,
    ) -> VfsStorageResult<i32> {
        let count = sqlx::query_scalar::<_, i64>(
            r#"
            SELECT COUNT(*)
            FROM reson_vfs_file_manifests
            WHERE scope_key = $1
              AND pack_key = $2
            "#,
        )
        .bind(&scope.key)
        .bind(pack_key)
        .fetch_one(&self.pool)
        .await
        .map_err(internal)?;
        sqlx::query(
            r#"
            UPDATE reson_vfs_packs
            SET reference_count = $3,
                updated_at = now(),
                last_compaction_checked_at = now()
            WHERE scope_key = $1
              AND pack_key = $2
            "#,
        )
        .bind(&scope.key)
        .bind(pack_key)
        .bind(count as i32)
        .execute(&self.pool)
        .await
        .map_err(internal)?;
        Ok(count as i32)
    }

    async fn delete_pack_records(&self, packs: &[(VfsIndexScope, String)]) -> VfsStorageResult<()> {
        if packs.is_empty() {
            return Ok(());
        }
        let mut by_scope = std::collections::HashMap::<String, Vec<String>>::new();
        for (scope, pack_key) in packs {
            by_scope
                .entry(scope.key.clone())
                .or_default()
                .push(pack_key.clone());
        }
        let mut tx = self.pool.begin().await.map_err(internal)?;
        for (scope_key, pack_keys) in by_scope {
            sqlx::query(
                r#"
                DELETE FROM reson_vfs_packs
                WHERE scope_key = $1
                  AND pack_key = ANY($2::text[])
                "#,
            )
            .bind(&scope_key)
            .bind(pack_keys)
            .execute(&mut *tx)
            .await
            .map_err(internal)?;
        }
        tx.commit().await.map_err(internal)?;
        Ok(())
    }
}

const ENTRY_WITH_MANIFEST_SQL: &str = r#"
SELECT
    e.logical_path,
    e.parent_logical_path,
    e.entry_name,
    e.entry_kind,
    e.size_bytes,
    e.content_hash,
    e.current_version_id,
    e.updated_at,
    m.logical_path AS manifest_logical_path,
    m.content_hash AS manifest_content_hash,
    m.logical_size_bytes AS manifest_logical_size_bytes,
    m.pack_key AS manifest_pack_key,
    m.pack_slot_offset AS manifest_pack_slot_offset,
    m.pack_slot_length AS manifest_pack_slot_length,
    m.pack_slot_compression AS manifest_pack_slot_compression,
    m.token_count AS manifest_token_count
FROM reson_vfs_entries e
LEFT JOIN reson_vfs_file_versions v
  ON v.scope_key = e.scope_key
 AND v.logical_path = e.logical_path
 AND v.version_id = e.current_version_id
LEFT JOIN reson_vfs_file_manifests m
  ON m.scope_key = v.scope_key
 AND m.id = v.manifest_id
WHERE e.scope_key = $1
  AND e.logical_path = $2
"#;

const ENTRIES_WITH_MANIFEST_BY_PATHS_SQL: &str = r#"
SELECT
    e.logical_path,
    e.parent_logical_path,
    e.entry_name,
    e.entry_kind,
    e.size_bytes,
    e.content_hash,
    e.current_version_id,
    e.updated_at,
    m.logical_path AS manifest_logical_path,
    m.content_hash AS manifest_content_hash,
    m.logical_size_bytes AS manifest_logical_size_bytes,
    m.pack_key AS manifest_pack_key,
    m.pack_slot_offset AS manifest_pack_slot_offset,
    m.pack_slot_length AS manifest_pack_slot_length,
    m.pack_slot_compression AS manifest_pack_slot_compression,
    m.token_count AS manifest_token_count
FROM reson_vfs_entries e
LEFT JOIN reson_vfs_file_versions v
  ON v.scope_key = e.scope_key
 AND v.logical_path = e.logical_path
 AND v.version_id = e.current_version_id
LEFT JOIN reson_vfs_file_manifests m
  ON m.scope_key = v.scope_key
 AND m.id = v.manifest_id
WHERE e.scope_key = $1
  AND e.logical_path = ANY($2::text[])
"#;

struct EntryBatchRow {
    id: String,
    logical_path: String,
    parent_logical_path: String,
    entry_name: String,
    size_bytes: i64,
    content_hash: Option<String>,
    storage_backend: String,
    updated_at: DateTime<Utc>,
}

struct ManifestBatchRow {
    id: String,
    manifest: VfsFileManifest,
}

struct PromoteManifestVersionItem<'a> {
    version_id: String,
    manifest_id: String,
    logical_path: &'a str,
    content_hash: &'a str,
    logical_size_bytes: i64,
    expected_current_version: Option<&'a str>,
}

async fn insert_pack_record(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    commit: &VfsPackedCommit,
) -> VfsStorageResult<()> {
    sqlx::query(
        r#"
        INSERT INTO reson_vfs_packs (
            scope_key,
            pack_key,
            total_slot_count,
            reference_count,
            total_bytes,
            compacted_from_pack_keys
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (scope_key, pack_key) DO NOTHING
        "#,
    )
    .bind(&scope.key)
    .bind(&commit.pack.pack_key)
    .bind(commit.pack.total_slot_count)
    .bind(commit.pack.reference_count)
    .bind(commit.pack.total_bytes)
    .bind(commit.pack.compacted_from_pack_keys.as_deref())
    .execute(&mut **tx)
    .await
    .map_err(internal)?;
    Ok(())
}

async fn insert_pack_record_direct(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    pack: &VfsPackRecord,
) -> VfsStorageResult<()> {
    sqlx::query(
        r#"
        INSERT INTO reson_vfs_packs (
            scope_key,
            pack_key,
            total_slot_count,
            reference_count,
            total_bytes,
            compacted_from_pack_keys
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        "#,
    )
    .bind(&scope.key)
    .bind(&pack.pack_key)
    .bind(pack.total_slot_count)
    .bind(pack.reference_count)
    .bind(pack.total_bytes)
    .bind(pack.compacted_from_pack_keys.as_deref())
    .execute(&mut **tx)
    .await
    .map_err(internal)?;
    Ok(())
}

async fn bump_pack_reference_count(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    pack_key: &str,
    by: i32,
) -> VfsStorageResult<()> {
    sqlx::query(
        r#"
        UPDATE reson_vfs_packs
        SET reference_count = reference_count + $3,
            updated_at = now()
        WHERE scope_key = $1
          AND pack_key = $2
        "#,
    )
    .bind(&scope.key)
    .bind(pack_key)
    .bind(by)
    .execute(&mut **tx)
    .await
    .map_err(internal)?;
    Ok(())
}

async fn repoint_manifests_to_new_pack_batch(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    updates: &[VfsManifestRepoint],
) -> VfsStorageResult<()> {
    if updates.is_empty() {
        return Ok(());
    }
    const REPOINT_CHUNK_SIZE: usize = 8_000;
    for chunk in updates.chunks(REPOINT_CHUNK_SIZE) {
        let mut qb = QueryBuilder::<Postgres>::new(
            r#"
            UPDATE reson_vfs_file_manifests AS m
            SET
                pack_key = u.new_pack_key,
                pack_slot_offset = u.new_pack_slot_offset,
                pack_slot_length = u.new_pack_slot_length,
                pack_slot_compression = u.new_pack_slot_compression
            FROM (
            "#,
        );
        qb.push_values(chunk, |mut row, update| {
            row.push_bind(update.manifest_id.as_str())
                .push_bind(scope.key.as_str())
                .push_bind(update.new_pack_key.as_str())
                .push_bind(update.new_pack_slot_offset)
                .push_bind(update.new_pack_slot_length)
                .push_bind(update.new_pack_slot_compression);
        });
        qb.push(
            r#"
            ) AS u(
                manifest_id,
                scope_key,
                new_pack_key,
                new_pack_slot_offset,
                new_pack_slot_length,
                new_pack_slot_compression
            )
            WHERE m.id = u.manifest_id
              AND m.scope_key = u.scope_key
            "#,
        );
        qb.build().execute(&mut **tx).await.map_err(internal)?;
    }
    Ok(())
}

async fn update_pack_reference_count(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    pack_key: &str,
) -> VfsStorageResult<()> {
    let live_count = sqlx::query_scalar::<_, i64>(
        r#"
        SELECT COUNT(*)::bigint
        FROM reson_vfs_file_manifests
        WHERE scope_key = $1
          AND pack_key = $2
        "#,
    )
    .bind(&scope.key)
    .bind(pack_key)
    .fetch_one(&mut **tx)
    .await
    .map_err(internal)?;
    sqlx::query(
        r#"
        UPDATE reson_vfs_packs
        SET
            reference_count = $3,
            updated_at = now()
        WHERE scope_key = $1
          AND pack_key = $2
        "#,
    )
    .bind(&scope.key)
    .bind(pack_key)
    .bind(live_count as i32)
    .execute(&mut **tx)
    .await
    .map_err(internal)?;
    Ok(())
}

async fn upsert_entries_batch(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    rows: &[EntryBatchRow],
) -> VfsStorageResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut qb = QueryBuilder::<Postgres>::new(
        r#"
        INSERT INTO reson_vfs_entries (
            id,
            scope_key,
            logical_path,
            parent_logical_path,
            entry_name,
            entry_kind,
            size_bytes,
            content_hash,
            storage_backend,
            current_version_id,
            materialization_generation,
            updated_at
        )
        "#,
    );
    qb.push_values(rows, |mut row, req| {
        row.push_bind(req.id.as_str())
            .push_bind(scope.key.as_str())
            .push_bind(req.logical_path.as_str())
            .push_bind(req.parent_logical_path.as_str())
            .push_bind(req.entry_name.as_str())
            .push_bind("file")
            .push_bind(req.size_bytes)
            .push_bind(req.content_hash.as_deref())
            .push_bind(req.storage_backend.as_str())
            .push_bind(Option::<&str>::None)
            .push_bind(0_i64)
            .push_bind(req.updated_at);
    });
    qb.push(
        r#"
        ON CONFLICT (scope_key, logical_path) DO UPDATE
        SET
            parent_logical_path = EXCLUDED.parent_logical_path,
            entry_name = EXCLUDED.entry_name,
            entry_kind = EXCLUDED.entry_kind,
            size_bytes = EXCLUDED.size_bytes,
            content_hash = EXCLUDED.content_hash,
            storage_backend = EXCLUDED.storage_backend,
            updated_at = EXCLUDED.updated_at
        "#,
    );
    qb.build().execute(&mut **tx).await.map_err(internal)?;
    Ok(())
}

async fn insert_manifests_batch(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    rows: &[ManifestBatchRow],
) -> VfsStorageResult<()> {
    if rows.is_empty() {
        return Ok(());
    }
    let mut qb = QueryBuilder::<Postgres>::new(
        r#"
        INSERT INTO reson_vfs_file_manifests (
            id,
            scope_key,
            logical_path,
            content_hash,
            logical_size_bytes,
            pack_key,
            pack_slot_offset,
            pack_slot_length,
            pack_slot_compression,
            token_count
        )
        "#,
    );
    qb.push_values(rows, |mut row, req| {
        row.push_bind(req.id.as_str())
            .push_bind(scope.key.as_str())
            .push_bind(req.manifest.logical_path.as_str())
            .push_bind(req.manifest.content_hash.as_str())
            .push_bind(req.manifest.logical_size_bytes)
            .push_bind(req.manifest.pack_slot.pack_key.as_str())
            .push_bind(req.manifest.pack_slot.pack_slot_offset)
            .push_bind(req.manifest.pack_slot.pack_slot_length)
            .push_bind(req.manifest.pack_slot.pack_slot_compression)
            .push_bind(req.manifest.token_count);
    });
    qb.build().execute(&mut **tx).await.map_err(internal)?;
    Ok(())
}

async fn batch_promote_manifest_versions(
    tx: &mut Transaction<'_, Postgres>,
    scope: &VfsIndexScope,
    items: &[PromoteManifestVersionItem<'_>],
) -> VfsStorageResult<Vec<String>> {
    if items.is_empty() {
        return Ok(Vec::new());
    }
    let mut qb = build_batch_promote_manifest_versions_query(scope, items);
    let rows: Vec<(String,)> = qb
        .build_query_as()
        .fetch_all(&mut **tx)
        .await
        .map_err(internal)?;
    Ok(rows.into_iter().map(|(path,)| path).collect())
}

fn build_batch_promote_manifest_versions_query<'a>(
    scope: &'a VfsIndexScope,
    items: &'a [PromoteManifestVersionItem<'a>],
) -> QueryBuilder<'a, Postgres> {
    let mut qb = QueryBuilder::<Postgres>::new(
        "WITH input(version_id, manifest_id, scope_key, logical_path, content_hash, logical_size_bytes, expected_current_version_id) AS (",
    );
    qb.push_values(items, |mut row, item| {
        row.push_bind(item.version_id.as_str())
            .push_bind(item.manifest_id.as_str())
            .push_bind(scope.key.as_str())
            .push_bind(item.logical_path)
            .push_bind(item.content_hash)
            .push_bind(item.logical_size_bytes)
            .push_bind(item.expected_current_version);
    });
    qb.push(
        r#"
        ),
        next_versions AS (
            SELECT
                i.version_id,
                i.manifest_id,
                i.scope_key,
                i.logical_path,
                i.content_hash,
                i.logical_size_bytes,
                i.expected_current_version_id,
                COALESCE(
                    (
                        SELECT MAX(v.version_no)
                        FROM reson_vfs_file_versions v
                        WHERE v.scope_key = i.scope_key
                          AND v.logical_path = i.logical_path
                    ),
                    0
                ) + 1 AS new_version_no
            FROM input i
        ),
        inserted_versions AS (
            INSERT INTO reson_vfs_file_versions (
                scope_key,
                logical_path,
                version_id,
                version_no,
                manifest_id,
                content_hash,
                logical_size_bytes
            )
            SELECT
                scope_key,
                logical_path,
                version_id,
                new_version_no,
                manifest_id,
                content_hash,
                logical_size_bytes
            FROM next_versions
            RETURNING scope_key, logical_path, version_id
        ),
        cas_updates AS (
            UPDATE reson_vfs_entries e
            SET current_version_id = iv.version_id,
                materialization_generation = e.materialization_generation + 1,
                updated_at = now()
            FROM inserted_versions iv
            JOIN input i
              ON i.scope_key = iv.scope_key
             AND i.logical_path = iv.logical_path
            WHERE e.scope_key = iv.scope_key
              AND e.logical_path = iv.logical_path
              AND e.current_version_id IS NOT DISTINCT FROM i.expected_current_version_id
            RETURNING e.logical_path
        )
        SELECT logical_path FROM cas_updates
        "#,
    );
    qb
}

fn entry_kind_from_db(value: &str) -> VfsStorageResult<VfsStorageEntryKind> {
    match value {
        "file" => Ok(VfsStorageEntryKind::File),
        "directory" => Ok(VfsStorageEntryKind::Directory),
        other => Err(VfsStorageError::Internal(format!(
            "unknown vfs entry kind from postgres: {other}"
        ))),
    }
}

fn internal(error: impl std::fmt::Display) -> VfsStorageError {
    VfsStorageError::Internal(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::Execute;

    #[test]
    fn entry_kind_mapping_accepts_known_values() {
        assert_eq!(
            entry_kind_from_db("file").unwrap(),
            VfsStorageEntryKind::File
        );
        assert_eq!(
            entry_kind_from_db("directory").unwrap(),
            VfsStorageEntryKind::Directory
        );
    }

    #[test]
    fn entry_kind_mapping_rejects_unknown_values() {
        let err = entry_kind_from_db("socket").unwrap_err();
        assert!(err.to_string().contains("unknown vfs entry kind"));
    }

    #[test]
    fn batch_promote_sql_preserves_other_you_batch_cas_shape() {
        let scope = VfsIndexScope::new("nym:one");
        let items = vec![
            PromoteManifestVersionItem {
                version_id: "version-1".to_string(),
                manifest_id: "manifest-1".to_string(),
                logical_path: "a.md",
                content_hash: "hash-a",
                logical_size_bytes: 5,
                expected_current_version: None,
            },
            PromoteManifestVersionItem {
                version_id: "version-2".to_string(),
                manifest_id: "manifest-2".to_string(),
                logical_path: "b.md",
                content_hash: "hash-b",
                logical_size_bytes: 7,
                expected_current_version: Some("old-version"),
            },
        ];
        let mut qb = build_batch_promote_manifest_versions_query(&scope, &items);
        let query = qb.build_query_as::<(String,)>();
        let sql = query.sql();

        assert!(sql.contains("WITH input("), "{sql}");
        assert!(sql.contains("next_versions AS"), "{sql}");
        assert!(sql.contains("inserted_versions AS"), "{sql}");
        assert!(sql.contains("cas_updates AS"), "{sql}");
        assert!(sql.contains("SELECT MAX(v.version_no)"), "{sql}");
        assert!(
            sql.contains("current_version_id IS NOT DISTINCT FROM"),
            "{sql}"
        );
        assert!(!sql.contains("FOR UPDATE"), "{sql}");
    }
}
