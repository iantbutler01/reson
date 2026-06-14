-- Default Postgres manifest/index schema for reson-vfs packed storage.
-- Product-specific owners, auth, leases, and audit rows should live outside these tables.

CREATE TABLE IF NOT EXISTS reson_vfs_entries (
    id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    logical_path TEXT NOT NULL,
    parent_logical_path TEXT NOT NULL DEFAULT '',
    entry_name TEXT NOT NULL,
    entry_kind TEXT NOT NULL CHECK (entry_kind IN ('file', 'directory')),
    size_bytes BIGINT NOT NULL DEFAULT 0 CHECK (size_bytes >= 0),
    content_hash TEXT,
    storage_backend TEXT NOT NULL DEFAULT 'object_store',
    current_version_id TEXT,
    materialization_generation BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope_key, logical_path),
    UNIQUE (scope_key, id)
);

CREATE INDEX IF NOT EXISTS reson_vfs_entries_parent_idx
    ON reson_vfs_entries (scope_key, parent_logical_path, entry_kind, entry_name);

CREATE INDEX IF NOT EXISTS reson_vfs_entries_scope_kind_path_idx
    ON reson_vfs_entries (scope_key, entry_kind, logical_path);

CREATE INDEX IF NOT EXISTS reson_vfs_entries_scope_updated_idx
    ON reson_vfs_entries (scope_key, updated_at DESC);

CREATE TABLE IF NOT EXISTS reson_vfs_packs (
    scope_key TEXT NOT NULL,
    pack_key TEXT NOT NULL,
    total_slot_count INTEGER NOT NULL CHECK (total_slot_count >= 0),
    reference_count INTEGER NOT NULL CHECK (reference_count >= 0),
    total_bytes BIGINT NOT NULL CHECK (total_bytes >= 0),
    compacted_from_pack_keys TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_compaction_checked_at TIMESTAMPTZ,
    PRIMARY KEY (scope_key, pack_key)
);

CREATE TABLE IF NOT EXISTS reson_vfs_file_manifests (
    id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    logical_path TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    logical_size_bytes BIGINT NOT NULL CHECK (logical_size_bytes >= 0),
    pack_key TEXT NOT NULL,
    pack_slot_offset BIGINT NOT NULL CHECK (pack_slot_offset >= 0),
    pack_slot_length BIGINT NOT NULL CHECK (pack_slot_length >= 0),
    pack_slot_compression SMALLINT NOT NULL,
    token_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope_key, id),
    FOREIGN KEY (scope_key, pack_key)
        REFERENCES reson_vfs_packs (scope_key, pack_key)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS reson_vfs_file_manifests_path_idx
    ON reson_vfs_file_manifests (scope_key, logical_path, created_at DESC);

CREATE INDEX IF NOT EXISTS reson_vfs_file_manifests_pack_idx
    ON reson_vfs_file_manifests (scope_key, pack_key);

CREATE TABLE IF NOT EXISTS reson_vfs_file_versions (
    scope_key TEXT NOT NULL,
    logical_path TEXT NOT NULL,
    version_id TEXT NOT NULL,
    version_no BIGINT NOT NULL CHECK (version_no > 0),
    manifest_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    logical_size_bytes BIGINT NOT NULL CHECK (logical_size_bytes >= 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope_key, logical_path, version_id),
    UNIQUE (scope_key, logical_path, version_no),
    FOREIGN KEY (scope_key, manifest_id)
        REFERENCES reson_vfs_file_manifests (scope_key, id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS reson_vfs_file_versions_manifest_idx
    ON reson_vfs_file_versions (scope_key, manifest_id);

CREATE TABLE IF NOT EXISTS reson_vfs_mutations (
    id TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    logical_path TEXT NOT NULL,
    operation TEXT NOT NULL,
    source_path TEXT,
    dest_path TEXT,
    prior_version_id TEXT,
    next_version_id TEXT,
    content_hash_before TEXT,
    content_hash_after TEXT,
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (scope_key, id)
);

CREATE INDEX IF NOT EXISTS reson_vfs_mutations_path_created_idx
    ON reson_vfs_mutations (scope_key, logical_path, created_at DESC);
