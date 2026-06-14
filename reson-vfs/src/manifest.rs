// @dive-file: Generic manifest and pack-slot primitives for optimized VFS storage.
// @dive-rel: Keeps logical file hashes separate from pack object identity and slot coordinates.
// @dive-rel: Provides pack-build planning used by object-backed storage implementations.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    VfsStorageError, VfsStorageResult,
    pack::{BuiltPack, PackBuilder, SlotCompression, hex_hash},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsPackSlotRef {
    pub pack_key: String,
    pub pack_slot_offset: i64,
    pub pack_slot_length: i64,
    pub pack_slot_compression: i16,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsFileManifest {
    pub logical_path: String,
    pub content_hash: String,
    pub logical_size_bytes: i64,
    pub pack_slot: VfsPackSlotRef,
    pub token_count: Option<i32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsObjectHead {
    pub logical_path: String,
    pub content_hash: Option<String>,
    pub size_bytes: i64,
    pub pack_slot: VfsPackSlotRef,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct VfsPackRecord {
    pub pack_key: String,
    pub total_slot_count: i32,
    pub reference_count: i32,
    pub total_bytes: i64,
    pub compacted_from_pack_keys: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
pub struct VfsPackInput<'a> {
    pub logical_path: &'a str,
    pub bytes: &'a [u8],
    pub compression: SlotCompression,
    pub token_count: Option<i32>,
}

#[derive(Clone, Debug)]
pub struct VfsBuiltPackManifest {
    pub pack: BuiltPack,
    pub pack_record: VfsPackRecord,
    pub file_manifests: Vec<VfsFileManifest>,
    pub object_heads: Vec<VfsObjectHead>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfsWriteCandidate {
    pub logical_path: String,
    pub content_hash: String,
    pub logical_size_bytes: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VfsChangedWritePlan {
    pub changed_paths: Vec<String>,
    pub unchanged_paths: Vec<String>,
}

pub fn build_pack_manifest(
    pack_key: impl Into<String>,
    inputs: &[VfsPackInput<'_>],
) -> VfsStorageResult<VfsBuiltPackManifest> {
    if inputs.is_empty() {
        return Err(VfsStorageError::BadRequest(
            "cannot build vfs pack with no files".to_string(),
        ));
    }
    let pack_key = pack_key.into();
    let mut builder = PackBuilder::new();
    let mut logical_hashes = Vec::with_capacity(inputs.len());
    for input in inputs {
        builder.add(input.logical_path, input.bytes, input.compression)?;
        logical_hashes.push(hex_hash(input.bytes));
    }
    let pack = builder.finish()?;
    let mut file_manifests = Vec::with_capacity(inputs.len());
    let mut object_heads = Vec::with_capacity(inputs.len());
    for ((input, slot), content_hash) in inputs.iter().zip(pack.slots.iter()).zip(logical_hashes) {
        let pack_slot = VfsPackSlotRef {
            pack_key: pack_key.clone(),
            pack_slot_offset: checked_i64(slot.pack_slot_offset, "pack_slot_offset")?,
            pack_slot_length: checked_i64(slot.pack_slot_length, "pack_slot_length")?,
            pack_slot_compression: slot.compression.as_db_smallint(),
        };
        file_manifests.push(VfsFileManifest {
            logical_path: input.logical_path.to_string(),
            content_hash: content_hash.clone(),
            logical_size_bytes: checked_i64(input.bytes.len() as u64, "logical_size_bytes")?,
            pack_slot: pack_slot.clone(),
            token_count: input.token_count,
        });
        object_heads.push(VfsObjectHead {
            logical_path: input.logical_path.to_string(),
            content_hash: Some(content_hash),
            size_bytes: checked_i64(input.bytes.len() as u64, "size_bytes")?,
            pack_slot,
        });
    }
    let total_slot_count = checked_i32(pack.slot_count(), "total_slot_count")?;
    Ok(VfsBuiltPackManifest {
        pack_record: VfsPackRecord {
            pack_key,
            total_slot_count,
            reference_count: total_slot_count,
            total_bytes: checked_i64(pack.total_bytes(), "total_bytes")?,
            compacted_from_pack_keys: None,
        },
        pack,
        file_manifests,
        object_heads,
    })
}

pub fn candidate_for_bytes(logical_path: impl Into<String>, bytes: &[u8]) -> VfsWriteCandidate {
    VfsWriteCandidate {
        logical_path: logical_path.into(),
        content_hash: hex_hash(bytes),
        logical_size_bytes: bytes.len() as i64,
    }
}

pub fn plan_changed_writes(
    candidates: &[VfsWriteCandidate],
    current_hash_by_path: &HashMap<String, String>,
) -> VfsChangedWritePlan {
    let mut changed_paths = Vec::new();
    let mut unchanged_paths = Vec::new();
    for candidate in candidates {
        if current_hash_by_path
            .get(&candidate.logical_path)
            .is_some_and(|current_hash| current_hash == &candidate.content_hash)
        {
            unchanged_paths.push(candidate.logical_path.clone());
        } else {
            changed_paths.push(candidate.logical_path.clone());
        }
    }
    VfsChangedWritePlan {
        changed_paths,
        unchanged_paths,
    }
}

fn checked_i64(value: u64, field: &str) -> VfsStorageResult<i64> {
    i64::try_from(value).map_err(|_| VfsStorageError::Internal(format!("{field} exceeds i64::MAX")))
}

fn checked_i32(value: u32, field: &str) -> VfsStorageResult<i32> {
    i32::try_from(value).map_err(|_| VfsStorageError::Internal(format!("{field} exceeds i32::MAX")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::extract_slot;

    #[test]
    fn pack_manifest_uses_logical_file_hashes_and_slot_coordinates() {
        let built = build_pack_manifest(
            "packs/one.nympack",
            &[
                VfsPackInput {
                    logical_path: "a.md",
                    bytes: b"alpha",
                    compression: SlotCompression::Zstd,
                    token_count: Some(1),
                },
                VfsPackInput {
                    logical_path: "b.md",
                    bytes: b"beta beta beta",
                    compression: SlotCompression::Zstd,
                    token_count: Some(3),
                },
            ],
        )
        .expect("pack manifest");

        assert_eq!(built.pack_record.total_slot_count, 2);
        assert_eq!(built.pack_record.reference_count, 2);
        assert_eq!(built.file_manifests[0].content_hash, hex_hash(b"alpha"));
        assert_eq!(
            built.file_manifests[1].content_hash,
            hex_hash(b"beta beta beta")
        );
        assert_eq!(
            built.file_manifests[0].pack_slot.pack_key,
            "packs/one.nympack"
        );

        let second = &built.file_manifests[1].pack_slot;
        let extracted = extract_slot(
            &built.pack.pack_bytes,
            second.pack_slot_offset as u64,
            second.pack_slot_length as u64,
        )
        .expect("extract second");
        assert_eq!(extracted.bytes, b"beta beta beta");
    }

    #[test]
    fn changed_write_plan_uses_candidate_hashes() {
        let candidates = vec![
            candidate_for_bytes("same.md", b"same"),
            candidate_for_bytes("changed.md", b"new"),
            candidate_for_bytes("missing.md", b"body"),
        ];
        let current_hash_by_path = HashMap::from([
            ("same.md".to_string(), hex_hash(b"same")),
            ("changed.md".to_string(), hex_hash(b"old")),
        ]);

        let plan = plan_changed_writes(&candidates, &current_hash_by_path);

        assert_eq!(plan.unchanged_paths, vec!["same.md"]);
        assert_eq!(plan.changed_paths, vec!["changed.md", "missing.md"]);
    }
}
