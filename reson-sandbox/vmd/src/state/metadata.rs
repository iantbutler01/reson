use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde_json::Value;

use super::types::VmMetadata;

const METADATA_FILENAME: &str = "vm.json";

fn metadata_path(dir: &Path) -> PathBuf {
    dir.join(METADATA_FILENAME)
}

/// Load VM metadata from `vm.json`.
pub fn load_metadata(dir: &Path) -> Result<VmMetadata> {
    let path = metadata_path(dir);
    let data = fs::read(&path).with_context(|| format!("read metadata {}", path.display()))?;
    let mut meta: VmMetadata = serde_json::from_slice(&data)
        .with_context(|| format!("decode metadata {}", path.display()))?;

    if meta.metadata.is_empty() {
        meta.metadata = Default::default();
    }
    if meta.snapshots.is_empty() {
        meta.snapshots = Default::default();
    }
    Ok(meta)
}

/// Persist VM metadata to `vm.json` atomically.
pub fn save_metadata(dir: &Path, meta: &mut VmMetadata) -> Result<()> {
    meta.updated_at = Utc::now();
    if meta.metadata.is_empty() {
        meta.metadata = Default::default();
    }

    let mut tmp = tempfile::Builder::new()
        .prefix("vm-")
        .suffix(".json")
        .tempfile_in(dir)
        .context("create temp metadata file")?;

    let path = tmp.path().to_path_buf();

    {
        let file = tmp.as_file_mut();
        let mut value = serde_json::to_value(meta)?;
        normalize_timestamps(&mut value);
        let data = serde_json::to_vec_pretty(&value)?;
        file.write_all(&data)?;
        file.sync_all()?;
    }

    let dst = metadata_path(dir);
    fs::rename(path, &dst).with_context(|| format!("rename metadata to {}", dst.display()))?;
    Ok(())
}

fn normalize_timestamps(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                if k.ends_with("_at") {
                    if let Value::String(s) = v {
                        if let Ok(parsed) = DateTime::parse_from_rfc3339(s) {
                            *s = parsed.with_timezone(&Utc).to_rfc3339();
                        }
                    }
                }
                normalize_timestamps(v);
            }
        }
        Value::Array(arr) => {
            for item in arr {
                normalize_timestamps(item);
            }
        }
        _ => {}
    }
}
