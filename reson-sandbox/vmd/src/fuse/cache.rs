use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use super::client::{RemoteDirEntry, RemoteMetadata};

const FILE_TTL: Duration = Duration::from_secs(60);
const DIR_TTL: Duration = Duration::from_secs(5);
const MAX_FILE_BYTES: usize = 10 * 1024 * 1024;
const MAX_TOTAL_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone)]
struct CachedFile {
    bytes: Vec<u8>,
    metadata: Option<RemoteMetadata>,
    expires_at: Instant,
    last_access: Instant,
}

#[derive(Clone)]
struct CachedDir {
    entries: Vec<RemoteDirEntry>,
    expires_at: Instant,
    last_access: Instant,
}

#[derive(Default)]
struct CacheState {
    file_bytes: usize,
    files: HashMap<String, CachedFile>,
    dirs: HashMap<String, CachedDir>,
}

#[derive(Default)]
pub struct NymFuseCache {
    inner: Mutex<CacheState>,
}

impl NymFuseCache {
    pub fn get_file(&self, path: &str) -> Option<(Vec<u8>, Option<RemoteMetadata>)> {
        let mut inner = self.lock_inner();
        let entry = inner.files.get_mut(path)?;
        if entry.expires_at <= Instant::now() {
            let removed = inner.files.remove(path)?;
            inner.file_bytes = inner.file_bytes.saturating_sub(removed.bytes.len());
            return None;
        }
        entry.last_access = Instant::now();
        Some((entry.bytes.clone(), entry.metadata.clone()))
    }

    pub fn put_file(&self, path: &str, bytes: Vec<u8>, metadata: Option<RemoteMetadata>) {
        if bytes.len() > MAX_FILE_BYTES {
            return;
        }
        let mut inner = self.lock_inner();
        if let Some(previous) = inner.files.remove(path) {
            inner.file_bytes = inner.file_bytes.saturating_sub(previous.bytes.len());
        }
        inner.file_bytes += bytes.len();
        inner.files.insert(
            path.to_string(),
            CachedFile {
                bytes,
                metadata,
                expires_at: Instant::now() + FILE_TTL,
                last_access: Instant::now(),
            },
        );
        while inner.file_bytes > MAX_TOTAL_BYTES {
            let Some(oldest_key) = inner
                .files
                .iter()
                .min_by_key(|(_, value)| value.last_access)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            if let Some(removed) = inner.files.remove(oldest_key.as_str()) {
                inner.file_bytes = inner.file_bytes.saturating_sub(removed.bytes.len());
            }
        }
    }

    pub fn get_dir(&self, path: &str) -> Option<Vec<RemoteDirEntry>> {
        let mut inner = self.lock_inner();
        let entry = inner.dirs.get_mut(path)?;
        if entry.expires_at <= Instant::now() {
            inner.dirs.remove(path);
            return None;
        }
        entry.last_access = Instant::now();
        Some(entry.entries.clone())
    }

    pub fn put_dir(&self, path: &str, entries: Vec<RemoteDirEntry>) {
        let mut inner = self.lock_inner();
        inner.dirs.insert(
            path.to_string(),
            CachedDir {
                entries,
                expires_at: Instant::now() + DIR_TTL,
                last_access: Instant::now(),
            },
        );
    }

    pub fn invalidate(&self, path: &str) {
        let mut inner = self.lock_inner();
        if let Some(previous) = inner.files.remove(path) {
            inner.file_bytes = inner.file_bytes.saturating_sub(previous.bytes.len());
        }
        inner.dirs.remove(path);
        if let Some(parent) = parent_path(path) {
            inner.dirs.remove(parent.as_str());
        }
    }

    fn lock_inner(&self) -> MutexGuard<'_, CacheState> {
        self.inner.lock().unwrap_or_else(|err| err.into_inner())
    }
}

fn parent_path(path: &str) -> Option<String> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return None;
    }
    trimmed
        .rsplit_once('/')
        .map(|(parent, _)| parent.to_string())
        .or(Some(String::new()))
}
