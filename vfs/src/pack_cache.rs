// @dive-file: Bounded in-memory LRU cache for immutable VFS pack objects.
// @dive-rel: Shared `Arc<Vec<u8>>` values let many concurrent slot extractions reuse the same
// @dive-rel: fetched buffer with zero copy. Packs are immutable, so this cache has no
// @dive-rel: invalidation logic beyond explicit remove and size-based eviction.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// LRU cache bounded by total cached bytes across all entries.
///
/// Insertion semantics:
/// * On `put`, the entry is inserted (or replaces an existing entry with the same key) and
///   becomes the most-recently-used.
/// * If the total cached bytes exceeds `max_total_bytes` after the insert, the least-recently-used
///   entries are evicted until the total fits.
/// * Entries larger than `max_total_bytes` on their own are still inserted and will evict
///   everything else. The caller should size the cache to fit its expected working set.
pub struct PackCache {
    inner: Mutex<Inner>,
    max_total_bytes: usize,
}

struct Inner {
    entries: HashMap<String, Entry>,
    /// Monotonic counter used as the "last touched" timestamp for LRU. Higher = more recent.
    sequence: u64,
    total_bytes: usize,
}

struct Entry {
    bytes: Arc<Vec<u8>>,
    last_touched: u64,
    size: usize,
}

impl PackCache {
    pub fn new(max_total_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                sequence: 0,
                total_bytes: 0,
            }),
            max_total_bytes,
        }
    }

    pub fn get(&self, pack_key: &str) -> Option<Arc<Vec<u8>>> {
        let mut inner = self.inner.lock().ok()?;
        inner.sequence = inner.sequence.saturating_add(1);
        let new_seq = inner.sequence;
        let entry = inner.entries.get_mut(pack_key)?;
        entry.last_touched = new_seq;
        Some(Arc::clone(&entry.bytes))
    }

    pub fn put(&self, pack_key: String, bytes: Arc<Vec<u8>>) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        let size = bytes.len();
        inner.sequence = inner.sequence.saturating_add(1);
        let seq = inner.sequence;
        if let Some(existing) = inner.entries.remove(&pack_key) {
            inner.total_bytes = inner.total_bytes.saturating_sub(existing.size);
        }
        inner.entries.insert(
            pack_key,
            Entry {
                bytes,
                last_touched: seq,
                size,
            },
        );
        inner.total_bytes = inner.total_bytes.saturating_add(size);
        self.evict_if_needed(&mut inner);
    }

    pub fn remove(&self, pack_key: &str) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if let Some(existing) = inner.entries.remove(pack_key) {
            inner.total_bytes = inner.total_bytes.saturating_sub(existing.size);
        }
    }

    pub fn total_bytes(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.total_bytes)
            .unwrap_or(0)
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .map(|inner| inner.entries.len())
            .unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn evict_if_needed(&self, inner: &mut Inner) {
        // Stop once the cache fits OR only one entry remains. The "one entry remains" rule
        // lets a single oversized entry survive insertion — the caller took a hit but at least
        // their most-recent pack is still cached instead of being nuked alongside the rest.
        while inner.total_bytes > self.max_total_bytes && inner.entries.len() > 1 {
            let victim_key = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_touched)
                .map(|(k, _)| k.clone());
            let Some(key) = victim_key else { break };
            if let Some(victim) = inner.entries.remove(&key) {
                inner.total_bytes = inner.total_bytes.saturating_sub(victim.size);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_bytes(size: usize) -> Arc<Vec<u8>> {
        Arc::new(vec![0u8; size])
    }

    #[test]
    fn get_hit_and_miss() {
        let cache = PackCache::new(1024);
        assert!(cache.get("missing").is_none());
        cache.put("a".to_string(), mk_bytes(10));
        let got = cache.get("a").unwrap();
        assert_eq!(got.len(), 10);
    }

    #[test]
    fn put_replaces_existing_key() {
        let cache = PackCache::new(1024);
        cache.put("a".to_string(), mk_bytes(10));
        assert_eq!(cache.total_bytes(), 10);
        cache.put("a".to_string(), mk_bytes(25));
        assert_eq!(cache.total_bytes(), 25);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn remove_shrinks_total_bytes() {
        let cache = PackCache::new(1024);
        cache.put("a".to_string(), mk_bytes(10));
        cache.put("b".to_string(), mk_bytes(20));
        assert_eq!(cache.total_bytes(), 30);
        cache.remove("a");
        assert_eq!(cache.total_bytes(), 20);
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_some());
    }

    #[test]
    fn eviction_drops_least_recently_used() {
        let cache = PackCache::new(50);
        cache.put("a".to_string(), mk_bytes(20));
        cache.put("b".to_string(), mk_bytes(20));
        let _ = cache.get("a");
        cache.put("c".to_string(), mk_bytes(20));
        assert!(cache.get("a").is_some());
        assert!(cache.get("b").is_none());
        assert!(cache.get("c").is_some());
    }

    #[test]
    fn oversized_entry_evicts_everything_else() {
        let cache = PackCache::new(50);
        cache.put("a".to_string(), mk_bytes(20));
        cache.put("b".to_string(), mk_bytes(20));
        cache.put("big".to_string(), mk_bytes(60));
        assert!(cache.get("a").is_none());
        assert!(cache.get("b").is_none());
        assert!(cache.get("big").is_some());
    }

    #[test]
    fn arcs_share_across_readers() {
        let cache = PackCache::new(1024);
        let original = mk_bytes(32);
        cache.put("k".to_string(), Arc::clone(&original));
        let a = cache.get("k").unwrap();
        let b = cache.get("k").unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert!(Arc::ptr_eq(&a, &original));
    }
}
