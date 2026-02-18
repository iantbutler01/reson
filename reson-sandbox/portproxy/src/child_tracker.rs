use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::Mutex;

#[derive(Clone, Default)]
pub struct ChildTracker {
    inner: Arc<Mutex<HashSet<i32>>>,
}

impl ChildTracker {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    pub async fn register(&self, pid: i32) {
        let mut guard = self.inner.lock().await;
        guard.insert(pid);
    }

    pub async fn unregister(&self, pid: i32) {
        let mut guard = self.inner.lock().await;
        guard.remove(&pid);
    }

    pub async fn snapshot(&self) -> Vec<i32> {
        let guard = self.inner.lock().await;
        guard.iter().copied().collect()
    }
}
