//! Generic key-value store abstraction
//!
//! Provides Python Cache/Store parity with prefix/suffix namespacing and mailbox support.

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use std::collections::{HashMap, HashSet};

use crate::error::Result;

/// Generic key-value store trait (matches Python's Cache/Store)
#[async_trait]
pub trait Store: Send + Sync {
    /// Get a value by key
    async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>>;

    /// Set a value by key
    async fn set<T: Serialize + Send + Sync>(&self, key: &str, value: &T) -> Result<()>;

    /// Delete a key
    async fn delete(&self, key: &str) -> Result<()>;

    /// Clear all keys (respecting prefix if set)
    async fn clear(&self) -> Result<()>;

    /// Get all key-value pairs
    async fn get_all(&self) -> Result<HashMap<String, serde_json::Value>>;

    /// Get all keys
    async fn keys(&self) -> Result<HashSet<String>>;

    /// Publish a message to a mailbox (pub/sub)
    async fn publish_to_mailbox(&self, mailbox_id: &str, value: &serde_json::Value) -> Result<()>;

    /// Get a message from a mailbox (blocking with optional timeout)
    async fn get_message(
        &self,
        mailbox_id: &str,
        timeout_secs: Option<f64>,
    ) -> Result<Option<serde_json::Value>>;

    /// Close/cleanup the store
    async fn close(&self) -> Result<()>;

    // Namespacing methods
    /// Get current prefix
    async fn get_prefix(&self) -> String;

    /// Get current suffix
    async fn get_suffix(&self) -> String;

    /// Apply prefix and suffix to a key
    async fn apply_key_modifications(&self, key: &str) -> String {
        let prefix = self.get_prefix().await;
        let suffix = self.get_suffix().await;
        let sep = self.get_affix_sep();

        let mut modified = key.to_string();

        if !prefix.is_empty() {
            modified = format!("{}{}{}", prefix, sep, modified);
        }

        if !suffix.is_empty() {
            modified = format!("{}{}{}", modified, sep, suffix);
        }

        modified
    }

    /// Get the affix separator (default ":")
    fn get_affix_sep(&self) -> &str {
        ":"
    }
}
