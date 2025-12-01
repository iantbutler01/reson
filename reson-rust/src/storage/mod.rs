//! Storage backends for conversation history and generic key-value storage
//!
//! Provides abstraction over different storage mechanisms.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::utils::ConversationMessage;

// Generic key-value store (matches Python's Cache/Store)
pub mod store;
pub use store::Store;

// Redis backend (behind feature flag)
#[cfg(feature = "storage")]
pub mod redis_store;
#[cfg(feature = "storage")]
pub use redis_store::RedisStore;

// Postgres backend (behind feature flag)
#[cfg(feature = "storage")]
pub mod postgres_store;
#[cfg(feature = "storage")]
pub use postgres_store::PostgresStore;

/// Trait for storage backends
#[async_trait]
pub trait Storage: Send + Sync {
    /// Store messages for a conversation
    async fn store_messages(&mut self, conversation_id: &str, messages: Vec<ConversationMessage>) -> Result<()>;

    /// Retrieve messages for a conversation
    async fn get_messages(&self, conversation_id: &str) -> Result<Vec<ConversationMessage>>;

    /// Append a message to an existing conversation
    async fn append_message(&mut self, conversation_id: &str, message: ConversationMessage) -> Result<()>;

    /// Delete a conversation
    async fn delete_conversation(&mut self, conversation_id: &str) -> Result<()>;

    /// List all conversation IDs
    async fn list_conversations(&self) -> Result<Vec<String>>;

    /// Check if a conversation exists
    async fn conversation_exists(&self, conversation_id: &str) -> Result<bool>;
}

/// In-memory storage backend for conversations
pub struct MemoryStore {
    conversations: Arc<RwLock<HashMap<String, Vec<ConversationMessage>>>>,
}

/// In-memory generic key-value store (implements Store trait)
pub struct MemoryKVStore {
    data: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    pub prefix: Arc<RwLock<String>>,
    pub suffix: Arc<RwLock<String>>,
}

impl MemoryKVStore {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            prefix: Arc::new(RwLock::new(String::new())),
            suffix: Arc::new(RwLock::new(String::new())),
        }
    }
}

impl Default for MemoryKVStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Store for MemoryKVStore {
    async fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let modified_key = self.apply_key_modifications(key).await;
        let store = self.data.read().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        if let Some(value) = store.get(&modified_key) {
            let deserialized = serde_json::from_value(value.clone())
                .map_err(|e| Error::NonRetryable(format!("Deserialization error: {}", e)))?;
            Ok(Some(deserialized))
        } else {
            Ok(None)
        }
    }

    async fn set<T: serde::Serialize + Send + Sync>(&self, key: &str, value: &T) -> Result<()> {
        let modified_key = self.apply_key_modifications(key).await;
        let serialized = serde_json::to_value(value)
            .map_err(|e| Error::NonRetryable(format!("Serialization error: {}", e)))?;

        let mut store = self.data.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        store.insert(modified_key, serialized);
        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let modified_key = self.apply_key_modifications(key).await;
        let mut store = self.data.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        store.remove(&modified_key);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let prefix = self.get_prefix().await;
        let suffix = self.get_suffix().await;

        let mut store = self.data.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        if prefix.is_empty() && suffix.is_empty() {
            // Clear everything
            store.clear();
        } else {
            // Clear only keys matching prefix/suffix
            let sep = self.get_affix_sep();
            let keys_to_remove: Vec<String> = store
                .keys()
                .filter(|k| {
                    let mut matches = true;
                    if !prefix.is_empty() {
                        matches = matches && k.starts_with(&format!("{}{}", prefix, sep));
                    }
                    if !suffix.is_empty() {
                        matches = matches && k.ends_with(&format!("{}{}", sep, suffix));
                    }
                    matches
                })
                .cloned()
                .collect();

            for key in keys_to_remove {
                store.remove(&key);
            }
        }

        Ok(())
    }

    async fn get_all(&self) -> Result<HashMap<String, serde_json::Value>> {
        let store = self.data.read().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        Ok(store.clone())
    }

    async fn keys(&self) -> Result<std::collections::HashSet<String>> {
        let store = self.data.read().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        Ok(store.keys().cloned().collect())
    }

    async fn publish_to_mailbox(&self, mailbox_id: &str, value: &serde_json::Value) -> Result<()> {
        // For in-memory implementation, we'll use a simple list stored at the mailbox key
        let modified_mailbox_id = self.apply_key_modifications(mailbox_id).await;

        let mut store = self.data.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        let mailbox = store
            .entry(modified_mailbox_id)
            .or_insert_with(|| serde_json::json!([]));

        if let Some(arr) = mailbox.as_array_mut() {
            arr.push(value.clone());
        } else {
            *mailbox = serde_json::json!([value.clone()]);
        }

        Ok(())
    }

    async fn get_message(&self, mailbox_id: &str, _timeout_secs: Option<f64>) -> Result<Option<serde_json::Value>> {
        // For in-memory implementation, we can't truly block, so we just pop the first message
        let modified_mailbox_id = self.apply_key_modifications(mailbox_id).await;

        let mut store = self.data.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        if let Some(mailbox) = store.get_mut(&modified_mailbox_id) {
            if let Some(arr) = mailbox.as_array_mut() {
                if !arr.is_empty() {
                    return Ok(Some(arr.remove(0)));
                }
            }
        }

        Ok(None)
    }

    async fn close(&self) -> Result<()> {
        // No-op for in-memory store
        Ok(())
    }

    async fn get_prefix(&self) -> String {
        self.prefix.read().unwrap().clone()
    }

    async fn get_suffix(&self) -> String {
        self.suffix.read().unwrap().clone()
    }
}

impl MemoryStore {
    pub fn new() -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for MemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Storage for MemoryStore {
    async fn store_messages(&mut self, conversation_id: &str, messages: Vec<ConversationMessage>) -> Result<()> {
        let mut store = self.conversations.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        store.insert(conversation_id.to_string(), messages);
        Ok(())
    }

    async fn get_messages(&self, conversation_id: &str) -> Result<Vec<ConversationMessage>> {
        let store = self.conversations.read().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        Ok(store.get(conversation_id).cloned().unwrap_or_default())
    }

    async fn append_message(&mut self, conversation_id: &str, message: ConversationMessage) -> Result<()> {
        let mut store = self.conversations.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        store
            .entry(conversation_id.to_string())
            .or_insert_with(Vec::new)
            .push(message);

        Ok(())
    }

    async fn delete_conversation(&mut self, conversation_id: &str) -> Result<()> {
        let mut store = self.conversations.write().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        store.remove(conversation_id);
        Ok(())
    }

    async fn list_conversations(&self) -> Result<Vec<String>> {
        let store = self.conversations.read().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        Ok(store.keys().cloned().collect())
    }

    async fn conversation_exists(&self, conversation_id: &str) -> Result<bool> {
        let store = self.conversations.read().map_err(|e| {
            Error::NonRetryable(format!("Lock poisoned: {}", e))
        })?;

        Ok(store.contains_key(conversation_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[tokio::test]
    async fn test_memory_store_new() {
        let store = MemoryStore::new();
        let conversations = store.list_conversations().await.unwrap();
        assert_eq!(conversations.len(), 0);
    }

    #[tokio::test]
    async fn test_memory_store_store_and_get() {
        let mut store = MemoryStore::new();
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::user("Hello")),
            ConversationMessage::Chat(ChatMessage::assistant("Hi!")),
        ];

        store.store_messages("conv1", messages.clone()).await.unwrap();

        let retrieved = store.get_messages("conv1").await.unwrap();
        assert_eq!(retrieved.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_store_get_nonexistent() {
        let store = MemoryStore::new();
        let messages = store.get_messages("nonexistent").await.unwrap();
        assert_eq!(messages.len(), 0);
    }

    #[tokio::test]
    async fn test_memory_store_append() {
        let mut store = MemoryStore::new();

        store
            .append_message("conv1", ConversationMessage::Chat(ChatMessage::user("First")))
            .await
            .unwrap();

        store
            .append_message("conv1", ConversationMessage::Chat(ChatMessage::assistant("Second")))
            .await
            .unwrap();

        let messages = store.get_messages("conv1").await.unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_store_delete() {
        let mut store = MemoryStore::new();
        let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];

        store.store_messages("conv1", messages).await.unwrap();
        assert!(store.conversation_exists("conv1").await.unwrap());

        store.delete_conversation("conv1").await.unwrap();
        assert!(!store.conversation_exists("conv1").await.unwrap());
    }

    #[tokio::test]
    async fn test_memory_store_list_conversations() {
        let mut store = MemoryStore::new();

        store
            .append_message("conv1", ConversationMessage::Chat(ChatMessage::user("A")))
            .await
            .unwrap();

        store
            .append_message("conv2", ConversationMessage::Chat(ChatMessage::user("B")))
            .await
            .unwrap();

        let mut conversations = store.list_conversations().await.unwrap();
        conversations.sort();

        assert_eq!(conversations, vec!["conv1", "conv2"]);
    }

    #[tokio::test]
    async fn test_memory_store_conversation_exists() {
        let mut store = MemoryStore::new();

        assert!(!store.conversation_exists("conv1").await.unwrap());

        store
            .append_message("conv1", ConversationMessage::Chat(ChatMessage::user("Hello")))
            .await
            .unwrap();

        assert!(store.conversation_exists("conv1").await.unwrap());
    }

    // MemoryKVStore tests
    #[tokio::test]
    async fn test_memory_kv_store_new() {
        let store = MemoryKVStore::new();
        let keys = store.keys().await.unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[tokio::test]
    async fn test_memory_kv_store_set_and_get() {
        let store = MemoryKVStore::new();

        store.set("key1", &"value1".to_string()).await.unwrap();

        let result: Option<String> = store.get("key1").await.unwrap();
        assert_eq!(result, Some("value1".to_string()));
    }

    #[tokio::test]
    async fn test_memory_kv_store_get_nonexistent() {
        let store = MemoryKVStore::new();
        let result: Option<String> = store.get("nonexistent").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn test_memory_kv_store_delete() {
        let store = MemoryKVStore::new();

        store.set("key1", &"value1".to_string()).await.unwrap();
        assert!(store.get::<String>("key1").await.unwrap().is_some());

        store.delete("key1").await.unwrap();
        assert!(store.get::<String>("key1").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_memory_kv_store_clear() {
        let store = MemoryKVStore::new();

        store.set("key1", &"value1".to_string()).await.unwrap();
        store.set("key2", &"value2".to_string()).await.unwrap();

        store.clear().await.unwrap();

        let keys = store.keys().await.unwrap();
        assert_eq!(keys.len(), 0);
    }

    #[tokio::test]
    async fn test_memory_kv_store_keys() {
        let store = MemoryKVStore::new();

        store.set("key1", &"value1".to_string()).await.unwrap();
        store.set("key2", &"value2".to_string()).await.unwrap();

        let keys = store.keys().await.unwrap();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains("key1"));
        assert!(keys.contains("key2"));
    }

    #[tokio::test]
    async fn test_memory_kv_store_get_all() {
        let store = MemoryKVStore::new();

        store.set("key1", &"value1".to_string()).await.unwrap();
        store.set("key2", &42i32).await.unwrap();

        let all = store.get_all().await.unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all.get("key1").unwrap(), "value1");
        assert_eq!(all.get("key2").unwrap(), &42);
    }

    #[tokio::test]
    async fn test_memory_kv_store_mailbox() {
        let store = MemoryKVStore::new();

        store.publish_to_mailbox("mailbox1", &serde_json::json!({"msg": "hello"})).await.unwrap();
        store.publish_to_mailbox("mailbox1", &serde_json::json!({"msg": "world"})).await.unwrap();

        let msg1 = store.get_message("mailbox1", None).await.unwrap();
        assert_eq!(msg1, Some(serde_json::json!({"msg": "hello"})));

        let msg2 = store.get_message("mailbox1", None).await.unwrap();
        assert_eq!(msg2, Some(serde_json::json!({"msg": "world"})));

        let msg3 = store.get_message("mailbox1", None).await.unwrap();
        assert_eq!(msg3, None);
    }

    #[tokio::test]
    async fn test_memory_kv_store_prefix_suffix() {
        let store = MemoryKVStore::new();

        // Set prefix
        {
            let mut prefix = store.prefix.write().unwrap();
            *prefix = "test".to_string();
        }

        store.set("key1", &"value1".to_string()).await.unwrap();

        // Key should be stored with prefix
        let all = store.get_all().await.unwrap();
        assert!(all.contains_key("test:key1"));
        assert!(!all.contains_key("key1"));

        // Should be able to retrieve with same prefix
        let result: Option<String> = store.get("key1").await.unwrap();
        assert_eq!(result, Some("value1".to_string()));
    }

    #[tokio::test]
    async fn test_memory_kv_store_apply_key_modifications() {
        let store = MemoryKVStore::new();

        // Set prefix and suffix
        {
            let mut prefix = store.prefix.write().unwrap();
            *prefix = "pre".to_string();
        }
        {
            let mut suffix = store.suffix.write().unwrap();
            *suffix = "suf".to_string();
        }

        let modified = store.apply_key_modifications("key").await;
        assert_eq!(modified, "pre:key:suf");
    }
}
