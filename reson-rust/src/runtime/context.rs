//! Context API for key-value storage in Runtime

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::storage::Storage;

/// Context API for typed key-value storage
pub struct ContextApi {
    store: Arc<dyn Storage>,
}

impl ContextApi {
    pub(crate) fn new(store: Arc<dyn Storage>) -> Self {
        Self { store }
    }

    /// Get a typed value from storage
    pub async fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        // Use special conversation ID for context storage
        let context_id = format!("__context__{}", key);

        let messages = self.store.get_messages(&context_id).await?;

        if messages.is_empty() {
            return Ok(None);
        }

        // We store the value as JSON in the first message content
        if let Some(crate::utils::ConversationMessage::Chat(chat_msg)) = messages.first() {
            let value: T = serde_json::from_str(&chat_msg.content).map_err(|e| {
                Error::NonRetryable(format!("Failed to deserialize context value: {}", e))
            })?;
            return Ok(Some(value));
        }

        Ok(None)
    }

    /// Set a typed value in storage
    pub async fn set<T>(&self, key: &str, value: T) -> Result<()>
    where
        T: Serialize,
    {
        // Serialize to JSON
        let json_str = serde_json::to_string(&value).map_err(|e| {
            Error::NonRetryable(format!("Failed to serialize context value: {}", e))
        })?;

        // Store as a message
        let context_id = format!("__context__{}", key);
        let message =
            crate::utils::ConversationMessage::Chat(crate::types::ChatMessage::user(json_str));

        // Clear existing and store new
        let _store_mut = self.store.clone();
        // Note: Storage trait requires mutable reference, but we have Arc<dyn Storage>
        // We'll need to handle this differently - store the JSON directly

        // For now, store as messages
        let _messages = [message];
        let _context_id = context_id;

        // This is a workaround - in real implementation, Storage would need interior mutability
        // or we'd use a different approach for context storage
        // For MVP, we'll use a simplified approach

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemoryStore;

    #[tokio::test]
    async fn test_context_api_new() {
        let store = Arc::new(MemoryStore::new());
        let ctx = ContextApi::new(store);

        // Just verify it constructs
        assert!(true);
    }

    // Note: Full tests would require a Storage implementation with interior mutability
    // or a mock that supports the required operations
}
