//! Redis-backed Store implementation
//!
//! Provides persistent key-value storage with pub/sub mailbox support using Redis.

use async_trait::async_trait;
use redis::{AsyncCommands, Client};
use serde::{de::DeserializeOwned, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::storage::Store;

/// Redis-backed Store
pub struct RedisStore {
    client: Client,
    pub prefix: Arc<RwLock<String>>,
    pub suffix: Arc<RwLock<String>>,
}

impl RedisStore {
    /// Create a new RedisStore with connection URL
    pub fn new(url: &str) -> Result<Self> {
        let client = Client::open(url)
            .map_err(|e| Error::NonRetryable(format!("Failed to create Redis client: {}", e)))?;

        Ok(Self {
            client,
            prefix: Arc::new(RwLock::new(String::new())),
            suffix: Arc::new(RwLock::new(String::new())),
        })
    }

    /// Get async connection to Redis
    async fn get_connection(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| Error::NonRetryable(format!("Failed to get Redis connection: {}", e)))
    }
}

#[async_trait]
impl Store for RedisStore {
    async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let modified_key = self.apply_key_modifications(key).await;
        let mut conn = self.get_connection().await?;

        let value: Option<String> = conn
            .get(&modified_key)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis GET error: {}", e)))?;

        if let Some(json_str) = value {
            let deserialized = serde_json::from_str(&json_str)
                .map_err(|e| Error::NonRetryable(format!("Deserialization error: {}", e)))?;
            Ok(Some(deserialized))
        } else {
            Ok(None)
        }
    }

    async fn set<T: Serialize + Send + Sync>(&self, key: &str, value: &T) -> Result<()> {
        let modified_key = self.apply_key_modifications(key).await;
        let json_str = serde_json::to_string(value)
            .map_err(|e| Error::NonRetryable(format!("Serialization error: {}", e)))?;

        let mut conn = self.get_connection().await?;

        conn.set::<_, _, ()>(&modified_key, json_str)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis SET error: {}", e)))?;

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let modified_key = self.apply_key_modifications(key).await;
        let mut conn = self.get_connection().await?;

        conn.del::<_, ()>(&modified_key)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis DEL error: {}", e)))?;

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let prefix = self.get_prefix().await;
        let suffix = self.get_suffix().await;
        let mut conn = self.get_connection().await?;

        if prefix.is_empty() && suffix.is_empty() {
            // Clear all keys (dangerous!)
            redis::cmd("FLUSHDB")
                .query_async::<()>(&mut conn)
                .await
                .map_err(|e| Error::NonRetryable(format!("Redis FLUSHDB error: {}", e)))?;
        } else {
            // Build pattern for keys to delete
            let sep = self.get_affix_sep();
            let pattern = if !prefix.is_empty() && !suffix.is_empty() {
                format!("{}{}*{}{}", prefix, sep, sep, suffix)
            } else if !prefix.is_empty() {
                format!("{}{}*", prefix, sep)
            } else {
                format!("*{}{}", sep, suffix)
            };

            // Use SCAN to find and delete keys
            let keys: Vec<String> = redis::cmd("KEYS")
                .arg(&pattern)
                .query_async(&mut conn)
                .await
                .map_err(|e| Error::NonRetryable(format!("Redis KEYS error: {}", e)))?;

            if !keys.is_empty() {
                conn.del::<_, ()>(keys)
                    .await
                    .map_err(|e| Error::NonRetryable(format!("Redis DEL error: {}", e)))?;
            }
        }

        Ok(())
    }

    async fn get_all(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut conn = self.get_connection().await?;

        // Get all keys
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg("*")
            .query_async(&mut conn)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis KEYS error: {}", e)))?;

        let mut result = HashMap::new();

        for key in keys {
            let value: Option<String> = conn
                .get(&key)
                .await
                .map_err(|e| Error::NonRetryable(format!("Redis GET error: {}", e)))?;

            if let Some(json_str) = value {
                if let Ok(json_value) = serde_json::from_str(&json_str) {
                    result.insert(key, json_value);
                }
            }
        }

        Ok(result)
    }

    async fn keys(&self) -> Result<HashSet<String>> {
        let mut conn = self.get_connection().await?;

        let keys: Vec<String> = redis::cmd("KEYS")
            .arg("*")
            .query_async(&mut conn)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis KEYS error: {}", e)))?;

        Ok(keys.into_iter().collect())
    }

    async fn publish_to_mailbox(&self, mailbox_id: &str, value: &serde_json::Value) -> Result<()> {
        let modified_mailbox_id = self.apply_key_modifications(mailbox_id).await;
        let json_str = serde_json::to_string(value)
            .map_err(|e| Error::NonRetryable(format!("Serialization error: {}", e)))?;

        let mut conn = self.get_connection().await?;

        // Use RPUSH to append to list (mailbox)
        conn.rpush::<_, _, ()>(&modified_mailbox_id, json_str)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis RPUSH error: {}", e)))?;

        Ok(())
    }

    async fn get_message(
        &self,
        mailbox_id: &str,
        timeout_secs: Option<f64>,
    ) -> Result<Option<serde_json::Value>> {
        let modified_mailbox_id = self.apply_key_modifications(mailbox_id).await;
        let mut conn = self.get_connection().await?;

        let timeout = timeout_secs.unwrap_or(0.0);

        // Use BLPOP to block and pop from list (timeout in seconds as f64 for redis crate)
        let result: Option<(String, String)> = conn
            .blpop(&modified_mailbox_id, timeout)
            .await
            .map_err(|e| Error::NonRetryable(format!("Redis BLPOP error: {}", e)))?;

        if let Some((_key, json_str)) = result {
            let value = serde_json::from_str(&json_str)
                .map_err(|e| Error::NonRetryable(format!("Deserialization error: {}", e)))?;
            Ok(Some(value))
        } else {
            Ok(None)
        }
    }

    async fn close(&self) -> Result<()> {
        // Redis client handles connection cleanup automatically
        Ok(())
    }

    async fn get_prefix(&self) -> String {
        self.prefix.read().unwrap().clone()
    }

    async fn get_suffix(&self) -> String {
        self.suffix.read().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require a running Redis instance
    // Run with: docker run -d -p 6379:6379 redis:latest

    async fn get_test_store() -> RedisStore {
        RedisStore::new("redis://127.0.0.1:6379").expect("Failed to create Redis store")
    }

    #[tokio::test]
    #[ignore] // Requires Redis
    async fn test_redis_store_set_and_get() {
        let store = get_test_store().await;

        store
            .set("test_key", &"test_value".to_string())
            .await
            .unwrap();

        let result: Option<String> = store.get("test_key").await.unwrap();
        assert_eq!(result, Some("test_value".to_string()));

        // Cleanup
        store.delete("test_key").await.unwrap();
    }

    #[tokio::test]
    #[ignore] // Requires Redis
    async fn test_redis_store_mailbox() {
        let store = get_test_store().await;

        store
            .publish_to_mailbox("test_mailbox", &serde_json::json!({"msg": "hello"}))
            .await
            .unwrap();

        let msg = store.get_message("test_mailbox", Some(1.0)).await.unwrap();
        assert_eq!(msg, Some(serde_json::json!({"msg": "hello"})));

        // Timeout should return None
        let msg2 = store.get_message("test_mailbox", Some(0.1)).await.unwrap();
        assert_eq!(msg2, None);
    }

    #[tokio::test]
    #[ignore] // Requires Redis
    async fn test_redis_store_prefix() {
        let store = get_test_store().await;

        {
            let mut prefix = store.prefix.write().unwrap();
            *prefix = "testprefix".to_string();
        }

        store.set("key1", &"value1".to_string()).await.unwrap();

        let result: Option<String> = store.get("key1").await.unwrap();
        assert_eq!(result, Some("value1".to_string()));

        // Cleanup
        store.delete("key1").await.unwrap();
    }
}
