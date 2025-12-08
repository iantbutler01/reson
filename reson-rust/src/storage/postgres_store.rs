//! PostgreSQL-backed Store implementation
//!
//! Provides persistent key-value storage using PostgreSQL JSONB column.

use async_trait::async_trait;
use serde::{de::DeserializeOwned, Serialize};
use sqlx::{PgPool, Row};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::error::{Error, Result};
use crate::storage::Store;

/// PostgreSQL-backed Store
pub struct PostgresStore {
    pool: PgPool,
    table_name: String,
    pub prefix: Arc<RwLock<String>>,
    pub suffix: Arc<RwLock<String>>,
}

impl PostgresStore {
    /// Create a new PostgresStore with connection URL
    pub async fn new(url: &str, table_name: Option<&str>) -> Result<Self> {
        let pool = PgPool::connect(url)
            .await
            .map_err(|e| Error::NonRetryable(format!("Failed to connect to PostgreSQL: {}", e)))?;

        let table_name = table_name.unwrap_or("reson_store").to_string();

        let store = Self {
            pool,
            table_name,
            prefix: Arc::new(RwLock::new(String::new())),
            suffix: Arc::new(RwLock::new(String::new())),
        };

        // Initialize table
        store.init_table().await?;

        Ok(store)
    }

    /// Initialize the storage table
    async fn init_table(&self) -> Result<()> {
        let create_table_sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                key TEXT PRIMARY KEY,
                value JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            "#,
            self.table_name
        );

        sqlx::query(&create_table_sql)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("Failed to create table: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl Store for PostgresStore {
    async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>> {
        let modified_key = self.apply_key_modifications(key).await;

        let query = format!("SELECT value FROM {} WHERE key = $1", self.table_name);

        let row = sqlx::query(&query)
            .bind(&modified_key)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL query error: {}", e)))?;

        if let Some(row) = row {
            let value: serde_json::Value = row.get("value");
            let deserialized = serde_json::from_value(value)
                .map_err(|e| Error::NonRetryable(format!("Deserialization error: {}", e)))?;
            Ok(Some(deserialized))
        } else {
            Ok(None)
        }
    }

    async fn set<T: Serialize + Send + Sync>(&self, key: &str, value: &T) -> Result<()> {
        let modified_key = self.apply_key_modifications(key).await;
        let json_value = serde_json::to_value(value)
            .map_err(|e| Error::NonRetryable(format!("Serialization error: {}", e)))?;

        let query = format!(
            r#"
            INSERT INTO {} (key, value, updated_at)
            VALUES ($1, $2, CURRENT_TIMESTAMP)
            ON CONFLICT (key)
            DO UPDATE SET value = $2, updated_at = CURRENT_TIMESTAMP
            "#,
            self.table_name
        );

        sqlx::query(&query)
            .bind(&modified_key)
            .bind(&json_value)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL insert error: {}", e)))?;

        Ok(())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        let modified_key = self.apply_key_modifications(key).await;

        let query = format!("DELETE FROM {} WHERE key = $1", self.table_name);

        sqlx::query(&query)
            .bind(&modified_key)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL delete error: {}", e)))?;

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let prefix = self.get_prefix().await;
        let suffix = self.get_suffix().await;

        let query = if prefix.is_empty() && suffix.is_empty() {
            // Clear all keys
            format!("DELETE FROM {}", self.table_name)
        } else {
            // Clear only keys matching prefix/suffix
            let sep = self.get_affix_sep();
            let mut conditions = Vec::new();

            if !prefix.is_empty() {
                conditions.push(format!("key LIKE '{}{}%'", prefix, sep));
            }
            if !suffix.is_empty() {
                conditions.push(format!("key LIKE '%{}{}'", sep, suffix));
            }

            format!(
                "DELETE FROM {} WHERE {}",
                self.table_name,
                conditions.join(" AND ")
            )
        };

        sqlx::query(&query)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL clear error: {}", e)))?;

        Ok(())
    }

    async fn get_all(&self) -> Result<HashMap<String, serde_json::Value>> {
        let query = format!("SELECT key, value FROM {}", self.table_name);

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL query error: {}", e)))?;

        let mut result = HashMap::new();
        for row in rows {
            let key: String = row.get("key");
            let value: serde_json::Value = row.get("value");
            result.insert(key, value);
        }

        Ok(result)
    }

    async fn keys(&self) -> Result<HashSet<String>> {
        let query = format!("SELECT key FROM {}", self.table_name);

        let rows = sqlx::query(&query)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL query error: {}", e)))?;

        let keys = rows.into_iter().map(|row| row.get("key")).collect();
        Ok(keys)
    }

    async fn publish_to_mailbox(&self, mailbox_id: &str, value: &serde_json::Value) -> Result<()> {
        // For PostgreSQL, we implement mailbox as an array in a JSONB column
        let modified_mailbox_id = self.apply_key_modifications(mailbox_id).await;

        let query = format!(
            r#"
            INSERT INTO {} (key, value, updated_at)
            VALUES ($1, jsonb_build_array($2), CURRENT_TIMESTAMP)
            ON CONFLICT (key)
            DO UPDATE SET
                value = {}.value || jsonb_build_array($2),
                updated_at = CURRENT_TIMESTAMP
            "#,
            self.table_name, self.table_name
        );

        sqlx::query(&query)
            .bind(&modified_mailbox_id)
            .bind(value)
            .execute(&self.pool)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL mailbox publish error: {}", e)))?;

        Ok(())
    }

    async fn get_message(
        &self,
        mailbox_id: &str,
        _timeout_secs: Option<f64>,
    ) -> Result<Option<serde_json::Value>> {
        // Note: PostgreSQL doesn't have blocking pop like Redis, so we implement non-blocking
        let modified_mailbox_id = self.apply_key_modifications(mailbox_id).await;

        // Begin transaction to atomically get and remove first message
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL transaction error: {}", e)))?;

        let query = format!(
            "SELECT value FROM {} WHERE key = $1 FOR UPDATE",
            self.table_name
        );

        let row = sqlx::query(&query)
            .bind(&modified_mailbox_id)
            .fetch_optional(&mut *tx)
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL query error: {}", e)))?;

        if let Some(row) = row {
            let value: serde_json::Value = row.get("value");

            if let Some(arr) = value.as_array() {
                if !arr.is_empty() {
                    let first_msg = arr[0].clone();
                    let remaining: Vec<_> = arr.iter().skip(1).cloned().collect();

                    if remaining.is_empty() {
                        // Delete the mailbox if empty
                        let delete_query =
                            format!("DELETE FROM {} WHERE key = $1", self.table_name);
                        sqlx::query(&delete_query)
                            .bind(&modified_mailbox_id)
                            .execute(&mut *tx)
                            .await
                            .map_err(|e| {
                                Error::NonRetryable(format!("PostgreSQL delete error: {}", e))
                            })?;
                    } else {
                        // Update with remaining messages
                        let update_query = format!(
                            "UPDATE {} SET value = $1, updated_at = CURRENT_TIMESTAMP WHERE key = $2",
                            self.table_name
                        );
                        sqlx::query(&update_query)
                            .bind(serde_json::Value::Array(remaining))
                            .bind(&modified_mailbox_id)
                            .execute(&mut *tx)
                            .await
                            .map_err(|e| {
                                Error::NonRetryable(format!("PostgreSQL update error: {}", e))
                            })?;
                    }

                    tx.commit().await.map_err(|e| {
                        Error::NonRetryable(format!("PostgreSQL commit error: {}", e))
                    })?;

                    return Ok(Some(first_msg));
                }
            }
        }

        tx.commit()
            .await
            .map_err(|e| Error::NonRetryable(format!("PostgreSQL commit error: {}", e)))?;

        Ok(None)
    }

    async fn close(&self) -> Result<()> {
        self.pool.close().await;
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

    // Note: These tests require a running PostgreSQL instance
    // Run with: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:latest

    async fn get_test_store() -> PostgresStore {
        let url = "postgresql://postgres:postgres@localhost:5432/postgres";
        PostgresStore::new(url, Some("test_reson_store"))
            .await
            .expect("Failed to create PostgreSQL store")
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_postgres_store_set_and_get() {
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
    #[ignore] // Requires PostgreSQL
    async fn test_postgres_store_mailbox() {
        let store = get_test_store().await;

        store
            .publish_to_mailbox("test_mailbox", &serde_json::json!({"msg": "hello"}))
            .await
            .unwrap();
        store
            .publish_to_mailbox("test_mailbox", &serde_json::json!({"msg": "world"}))
            .await
            .unwrap();

        let msg1 = store.get_message("test_mailbox", None).await.unwrap();
        assert_eq!(msg1, Some(serde_json::json!({"msg": "hello"})));

        let msg2 = store.get_message("test_mailbox", None).await.unwrap();
        assert_eq!(msg2, Some(serde_json::json!({"msg": "world"})));

        let msg3 = store.get_message("test_mailbox", None).await.unwrap();
        assert_eq!(msg3, None);
    }

    #[tokio::test]
    #[ignore] // Requires PostgreSQL
    async fn test_postgres_store_clear() {
        let store = get_test_store().await;

        store.set("key1", &"value1".to_string()).await.unwrap();
        store.set("key2", &"value2".to_string()).await.unwrap();

        store.clear().await.unwrap();

        let keys = store.keys().await.unwrap();
        assert_eq!(keys.len(), 0);
    }
}
