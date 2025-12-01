//! Example demonstrating Store trait usage with MemoryKVStore
//!
//! Run with: cargo run --example store_usage

use reson::storage::{Store, MemoryKVStore};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Reson Store Example ===\n");

    // Create an in-memory store
    let store = MemoryKVStore::new();

    // Basic key-value operations
    println!("1. Basic set/get:");
    store.set("user:name", &"Alice".to_string()).await?;
    store.set("user:age", &30u32).await?;

    let name: Option<String> = store.get("user:name").await?;
    let age: Option<u32> = store.get("user:age").await?;
    println!("   Name: {:?}, Age: {:?}", name, age);

    // List all keys
    println!("\n2. List all keys:");
    let keys = store.keys().await?;
    println!("   Keys: {:?}", keys);

    // Get all key-value pairs
    println!("\n3. Get all data:");
    let all_data = store.get_all().await?;
    println!("   All data: {}", serde_json::to_string_pretty(&all_data)?);

    // Mailbox pub/sub
    println!("\n4. Mailbox pub/sub:");
    store.publish_to_mailbox("notifications", &serde_json::json!({
        "type": "email",
        "message": "Welcome!"
    })).await?;

    store.publish_to_mailbox("notifications", &serde_json::json!({
        "type": "sms",
        "message": "Verification code: 123456"
    })).await?;

    let msg1 = store.get_message("notifications", None).await?;
    println!("   Message 1: {}", serde_json::to_string_pretty(&msg1)?);

    let msg2 = store.get_message("notifications", None).await?;
    println!("   Message 2: {}", serde_json::to_string_pretty(&msg2)?);

    let msg3 = store.get_message("notifications", None).await?;
    println!("   Message 3 (should be None): {:?}", msg3);

    // Prefix/suffix namespacing
    println!("\n5. Prefix namespacing:");
    {
        let mut prefix = store.prefix.write().unwrap();
        *prefix = "app".to_string();
    }

    store.set("config:version", &"1.0.0".to_string()).await?;
    let version: Option<String> = store.get("config:version").await?;
    println!("   Version (with prefix): {:?}", version);

    // Show actual key stored
    let all_keys = store.keys().await?;
    println!("   Actual keys in store: {:?}", all_keys);

    // Delete operation
    println!("\n6. Delete operation:");
    store.delete("user:age").await?;
    let age_after_delete: Option<u32> = store.get("user:age").await?;
    println!("   Age after delete: {:?}", age_after_delete);

    // Clear all (respects prefix)
    println!("\n7. Clear operation:");
    store.clear().await?;
    let keys_after_clear = store.keys().await?;
    println!("   Keys after clear: {:?}", keys_after_clear);

    println!("\n✅ Store example completed successfully!");

    Ok(())
}
