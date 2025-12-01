//! Example: Dynamic Tool Parsing
//!
//! This example demonstrates how Reson's NativeToolParser dynamically constructs
//! tool objects from JSON using a type registry, similar to Python's runtime type lookup.

use reson::parsers::{Deserializable, FieldDescription};
use reson::runtime::Runtime;
use serde::{Deserialize, Serialize};
use futures::future::BoxFuture;

/// Example tool: Chat message
#[derive(Debug, Serialize, Deserialize)]
struct Chat {
    #[serde(default)]
    recipient: String,
    #[serde(default)]
    message: String,
}

impl Deserializable for Chat {
    fn from_partial(partial: serde_json::Value) -> reson::error::Result<Self> {
        serde_json::from_value(partial).map_err(|e| {
            reson::error::Error::NonRetryable(format!("Failed to parse Chat: {}", e))
        })
    }

    fn validate_complete(&self) -> reson::error::Result<()> {
        if self.recipient.is_empty() {
            return Err(reson::error::Error::NonRetryable(
                "recipient is required".to_string(),
            ));
        }
        if self.message.is_empty() {
            return Err(reson::error::Error::NonRetryable(
                "message is required".to_string(),
            ));
        }
        Ok(())
    }

    fn field_descriptions() -> Vec<FieldDescription> {
        vec![
            FieldDescription {
                name: "recipient".to_string(),
                field_type: "string".to_string(),
                description: "Who to send the message to".to_string(),
                required: true,
            },
            FieldDescription {
                name: "message".to_string(),
                field_type: "string".to_string(),
                description: "The message content".to_string(),
                required: true,
            },
        ]
    }
}

/// Example tool: File operation
#[derive(Debug, Serialize, Deserialize)]
struct FileOp {
    #[serde(default)]
    path: String,
    #[serde(default)]
    operation: String,
}

impl Deserializable for FileOp {
    fn from_partial(partial: serde_json::Value) -> reson::error::Result<Self> {
        serde_json::from_value(partial).map_err(|e| {
            reson::error::Error::NonRetryable(format!("Failed to parse FileOp: {}", e))
        })
    }

    fn validate_complete(&self) -> reson::error::Result<()> {
        if self.path.is_empty() {
            return Err(reson::error::Error::NonRetryable(
                "path is required".to_string(),
            ));
        }
        Ok(())
    }

    fn field_descriptions() -> Vec<FieldDescription> {
        vec![
            FieldDescription {
                name: "path".to_string(),
                field_type: "string".to_string(),
                description: "File path".to_string(),
                required: true,
            },
            FieldDescription {
                name: "operation".to_string(),
                field_type: "string".to_string(),
                description: "Operation to perform".to_string(),
                required: false,
            },
        ]
    }
}

#[tokio::main]
async fn main() -> reson::error::Result<()> {
    println!("=== Dynamic Tool Parsing Example ===\n");

    // Create a runtime
    let runtime = Runtime::new();

    // Register tools with handlers
    // In Python: runtime.tool(handle_chat, name="Chat", tool_type=Chat)
    // In Rust:   runtime.tool::<Chat, _>(handle_chat, Some("Chat"))

    let chat_handler = |parsed_tool: reson::parsers::ParsedTool| -> BoxFuture<'static, reson::error::Result<String>> {
        Box::pin(async move {
            let chat: Chat = serde_json::from_value(parsed_tool.value)?;
            println!(
                "📨 Chat handler called: {} -> {}",
                chat.recipient, chat.message
            );
            Ok(format!("Sent message to {}", chat.recipient))
        })
    };

    runtime.tool::<Chat, _>(chat_handler, Some("Chat")).await?;

    let fileop_handler = |parsed_tool: reson::parsers::ParsedTool| -> BoxFuture<'static, reson::error::Result<String>> {
        Box::pin(async move {
            let file_op: FileOp = serde_json::from_value(parsed_tool.value)?;
            println!("📁 FileOp handler called: {} ({})", file_op.path, file_op.operation);
            Ok(format!("Performed {} on {}", file_op.operation, file_op.path))
        })
    };

    runtime.tool::<FileOp, _>(fileop_handler, Some("FileOp")).await?;

    println!("✓ Registered 2 tools: Chat, FileOp\n");

    // Get the parser (this would be used during streaming)
    let parser = runtime.get_parser().await;

    // Simulate streaming tool calls
    println!("=== Simulating Streaming Tool Calls ===\n");

    // Simulate a Chat tool call
    let json1 = r#"{"recipient":"Alice","message":"Hello!"}"#;
    let result1 = parser.parse_tool("Chat", json1, "call_123");

    if let Some(parsed) = result1.value {
        println!("✓ Parsed tool call:");
        println!("  - tool_name: {}", parsed.tool_name);
        println!("  - tool_use_id: {}", parsed.tool_use_id);
        println!("  - value: {:?}", parsed.value);
        println!();
    }

    // Simulate a FileOp tool call
    let json2 = r#"{"path":"/tmp/file.txt","operation":"read"}"#;
    let result2 = parser.parse_tool("FileOp", json2, "call_456");

    if let Some(parsed) = result2.value {
        println!("✓ Parsed tool call:");
        println!("  - tool_name: {}", parsed.tool_name);
        println!("  - tool_use_id: {}", parsed.tool_use_id);
        println!("  - value: {:?}", parsed.value);
        println!();
    }

    // Simulate incomplete JSON (streaming in progress)
    println!("=== Partial JSON (streaming) ===\n");
    let incomplete_json = r#"{"recipient":"Bob","mess"#;
    let result3 = parser.parse_tool("Chat", incomplete_json, "call_789");

    println!("✓ Handled incomplete JSON:");
    println!("  - is_partial: {}", result3.is_partial);
    println!("  - has_value: {}", result3.value.is_some());
    if let Some(parsed) = result3.value {
        println!("  - Fallback to defaults: {:?}", parsed.value);
    }
    println!();

    // Try an unregistered tool
    println!("=== Unregistered Tool ===\n");
    let result4 = parser.parse_tool("UnknownTool", "{}", "call_999");

    if let Some(error) = result4.error {
        println!("✗ Error (expected): {:?}", error);
    }

    println!("\n=== Key Concepts Demonstrated ===");
    println!("1. Tools are registered with runtime.tool::<T>(handler, name)");
    println!("2. NativeToolParser dynamically constructs typed objects from JSON");
    println!("3. ParsedTool wraps the value with metadata (tool_name, tool_use_id)");
    println!("4. Partial JSON is handled gracefully during streaming");
    println!("5. Type safety maintained despite dynamic construction");

    Ok(())
}
