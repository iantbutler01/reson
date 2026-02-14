//! WebSocket MCP server example
//!
//! A basic MCP server with a few tools, served over WebSocket.
//!
//! ```
//! cargo run --example websocket_server
//! ```
//!
//! Then connect with: ws://127.0.0.1:9090

use rmcp::model::{CallToolResult, Content};
use reson_mcp::server::{McpServer, ServerTransport};
use serde_json::json;

#[tokio::main]
async fn main() -> reson_mcp::Result<()> {
    let server = McpServer::builder("reson-ws-example")
        .with_version("0.1.0")
        .with_description("An example MCP server over WebSocket built with reson-mcp")
        .with_tool(
            "add",
            "Add two numbers together",
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number", "description": "First number" },
                    "b": { "type": "number", "description": "Second number" }
                },
                "required": ["a", "b"]
            }),
            |_name, args| {
                Box::pin(async move {
                    let args = args.unwrap_or_default();
                    let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let result = a + b;
                    eprintln!("[add] {} + {} = {}", a, b, result);
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "{} + {} = {}",
                        a, b, result
                    ))]))
                })
            },
        )
        .with_tool(
            "echo",
            "Echo back a message",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string", "description": "The message to echo" }
                },
                "required": ["message"]
            }),
            |_name, args| {
                Box::pin(async move {
                    let args = args.unwrap_or_default();
                    let msg = args
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(empty)");
                    eprintln!("[echo] {}", msg);
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "Echo: {}",
                        msg
                    ))]))
                })
            },
        )
        .with_tool(
            "reverse",
            "Reverse a string",
            json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "The text to reverse" }
                },
                "required": ["text"]
            }),
            |_name, args| {
                Box::pin(async move {
                    let args = args.unwrap_or_default();
                    let text = args
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let reversed: String = text.chars().rev().collect();
                    eprintln!("[reverse] \"{}\" -> \"{}\"", text, reversed);
                    Ok(CallToolResult::success(vec![Content::text(reversed)]))
                })
            },
        )
        .build();

    eprintln!("Starting MCP WebSocket server on ws://127.0.0.1:9090");
    server
        .serve(ServerTransport::WebSocket("127.0.0.1:9090".into()))
        .await?;

    Ok(())
}
