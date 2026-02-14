//! MCP client over WebSocket example
//!
//! Connects to the WebSocket MCP server, discovers tools, and calls them.
//!
//! Start the server first:
//! ```
//! cargo run --example websocket_server
//! ```
//!
//! Then run this client:
//! ```
//! cargo run --example client_websocket
//! ```

use reson_mcp::client::McpClient;
use serde_json::json;

#[tokio::main]
async fn main() -> reson_mcp::Result<()> {
    eprintln!("Connecting to MCP server at ws://127.0.0.1:9090 ...");
    let client = McpClient::websocket("ws://127.0.0.1:9090").await?;

    if let Some(info) = client.server_info() {
        eprintln!(
            "Connected to: {} v{}",
            info.server_info.name, info.server_info.version
        );
    }

    let tools = client.list_tools().await?;
    eprintln!("\nAvailable tools:");
    for tool in &tools.tools {
        eprintln!(
            "  - {} : {}",
            tool.name,
            tool.description.as_deref().unwrap_or("(no description)")
        );
    }

    eprintln!("\n--- Calling tools ---\n");

    let result = client.call_tool("add", json!({"a": 3, "b": 4})).await?;
    eprintln!("add(3, 4): {}", extract_text(&result));

    let result = client
        .call_tool("echo", json!({"message": "hello over websocket"}))
        .await?;
    eprintln!("echo: {}", extract_text(&result));

    let result = client
        .call_tool("reverse", json!({"text": "websocket"}))
        .await?;
    eprintln!("reverse: {}", extract_text(&result));

    eprintln!("\n--- Done ---");

    client.close().await?;
    Ok(())
}

fn extract_text(result: &rmcp::model::CallToolResult) -> &str {
    result
        .content
        .first()
        .and_then(|c| c.as_text())
        .map(|t| t.text.as_str())
        .unwrap_or("(no output)")
}
