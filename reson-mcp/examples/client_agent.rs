//! MCP client "agent" example
//!
//! Connects to the HTTP MCP server, discovers available tools,
//! and calls each one.
//!
//! Start the server first:
//! ```
//! cargo run --example http_server
//! ```
//!
//! Then run this client:
//! ```
//! cargo run --example client_agent
//! ```

use reson_mcp::client::McpClient;
use serde_json::json;

#[tokio::main]
async fn main() -> reson_mcp::Result<()> {
    // Connect to the HTTP MCP server
    eprintln!("Connecting to MCP server at http://127.0.0.1:8080/mcp ...");
    let client = McpClient::http("http://127.0.0.1:8080/mcp").await?;

    if let Some(info) = client.server_info() {
        eprintln!(
            "Connected to: {} v{}",
            info.server_info.name, info.server_info.version
        );
    }

    // Discover tools
    let tools = client.list_tools().await?;
    eprintln!("\nAvailable tools:");
    for tool in &tools.tools {
        eprintln!(
            "  - {} : {}",
            tool.name,
            tool.description.as_deref().unwrap_or("(no description)")
        );
    }

    // Use the tools like a tiny agent would
    eprintln!("\n--- Agent run ---\n");

    // Step 1: Add some numbers
    let result = client.call_tool("add", json!({"a": 17, "b": 25})).await?;
    let sum = extract_text(&result);
    eprintln!("Step 1 - add(17, 25): {}", sum);

    // Step 2: Echo the result
    let result = client.call_tool("echo", json!({"message": sum})).await?;
    eprintln!("Step 2 - echo: {}", extract_text(&result));

    // Step 3: Reverse it
    let result = client.call_tool("reverse", json!({"text": sum})).await?;
    eprintln!("Step 3 - reverse: {}", extract_text(&result));

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
