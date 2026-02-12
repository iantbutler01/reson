//! Integration tests for MCP server

use rmcp::model::{CallToolResult, Content};
use reson_mcp::server::{McpServer, ServerTransport};
use serde_json::json;

fn build_test_server() -> McpServer {
    McpServer::builder("test-server")
        .with_version("1.0.0")
        .with_description("A test calculator server")
        .with_tool(
            "add",
            "Add two numbers",
            json!({
                "type": "object",
                "properties": {
                    "a": { "type": "number" },
                    "b": { "type": "number" }
                },
                "required": ["a", "b"]
            }),
            |_name, args| {
                Box::pin(async move {
                    let args = args.unwrap_or_default();
                    let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Ok(CallToolResult::success(vec![Content::text(
                        (a + b).to_string(),
                    )]))
                })
            },
        )
        .with_tool(
            "echo",
            "Echo a message",
            json!({
                "type": "object",
                "properties": {
                    "message": { "type": "string" }
                },
                "required": ["message"]
            }),
            |_name, args| {
                Box::pin(async move {
                    let args = args.unwrap_or_default();
                    let msg = args
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("no message");
                    Ok(CallToolResult::success(vec![Content::text(msg.to_string())]))
                })
            },
        )
        .build()
}

/// Start the test server on WebSocket and return the port
async fn start_test_server(port: u16) -> tokio::task::JoinHandle<()> {
    let server = build_test_server();
    let addr = format!("127.0.0.1:{}", port);
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr))
            .await
            .expect("Server failed");
    })
}

#[tokio::test]
async fn test_server_list_tools() {
    let _server = start_test_server(18101).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18101")
        .await
        .expect("Failed to connect");

    let tools = client.list_tools().await.expect("Failed to list tools");
    assert_eq!(tools.tools.len(), 2);

    let tool_names: Vec<_> = tools.tools.iter().map(|t| t.name.as_ref()).collect();
    assert!(tool_names.contains(&"add"));
    assert!(tool_names.contains(&"echo"));

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_server_call_add_tool() {
    let _server = start_test_server(18102).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18102")
        .await
        .expect("Failed to connect");

    let result = client
        .call_tool("add", json!({"a": 7, "b": 3}))
        .await
        .expect("Failed to call tool");

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .expect("Expected text content");
    assert_eq!(text.text, "10");

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_server_call_echo_tool() {
    let _server = start_test_server(18103).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18103")
        .await
        .expect("Failed to connect");

    let result = client
        .call_tool("echo", json!({"message": "hello world"}))
        .await
        .expect("Failed to call tool");

    assert!(!result.is_error.unwrap_or(false));
    let text = result
        .content
        .first()
        .and_then(|c| c.as_text())
        .expect("Expected text content");
    assert_eq!(text.text, "hello world");

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_server_info() {
    let _server = start_test_server(18104).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18104")
        .await
        .expect("Failed to connect");

    let info = client
        .server_info()
        .expect("Server info should be available");
    assert_eq!(info.server_info.name, "test-server");
    assert_eq!(info.server_info.version, "1.0.0");
    assert_eq!(
        info.instructions.as_deref(),
        Some("A test calculator server")
    );

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_server_multiple_clients() {
    let _server = start_test_server(18105).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Connect two clients simultaneously
    let client1 = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18105")
        .await
        .expect("Client 1 failed to connect");
    let client2 = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18105")
        .await
        .expect("Client 2 failed to connect");

    // Both should be able to list tools
    let tools1 = client1.list_tools().await.expect("Client 1 list failed");
    let tools2 = client2.list_tools().await.expect("Client 2 list failed");
    assert_eq!(tools1.tools.len(), 2);
    assert_eq!(tools2.tools.len(), 2);

    // Both should be able to call tools
    let r1 = client1
        .call_tool("add", json!({"a": 1, "b": 2}))
        .await
        .expect("Client 1 call failed");
    let r2 = client2
        .call_tool("add", json!({"a": 10, "b": 20}))
        .await
        .expect("Client 2 call failed");

    assert_eq!(r1.content.first().and_then(|c| c.as_text()).unwrap().text, "3");
    assert_eq!(r2.content.first().and_then(|c| c.as_text()).unwrap().text, "30");

    client1.close().await.expect("Failed to close client 1");
    client2.close().await.expect("Failed to close client 2");
}
