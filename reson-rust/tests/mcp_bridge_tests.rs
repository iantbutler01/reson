//! Integration tests for MCP bridge (reson-agentic <-> reson-mcp)
//!
//! Tests use in-process WebSocket MCP servers to avoid external dependencies.

#![cfg(feature = "mcp")]

use reson_agentic::agentic;
use reson_agentic::mcp::{McpServer, ServerTransport};
use reson_agentic::runtime::Runtime;
use reson_agentic::types::{ChatMessage, ToolCall, ToolResult};
use reson_agentic::utils::ConversationMessage;
use reson_mcp::{CallToolResult, Content};
use serde_json::json;
use std::env;

/// Start a test MCP server on the given WebSocket port
fn build_calculator_server() -> reson_mcp::server::McpServer {
    reson_mcp::server::McpServer::builder("test-calculator")
        .with_description("A test calculator for integration tests")
        .with_tool(
            "add",
            "Add two numbers",
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
                    Ok(CallToolResult::success(vec![Content::text(
                        (a + b).to_string(),
                    )]))
                })
            },
        )
        .with_tool(
            "multiply",
            "Multiply two numbers",
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
                    Ok(CallToolResult::success(vec![Content::text(
                        (a * b).to_string(),
                    )]))
                })
            },
        )
        .build()
}

async fn start_ws_server(port: u16) -> tokio::task::JoinHandle<()> {
    let server = build_calculator_server();
    let addr = format!("127.0.0.1:{}", port);
    tokio::spawn(async move {
        server
            .serve(reson_mcp::server::ServerTransport::WebSocket(addr))
            .await
            .expect("Server failed");
    })
}

// --- Client bridge tests ---

#[tokio::test]
async fn test_mcp_client_registers_tools() {
    let _server = start_ws_server(19201).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime
        .mcp("ws://127.0.0.1:19201")
        .await
        .expect("Failed to connect to MCP server");

    let schemas = runtime.get_tool_schemas().await;
    assert_eq!(schemas.len(), 2);
    assert!(schemas.contains_key("add"));
    assert!(schemas.contains_key("multiply"));

    // Verify schema fields were extracted
    let add_schema = &schemas["add"];
    assert_eq!(add_schema.description, "Add two numbers");
    assert_eq!(add_schema.fields.len(), 2);
    let field_names: Vec<&str> = add_schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(field_names.contains(&"a"));
    assert!(field_names.contains(&"b"));
}

#[tokio::test]
async fn test_mcp_client_tool_execution() {
    let _server = start_ws_server(19202).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime
        .mcp("ws://127.0.0.1:19202")
        .await
        .expect("Failed to connect to MCP server");

    // Execute tool through the runtime
    let tool_call = json!({
        "_tool_name": "add",
        "a": 7,
        "b": 3
    });
    let result = runtime.execute_tool(&tool_call).await.expect("Tool execution failed");
    assert_eq!(result, "10");

    let tool_call = json!({
        "_tool_name": "multiply",
        "a": 6,
        "b": 4
    });
    let result = runtime.execute_tool(&tool_call).await.expect("Tool execution failed");
    assert_eq!(result, "24");
}

#[tokio::test]
async fn test_mcp_client_mixed_with_local_tools() {
    let _server = start_ws_server(19203).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();

    // Register a local tool
    runtime
        .register_tool(
            "local_echo",
            reson_agentic::runtime::ToolFunction::Sync(Box::new(|args| {
                let msg = args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("no message");
                Ok(format!("echo: {}", msg))
            })),
            None,
        )
        .await
        .unwrap();

    // Connect MCP server
    runtime
        .mcp("ws://127.0.0.1:19203")
        .await
        .expect("Failed to connect to MCP server");

    // Both local and MCP tools should work
    let local_result = runtime
        .execute_tool(&json!({"_tool_name": "local_echo", "message": "hello"}))
        .await
        .unwrap();
    assert_eq!(local_result, "echo: hello");

    let mcp_result = runtime
        .execute_tool(&json!({"_tool_name": "add", "a": 1, "b": 2}))
        .await
        .unwrap();
    assert_eq!(mcp_result, "3");
}

// --- Server wrapper tests ---

#[tokio::test]
async fn test_agent_server_single() {
    // Build server using our McpServer wrapper with .agent()
    let server = McpServer::new("agent-server")
        .description("Test agent server")
        .agent(
            "greet",
            "Greet someone",
            json!({
                "type": "object",
                "properties": {
                    "name": { "type": "string", "description": "Name to greet" }
                },
                "required": ["name"]
            }),
            |args| {
                Box::pin(async move {
                    let name = args
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("world");
                    Ok(format!("Hello, {}!", name))
                })
            },
        );

    let addr = "127.0.0.1:19204";
    tokio::spawn(async move {
        server.serve(ServerTransport::WebSocket(addr.into())).await.unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Connect with raw MCP client and verify
    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:19204")
        .await
        .expect("Failed to connect");

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.tools.len(), 1);
    assert_eq!(tools.tools[0].name.as_ref(), "greet");

    let result = client
        .call_tool("greet", json!({"name": "Alice"}))
        .await
        .unwrap();
    let text = result.content.first().and_then(|c| c.as_text()).unwrap();
    assert_eq!(text.text, "Hello, Alice!");

    client.close().await.unwrap();
}

#[tokio::test]
async fn test_agent_server_with_raw_tool() {
    // Mix .agent() and .tool() on the same server
    let server = McpServer::new("mixed-server")
        .agent(
            "summarize",
            "Summarize text",
            json!({
                "type": "object",
                "properties": {
                    "text": { "type": "string" }
                },
                "required": ["text"]
            }),
            |args| {
                Box::pin(async move {
                    let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    Ok(format!("Summary: {} chars", text.len()))
                })
            },
        )
        .tool(
            "ping",
            "Ping the server",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("pong")]))
                })
            },
        );

    let addr = "127.0.0.1:19205";
    tokio::spawn(async move {
        server.serve(ServerTransport::WebSocket(addr.into())).await.unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:19205")
        .await
        .expect("Failed to connect");

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.tools.len(), 2);

    // Agent-wrapped tool returns text
    let result = client
        .call_tool("summarize", json!({"text": "hello world"}))
        .await
        .unwrap();
    let text = result.content.first().and_then(|c| c.as_text()).unwrap();
    assert_eq!(text.text, "Summary: 11 chars");

    // Raw tool returns CallToolResult directly
    let result = client.call_tool("ping", json!({})).await.unwrap();
    let text = result.content.first().and_then(|c| c.as_text()).unwrap();
    assert_eq!(text.text, "pong");

    client.close().await.unwrap();
}

// --- End-to-end: server wrapper -> client bridge -> runtime ---

#[tokio::test]
async fn test_end_to_end_agent_server_to_runtime() {
    // Serve an agent over MCP
    let server = McpServer::new("e2e-server")
        .agent(
            "reverse",
            "Reverse a string",
            json!({
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "String to reverse" }
                },
                "required": ["input"]
            }),
            |args| {
                Box::pin(async move {
                    let input = args.get("input").and_then(|v| v.as_str()).unwrap_or("");
                    Ok(input.chars().rev().collect::<String>())
                })
            },
        );

    let addr = "127.0.0.1:19206";
    tokio::spawn(async move {
        server.serve(ServerTransport::WebSocket(addr.into())).await.unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Connect from a runtime
    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19206").await.unwrap();

    // Execute through runtime
    let result = runtime
        .execute_tool(&json!({"_tool_name": "reverse", "input": "hello"}))
        .await
        .unwrap();
    assert_eq!(result, "olleh");
}

// --- Namespace / label tests ---

#[tokio::test]
async fn test_mcp_as_prefixes_tool_names() {
    let _server = start_ws_server(19207).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime
        .mcp_as("ws://127.0.0.1:19207", "calc")
        .await
        .expect("Failed to connect");

    let schemas = runtime.get_tool_schemas().await;
    assert!(schemas.contains_key("calc_add"));
    assert!(schemas.contains_key("calc_multiply"));
    assert!(!schemas.contains_key("add"));
    assert!(!schemas.contains_key("multiply"));

    // Prefixed name works for execution (calls remote "add" under the hood)
    let result = runtime
        .execute_tool(&json!({"_tool_name": "calc_add", "a": 10, "b": 5}))
        .await
        .unwrap();
    assert_eq!(result, "15");
}

#[tokio::test]
async fn test_mcp_as_avoids_conflicts() {
    // Start two servers on different ports with overlapping tool names
    let _server1 = start_ws_server(19208).await;
    let _server2 = start_ws_server(19209).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp_as("ws://127.0.0.1:19208", "s1").await.unwrap();
    runtime.mcp_as("ws://127.0.0.1:19209", "s2").await.unwrap();

    let schemas = runtime.get_tool_schemas().await;
    assert_eq!(schemas.len(), 4); // s1_add, s1_multiply, s2_add, s2_multiply

    // Both work independently
    let r1 = runtime
        .execute_tool(&json!({"_tool_name": "s1_add", "a": 1, "b": 2}))
        .await
        .unwrap();
    let r2 = runtime
        .execute_tool(&json!({"_tool_name": "s2_add", "a": 10, "b": 20}))
        .await
        .unwrap();
    assert_eq!(r1, "3");
    assert_eq!(r2, "30");
}

// --- UI Apps tests (require mcp-apps feature) ---

#[cfg(feature = "mcp-apps")]
#[tokio::test]
async fn test_server_with_ui_resource() {
    use reson_agentic::mcp::UiResource;

    let html = "<html><body><h1>Calculator UI</h1></body></html>";
    let ui = UiResource::new("calc-server", "calculator-ui", html)
        .with_description("Interactive calculator");

    let server = McpServer::new("calc-server")
        .description("Calculator with UI")
        .agent(
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
            |args| {
                Box::pin(async move {
                    let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Ok((a + b).to_string())
                })
            },
        )
        .with_ui(ui);

    let addr = "127.0.0.1:19220";
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:19220")
        .await
        .expect("Failed to connect");

    // Tool should still work normally
    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.tools.len(), 1);
    assert_eq!(tools.tools[0].name.as_ref(), "add");

    // Tool metadata should contain ui.resourceUri
    let tool_meta = tools.tools[0].meta.as_ref().expect("Tool should have _meta");
    let ui_meta = tool_meta.0.get("ui").expect("Tool _meta should have 'ui' key");
    let resource_uri = ui_meta.get("resourceUri").and_then(|v| v.as_str()).unwrap();
    assert_eq!(resource_uri, "ui://calc-server/calculator-ui");

    // Resources should be listable
    let resources = client.list_resources().await.unwrap();
    assert_eq!(resources.resources.len(), 1);
    assert_eq!(resources.resources[0].raw.name, "calculator-ui");
    assert_eq!(
        resources.resources[0].raw.mime_type.as_deref(),
        Some("text/html;profile=mcp-app")
    );

    // Resource should be readable and return the HTML
    let read_result = client
        .read_resource("ui://calc-server/calculator-ui")
        .await
        .unwrap();
    assert_eq!(read_result.contents.len(), 1);
    // Verify the HTML content via JSON serialization (avoid leaking rmcp types)
    let content_json = serde_json::to_value(&read_result.contents[0]).unwrap();
    assert_eq!(content_json.get("text").and_then(|v| v.as_str()).unwrap(), html);
    assert_eq!(
        content_json.get("mimeType").and_then(|v| v.as_str()).unwrap(),
        "text/html;profile=mcp-app"
    );

    // Tool execution still works through the bridge
    let result = client.call_tool("add", json!({"a": 5, "b": 3})).await.unwrap();
    let text = result.content.first().and_then(|c| c.as_text()).unwrap();
    assert_eq!(text.text, "8");

    client.close().await.unwrap();
}

#[cfg(feature = "mcp-apps")]
#[tokio::test]
async fn test_ui_resource_with_csp_and_permissions() {
    use reson_agentic::mcp::{UiResource, UiResourceCsp};

    let html = "<html><body>Secure App</body></html>";
    let csp = UiResourceCsp {
        connect_domains: Some(vec!["api.example.com".into()]),
        resource_domains: Some(vec!["cdn.example.com".into()]),
        frame_domains: None,
        base_uri_domains: None,
    };
    let ui = UiResource::new("secure-server", "secure-app", html)
        .with_csp(csp)
        .with_border(true);

    let server = McpServer::new("secure-server")
        .tool(
            "status",
            "Get status",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("ok")]))
                })
            },
        )
        .with_ui(ui);

    let addr = "127.0.0.1:19221";
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:19221")
        .await
        .expect("Failed to connect");

    // Verify resource metadata includes CSP and border preferences
    let resources = client.list_resources().await.unwrap();
    assert_eq!(resources.resources.len(), 1);
    let resource_meta = resources.resources[0]
        .raw
        .meta
        .as_ref()
        .expect("Resource should have _meta");
    let ui_meta = resource_meta.0.get("ui").expect("Should have 'ui' meta");

    let csp_meta = ui_meta.get("csp").expect("Should have CSP");
    let connect = csp_meta.get("connectDomains").and_then(|v| v.as_array()).unwrap();
    assert_eq!(connect[0].as_str().unwrap(), "api.example.com");

    let prefers_border = ui_meta.get("prefersBorder").and_then(|v| v.as_bool());
    assert_eq!(prefers_border, Some(true));

    client.close().await.unwrap();
}

// --- Visibility filtering tests ---

#[cfg(feature = "mcp-apps")]
#[tokio::test]
async fn test_runtime_skips_app_only_tools() {
    use reson_agentic::mcp::{UiResource, Visibility};

    let server = McpServer::new("vis-filter-server")
        .tool(
            "model_tool",
            "Visible to model",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("model")]))
                })
            },
        )
        .with_ui(UiResource::new("vis-filter-server", "ui1", "<html></html>"))
        .visibility(vec![Visibility::Model])
        .tool(
            "app_tool",
            "Only for iframe",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("app")]))
                })
            },
        )
        .with_ui(UiResource::new("vis-filter-server", "ui2", "<html></html>"))
        .visibility(vec![Visibility::App])
        .tool(
            "both_tool",
            "For both",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("both")]))
                })
            },
        )
        .with_ui(UiResource::new("vis-filter-server", "ui3", "<html></html>"))
        .visibility(vec![Visibility::Model, Visibility::App]);

    let addr = "127.0.0.1:19230";
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19230").await.unwrap();

    let schemas = runtime.get_tool_schemas().await;
    // app_tool should be filtered out — only model_tool and both_tool registered
    assert_eq!(schemas.len(), 2);
    assert!(schemas.contains_key("model_tool"));
    assert!(schemas.contains_key("both_tool"));
    assert!(!schemas.contains_key("app_tool"));
}

#[cfg(feature = "mcp-apps")]
#[tokio::test]
async fn test_runtime_includes_tools_with_no_visibility() {
    use reson_agentic::mcp::UiResource;

    // Default visibility (None) means ["model", "app"] — should be registered
    let server = McpServer::new("default-vis-server")
        .tool(
            "default_tool",
            "Default visibility",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("default")]))
                })
            },
        )
        .with_ui(UiResource::new("default-vis-server", "ui", "<html></html>"));

    let addr = "127.0.0.1:19231";
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19231").await.unwrap();

    let schemas = runtime.get_tool_schemas().await;
    assert_eq!(schemas.len(), 1);
    assert!(schemas.contains_key("default_tool"));
}

// --- Error handling tests ---

#[tokio::test]
async fn test_connection_failure_returns_error() {
    let runtime = Runtime::new();
    let result = runtime.mcp("ws://127.0.0.1:19299").await;
    assert!(result.is_err(), "Connecting to non-existent server should fail");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("Failed to connect"),
        "Error should mention connection failure: {}",
        err
    );
}

#[tokio::test]
async fn test_tool_returning_error_content() {
    // Server with a tool that returns an error result
    let server = reson_mcp::server::McpServer::builder("error-server")
        .with_tool(
            "fail",
            "Always fails",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::error(vec![Content::text(
                        "something went wrong",
                    )]))
                })
            },
        )
        .build();

    let addr = "127.0.0.1:19222";
    tokio::spawn(async move {
        server
            .serve(reson_mcp::server::ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19222").await.unwrap();

    // The tool should still execute and return the text (error content is still text)
    let result = runtime
        .execute_tool(&json!({"_tool_name": "fail"}))
        .await
        .unwrap();
    assert_eq!(result, "something went wrong");
}

// --- Edge case tests ---

#[tokio::test]
async fn test_tool_with_no_parameters() {
    let server = reson_mcp::server::McpServer::builder("no-args-server")
        .with_tool(
            "ping",
            "Returns pong",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("pong")]))
                })
            },
        )
        .build();

    let addr = "127.0.0.1:19223";
    tokio::spawn(async move {
        server
            .serve(reson_mcp::server::ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19223").await.unwrap();

    let schemas = runtime.get_tool_schemas().await;
    assert!(schemas.contains_key("ping"));
    assert_eq!(schemas["ping"].fields.len(), 0);

    let result = runtime
        .execute_tool(&json!({"_tool_name": "ping"}))
        .await
        .unwrap();
    assert_eq!(result, "pong");
}

#[tokio::test]
async fn test_tool_returning_multiple_content_blocks() {
    let server = reson_mcp::server::McpServer::builder("multi-content-server")
        .with_tool(
            "multi",
            "Returns multiple content blocks",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![
                        Content::text("line one"),
                        Content::text("line two"),
                        Content::text("line three"),
                    ]))
                })
            },
        )
        .build();

    let addr = "127.0.0.1:19224";
    tokio::spawn(async move {
        server
            .serve(reson_mcp::server::ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19224").await.unwrap();

    // Multiple content blocks should be joined with newlines
    let result = runtime
        .execute_tool(&json!({"_tool_name": "multi"}))
        .await
        .unwrap();
    assert_eq!(result, "line one\nline two\nline three");
}

#[tokio::test]
async fn test_mcp_as_with_agent_server_e2e() {
    // Use McpServer wrapper (reson-agentic's) with namespacing
    let server = McpServer::new("math-service")
        .agent(
            "square",
            "Square a number",
            json!({
                "type": "object",
                "properties": {
                    "n": { "type": "number", "description": "Number to square" }
                },
                "required": ["n"]
            }),
            |args| {
                Box::pin(async move {
                    let n = args.get("n").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Ok((n * n).to_string())
                })
            },
        );

    let addr = "127.0.0.1:19225";
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr.into()))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp_as("ws://127.0.0.1:19225", "math").await.unwrap();

    // Tool registered under namespaced name
    let schemas = runtime.get_tool_schemas().await;
    assert!(schemas.contains_key("math_square"));
    assert!(!schemas.contains_key("square"));

    // Execution works via namespaced name
    let result = runtime
        .execute_tool(&json!({"_tool_name": "math_square", "n": 7}))
        .await
        .unwrap();
    assert_eq!(result, "49");
}

#[tokio::test]
async fn test_multiple_servers_without_namespace() {
    // Two servers with non-overlapping tool names, no namespace needed
    let server1 = reson_mcp::server::McpServer::builder("server-a")
        .with_tool(
            "tool_a",
            "Tool from server A",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("from A")]))
                })
            },
        )
        .build();

    let server2 = reson_mcp::server::McpServer::builder("server-b")
        .with_tool(
            "tool_b",
            "Tool from server B",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move {
                    Ok(CallToolResult::success(vec![Content::text("from B")]))
                })
            },
        )
        .build();

    tokio::spawn(async move {
        server1
            .serve(reson_mcp::server::ServerTransport::WebSocket(
                "127.0.0.1:19226".into(),
            ))
            .await
            .unwrap();
    });
    tokio::spawn(async move {
        server2
            .serve(reson_mcp::server::ServerTransport::WebSocket(
                "127.0.0.1:19227".into(),
            ))
            .await
            .unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let runtime = Runtime::new();
    runtime.mcp("ws://127.0.0.1:19226").await.unwrap();
    runtime.mcp("ws://127.0.0.1:19227").await.unwrap();

    let schemas = runtime.get_tool_schemas().await;
    assert_eq!(schemas.len(), 2);
    assert!(schemas.contains_key("tool_a"));
    assert!(schemas.contains_key("tool_b"));

    let ra = runtime
        .execute_tool(&json!({"_tool_name": "tool_a"}))
        .await
        .unwrap();
    let rb = runtime
        .execute_tool(&json!({"_tool_name": "tool_b"}))
        .await
        .unwrap();
    assert_eq!(ra, "from A");
    assert_eq!(rb, "from B");
}

// ============================================================================
// LLM Agent Tests - require API keys and a running MCP server
// ============================================================================

fn get_anthropic_key() -> Option<String> {
    env::var("ANTHROPIC_API_KEY").ok()
}

fn get_google_key() -> Option<String> {
    env::var("GOOGLE_GEMINI_API_KEY").ok()
}

fn require_tool_call_id(tool_call: &serde_json::Value) -> reson_agentic::error::Result<String> {
    tool_call
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .map(|s| s.to_string())
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool call id".to_string()))
}

fn parse_i64_value(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(num) => num.as_i64(),
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn require_i64_field(
    value: &serde_json::Value,
    field: &str,
) -> reson_agentic::error::Result<i64> {
    // Try top-level (args are flattened by call_llm)
    if let Some(raw) = value.get(field) {
        if let Some(v) = parse_i64_value(raw) {
            return Ok(v);
        }
    }
    // Try nested input (Anthropic format)
    if let Some(input) = value.get("input") {
        if let Some(raw) = input.get(field) {
            if let Some(v) = parse_i64_value(raw) {
                return Ok(v);
            }
        }
    }
    // Try nested arguments (OpenAI format)
    if let Some(func) = value.get("function") {
        if let Some(args_str) = func.get("arguments").and_then(|a| a.as_str()) {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(args_str) {
                if let Some(raw) = args.get(field) {
                    if let Some(v) = parse_i64_value(raw) {
                        return Ok(v);
                    }
                }
            }
        }
    }
    Err(reson_agentic::error::Error::NonRetryable(format!(
        "Missing '{}' field in {:?}",
        field, value
    )))
}

/// Agentic function that connects to an MCP calculator server and uses its tools
/// via Anthropic Claude.
#[agentic(model = "anthropic:claude-haiku-4-5-20251001")]
async fn anthropic_mcp_tool_agent(
    prompt: String,
    mcp_url: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    // Connect to the MCP server - its tools become available to the runtime
    runtime.mcp(&mcp_url).await?;

    // Ask the LLM to use the tool
    let tool_call = runtime
        .run(
            Some(&prompt),
            Some("You have access to calculator tools. Use them when asked to do math. Always use the tools, never calculate yourself."),
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Expected tool call, got: {:?}",
            tool_call
        )));
    }

    let tool_name = runtime.get_tool_name(&tool_call).unwrap();
    let tool_use_id = require_tool_call_id(&tool_call)?;

    // Execute the MCP tool through the runtime
    let tool_output = runtime.execute_tool(&tool_call).await?;

    // Build history with tool result and get final answer
    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;

    let mut tc = ToolCall::new(&tool_name, json!({"a": a, "b": b}));
    tc.tool_use_id = tool_use_id.clone();

    let history = vec![
        ConversationMessage::Chat(ChatMessage::user(&prompt)),
        ConversationMessage::ToolCall(tc),
        ConversationMessage::ToolResult(
            ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
        ),
    ];

    let response = runtime
        .run(
            Some("What was the result? Reply with just the number, nothing else."),
            Some("Respond with only the number."),
            Some(history),
            None,
            None,
            Some(0.0),
            None,
            Some(128),
            None,
            None,
        )
        .await?;

    Ok(response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string()))
}

/// Same test but with Google Gemini
#[agentic(model = "google-genai:gemini-2.0-flash")]
async fn google_mcp_tool_agent(
    prompt: String,
    mcp_url: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    runtime.mcp(&mcp_url).await?;

    let tool_call = runtime
        .run(
            Some(&prompt),
            Some("You have access to calculator tools. Use them when asked to do math. Always use the tools, never calculate yourself."),
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Expected tool call, got: {:?}",
            tool_call
        )));
    }

    let tool_name = runtime.get_tool_name(&tool_call).unwrap();
    let tool_use_id = require_tool_call_id(&tool_call)?;
    let tool_output = runtime.execute_tool(&tool_call).await?;

    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;

    let mut tc = ToolCall::new(&tool_name, json!({"a": a, "b": b}));
    tc.tool_use_id = tool_use_id.clone();

    let history = vec![
        ConversationMessage::Chat(ChatMessage::user(&prompt)),
        ConversationMessage::ToolCall(tc),
        ConversationMessage::ToolResult(
            ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
        ),
    ];

    let response = runtime
        .run(
            Some("What was the result? Reply with just the number, nothing else."),
            Some("Respond with only the number."),
            Some(history),
            None,
            None,
            Some(0.0),
            None,
            Some(128),
            None,
            None,
        )
        .await?;

    Ok(response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string()))
}

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_agent_with_mcp_tools() {
    let _ = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");

    // Start MCP calculator server
    let _server = start_ws_server(19210).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let result = anthropic_mcp_tool_agent(
        "Use the add tool to calculate 15 + 27.".to_string(),
        "ws://127.0.0.1:19210".to_string(),
    )
    .await
    .unwrap();

    println!("Anthropic MCP result: {}", result);
    let value = result.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 42);
}

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_agent_with_mcp_tools() {
    let _ = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");

    // Start MCP calculator server
    let _server = start_ws_server(19211).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let result = google_mcp_tool_agent(
        "Use the multiply tool to calculate 6 times 7.".to_string(),
        "ws://127.0.0.1:19211".to_string(),
    )
    .await
    .unwrap();

    println!("Google MCP result: {}", result);
    let value = result.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 42);
}
