//! Integration tests for MCP Apps extension (SEP-1865)

use rmcp::model::{CallToolResult, Content};
use reson_mcp::apps::{UiResource, UiResourceCsp, UiToolMeta, Visibility, MCP_APP_MIME_TYPE};
use reson_mcp::server::{McpServer, ServerTransport};
use serde_json::json;

const TEST_HTML: &str = "<html><body><h1>Chart</h1></body></html>";

fn build_apps_server() -> McpServer {
    McpServer::builder("apps-test-server")
        .with_version("1.0.0")
        .with_description("A server with UI tools")
        .with_tool(
            "chart",
            "Render a chart",
            json!({
                "type": "object",
                "properties": {
                    "data": { "type": "string" }
                },
                "required": ["data"]
            }),
            |_name, args| {
                Box::pin(async move {
                    let args = args.unwrap_or_default();
                    let data = args
                        .get("data")
                        .and_then(|v| v.as_str())
                        .unwrap_or("no data");
                    Ok(CallToolResult::success(vec![Content::text(format!(
                        "Rendered: {}",
                        data
                    ))]))
                })
            },
        )
        .with_ui(
            UiResource::new("apps-test-server", "chart", TEST_HTML)
                .with_description("Interactive chart UI"),
        )
        .with_tool(
            "plain",
            "A plain tool with no UI",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move { Ok(CallToolResult::success(vec![Content::text("ok")])) })
            },
        )
        .build()
}

async fn start_apps_server(port: u16) -> tokio::task::JoinHandle<()> {
    let server = build_apps_server();
    let addr = format!("127.0.0.1:{}", port);
    tokio::spawn(async move {
        server
            .serve(ServerTransport::WebSocket(addr))
            .await
            .expect("Server failed");
    })
}

#[tokio::test]
async fn test_tool_has_ui_meta() {
    let _server = start_apps_server(18201).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18201")
        .await
        .expect("Failed to connect");

    let tools = client.list_tools().await.expect("Failed to list tools");
    assert_eq!(tools.tools.len(), 2);

    // The "chart" tool should have _meta.ui with resourceUri
    let chart_tool = tools.tools.iter().find(|t| t.name == "chart").unwrap();
    let meta = chart_tool.meta.as_ref().expect("chart tool should have _meta");
    let ui_value = meta.0.get("ui").expect("_meta should have 'ui' key");
    let ui_meta: UiToolMeta =
        serde_json::from_value(ui_value.clone()).expect("should deserialize as UiToolMeta");
    assert_eq!(ui_meta.resource_uri.scheme(), "ui");
    assert_eq!(ui_meta.resource_uri.host_str(), Some("apps-test-server"));

    // The "plain" tool should have no _meta
    let plain_tool = tools.tools.iter().find(|t| t.name == "plain").unwrap();
    assert!(plain_tool.meta.is_none());

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_list_resources() {
    let _server = start_apps_server(18202).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18202")
        .await
        .expect("Failed to connect");

    let resources = client
        .list_resources()
        .await
        .expect("Failed to list resources");

    assert_eq!(resources.resources.len(), 1);
    let resource = &resources.resources[0];
    assert_eq!(resource.raw.name, "chart");
    assert!(resource.raw.uri.starts_with("ui://apps-test-server/"));
    assert_eq!(
        resource.raw.mime_type.as_deref(),
        Some(MCP_APP_MIME_TYPE)
    );
    assert_eq!(
        resource.raw.description.as_deref(),
        Some("Interactive chart UI")
    );

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_read_resource() {
    let _server = start_apps_server(18203).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18203")
        .await
        .expect("Failed to connect");

    // Read the resource by its URI
    let result = client
        .read_resource("ui://apps-test-server/chart")
        .await
        .expect("Failed to read resource");

    assert_eq!(result.contents.len(), 1);
    match &result.contents[0] {
        rmcp::model::ResourceContents::TextResourceContents {
            text, mime_type, ..
        } => {
            assert_eq!(text, TEST_HTML);
            assert_eq!(mime_type.as_deref(), Some(MCP_APP_MIME_TYPE));
        }
        _ => panic!("Expected TextResourceContents"),
    }

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_read_resource_not_found() {
    let _server = start_apps_server(18204).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18204")
        .await
        .expect("Failed to connect");

    let result = client.read_resource("ui://apps-test-server/nonexistent").await;
    assert!(result.is_err(), "Reading nonexistent resource should fail");

    client.close().await.expect("Failed to close");
}

#[tokio::test]
async fn test_ui_resource_with_csp() {
    let resource = UiResource::new("my-server", "dashboard", "<html></html>")
        .with_description("Dashboard")
        .with_csp(UiResourceCsp {
            connect_domains: Some(vec!["api.example.com".to_string()]),
            resource_domains: None,
            frame_domains: None,
            base_uri_domains: None,
        });

    assert_eq!(resource.name, "dashboard");
    assert!(resource.meta.is_some());
    let meta = resource.meta.unwrap();
    let csp = meta.csp.unwrap();
    assert_eq!(
        csp.connect_domains,
        Some(vec!["api.example.com".to_string()])
    );
}

#[tokio::test]
async fn test_server_capabilities_include_resources() {
    let _server = start_apps_server(18205).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18205")
        .await
        .expect("Failed to connect");

    let info = client
        .server_info()
        .expect("Server info should be available");

    // Server with UI resources should advertise resources capability
    assert!(
        info.capabilities.resources.is_some(),
        "Server with UI tools should have resources capability"
    );

    client.close().await.expect("Failed to close");
}

// --- Visibility tests ---

#[tokio::test]
async fn test_tool_visibility_set_via_builder() {
    let server = McpServer::builder("vis-server")
        .with_tool(
            "model_only",
            "Only for the LLM",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move { Ok(CallToolResult::success(vec![Content::text("ok")])) })
            },
        )
        .with_ui(UiResource::new("vis-server", "ui1", "<html></html>"))
        .visibility(vec![Visibility::Model])
        .with_tool(
            "app_only",
            "Only for the iframe",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move { Ok(CallToolResult::success(vec![Content::text("ok")])) })
            },
        )
        .with_ui(UiResource::new("vis-server", "ui2", "<html></html>"))
        .visibility(vec![Visibility::App])
        .with_tool(
            "both",
            "For model and app",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move { Ok(CallToolResult::success(vec![Content::text("ok")])) })
            },
        )
        .with_ui(UiResource::new("vis-server", "ui3", "<html></html>"))
        .visibility(vec![Visibility::Model, Visibility::App])
        .build();

    let addr = "127.0.0.1:18206";
    tokio::spawn(async move {
        server.serve(ServerTransport::WebSocket(addr.into())).await.unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18206")
        .await
        .expect("Failed to connect");

    let tools = client.list_tools().await.unwrap();
    assert_eq!(tools.tools.len(), 3);

    let model_only = tools.tools.iter().find(|t| t.name == "model_only").unwrap();
    let ui = model_only.meta.as_ref().unwrap().0.get("ui").unwrap();
    let vis: Vec<String> = ui["visibility"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(vis, vec!["model"]);

    let app_only = tools.tools.iter().find(|t| t.name == "app_only").unwrap();
    let ui = app_only.meta.as_ref().unwrap().0.get("ui").unwrap();
    let vis: Vec<String> = ui["visibility"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(vis, vec!["app"]);

    let both = tools.tools.iter().find(|t| t.name == "both").unwrap();
    let ui = both.meta.as_ref().unwrap().0.get("ui").unwrap();
    let vis: Vec<String> = ui["visibility"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(vis, vec!["model", "app"]);

    client.close().await.unwrap();
}

#[tokio::test]
async fn test_default_visibility_omitted() {
    // When no .visibility() is chained, _meta.ui should not have a visibility field
    let server = McpServer::builder("default-vis-server")
        .with_tool(
            "default_tool",
            "Default visibility",
            json!({"type": "object"}),
            |_name, _args| {
                Box::pin(async move { Ok(CallToolResult::success(vec![Content::text("ok")])) })
            },
        )
        .with_ui(UiResource::new("default-vis-server", "ui", "<html></html>"))
        .build();

    let addr = "127.0.0.1:18207";
    tokio::spawn(async move {
        server.serve(ServerTransport::WebSocket(addr.into())).await.unwrap();
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18207")
        .await
        .expect("Failed to connect");

    let tools = client.list_tools().await.unwrap();
    let tool = &tools.tools[0];
    let ui = tool.meta.as_ref().unwrap().0.get("ui").unwrap();
    // No visibility field means default ["model", "app"] per spec
    assert!(ui.get("visibility").is_none());

    client.close().await.unwrap();
}

// --- E2E resource flow: tool resourceUri -> resources/read ---

#[tokio::test]
async fn test_e2e_tool_resource_uri_resolves() {
    let _server = start_apps_server(18208).await;
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let client = reson_mcp::client::McpClient::websocket("ws://127.0.0.1:18208")
        .await
        .expect("Failed to connect");

    // 1. List tools and find one with a UI resource
    let tools = client.list_tools().await.unwrap();
    let chart_tool = tools.tools.iter().find(|t| t.name == "chart").unwrap();

    // 2. Extract the resourceUri from _meta.ui
    let ui_meta: UiToolMeta = serde_json::from_value(
        chart_tool.meta.as_ref().unwrap().0.get("ui").unwrap().clone(),
    )
    .unwrap();
    let resource_uri = ui_meta.resource_uri.to_string();

    // 3. Read that resource by its URI
    let read_result = client.read_resource(&resource_uri).await.unwrap();
    assert_eq!(read_result.contents.len(), 1);

    // 4. Verify we got the HTML
    match &read_result.contents[0] {
        rmcp::model::ResourceContents::TextResourceContents { text, mime_type, .. } => {
            assert_eq!(text, TEST_HTML);
            assert_eq!(mime_type.as_deref(), Some(MCP_APP_MIME_TYPE));
        }
        _ => panic!("Expected TextResourceContents"),
    }

    client.close().await.unwrap();
}
