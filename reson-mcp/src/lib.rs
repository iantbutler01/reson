//! # reson-mcp
//!
//! MCP (Model Context Protocol) support for reson.
//!
//! This crate provides:
//! - **Client**: Connect to MCP servers and use their tools
//! - **Server**: Expose tools as an MCP server
//! - **Apps**: MCP Apps extension (SEP-1865) for interactive UIs
//!
//! ## Quick Start - Client
//!
//! ```rust,no_run
//! use reson_mcp::client::McpClient;
//!
//! # async fn example() -> reson_mcp::Result<()> {
//! // Connect to an MCP server via HTTP
//! let client = McpClient::http("http://localhost:8080").await?;
//!
//! // List available tools
//! let tools = client.list_tools().await?;
//!
//! // Call a tool
//! let result = client.call_tool("read_file", serde_json::json!({"path": "/tmp/test.txt"})).await?;
//! # Ok(())
//! # }
//! ```

pub mod error;
pub mod transport;

#[cfg(feature = "client")]
pub mod client;

#[cfg(feature = "server")]
pub mod server;

#[cfg(feature = "apps")]
pub mod apps;

pub use error::{Error, Result};
pub use transport::WebSocketTransport;

// Re-export commonly used rmcp types
pub use rmcp::model::{CallToolResult, Content, Tool as McpTool};
pub use rmcp::ErrorData;
