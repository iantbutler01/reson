//! MCP Server - expose tools as MCP servers
//!
//! This module provides functionality to expose tools as MCP servers
//! that can be consumed by MCP hosts like Claude Desktop.

mod handler;

pub use handler::{McpServer, McpServerBuilder, ServerTransport};
