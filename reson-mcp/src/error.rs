//! Error types for reson-mcp

use thiserror::Error;

/// Result type alias for reson-mcp operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for MCP operations
#[derive(Error, Debug)]
pub enum Error {
    /// Transport error (connection failed, disconnected, etc.)
    #[error("Transport error: {0}")]
    Transport(String),

    /// Protocol error (invalid message format, unexpected response, etc.)
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Tool not found
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Tool execution failed
    #[error("Tool execution failed: {0}")]
    ToolExecution(String),

    /// Resource not found
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// rmcp error
    #[error("MCP error: {0}")]
    Mcp(#[from] rmcp::ErrorData),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Server initialization error
    #[error("Server initialization error: {0}")]
    ServerInit(String),

    /// Client not connected
    #[error("Client not connected")]
    NotConnected,
}
