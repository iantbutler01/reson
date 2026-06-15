//! Transport implementations for MCP
//!
//! Provides transport types shared by both client and server.

pub mod websocket;

pub use websocket::WebSocketTransport;
