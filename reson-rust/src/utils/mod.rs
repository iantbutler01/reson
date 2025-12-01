//! Utility functions
//!
//! Backoff, streaming helpers, UTF-8 handling, etc.

pub mod message_conversion;
pub mod sse;

pub use message_conversion::{convert_messages_to_provider_format, ConversationMessage};
pub use sse::parse_sse_stream;
