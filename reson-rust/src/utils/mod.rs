//! Utility functions
//!
//! Backoff, streaming helpers, UTF-8 handling, etc.

pub mod json_stream;
pub mod message_conversion;
pub mod sse;

pub use json_stream::{
    parse_json_value_strict_bytes, parse_json_value_strict_str, JsonStreamAccumulator,
};
pub use message_conversion::{
    convert_messages_to_provider_format, convert_messages_to_responses_input,
    media_part_to_google_format, ConversationMessage,
};
pub use sse::parse_sse_stream;
