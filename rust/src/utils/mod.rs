//! Utility functions
//!
//! Backoff, streaming helpers, UTF-8 handling, etc.

pub mod json_stream;
pub mod message_conversion;
pub mod sse;

pub use json_stream::{
    JsonStreamAccumulator, parse_json_value_strict_bytes, parse_json_value_strict_str,
};
pub use message_conversion::{
    ConversationMessage, convert_messages_to_provider_format, convert_messages_to_responses_input,
    media_part_to_google_format, messages_contain_image_input, validate_image_input_supported,
};
pub use sse::parse_sse_stream;
