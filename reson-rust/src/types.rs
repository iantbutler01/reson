//! Core types for Reson
//!
//! This module contains the fundamental data structures used throughout
//! the framework: messages, tool calls, tool results, and reasoning segments.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{Error, Result};

/// LLM Provider enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    /// Anthropic Claude (direct API)
    Anthropic,
    /// OpenAI GPT models (direct API)
    OpenAI,
    /// AWS Bedrock (Claude and others)
    Bedrock,
    /// Google Gemini via GenAI SDK
    GoogleGenAI,
    /// Anthropic Claude via Google Vertex AI
    GoogleAnthropic,
    /// OpenRouter proxy service
    OpenRouter,
}

impl Provider {
    /// Parse provider from model string (e.g., "anthropic:claude-3-opus")
    pub fn from_model_string(model: &str) -> Result<(Self, String)> {
        if let Some((provider_str, model_name)) = model.split_once(':') {
            let provider = match provider_str {
                "anthropic" => Provider::Anthropic,
                "openai" => Provider::OpenAI,
                "bedrock" => Provider::Bedrock,
                "google-genai" | "gemini" => Provider::GoogleGenAI,
                "google-anthropic" | "vertexai" => Provider::GoogleAnthropic,
                "openrouter" => Provider::OpenRouter,
                _ => return Err(Error::InvalidProvider(provider_str.to_string())),
            };
            Ok((provider, model_name.to_string()))
        } else {
            Err(Error::InvalidProvider(format!(
                "Invalid model string format: {}. Expected 'provider:model'",
                model
            )))
        }
    }

    /// Check if this provider supports native tool calling
    pub fn supports_native_tools(&self) -> bool {
        matches!(
            self,
            Provider::Anthropic
                | Provider::OpenAI
                | Provider::Bedrock
                | Provider::GoogleGenAI
                | Provider::GoogleAnthropic
                | Provider::OpenRouter
        )
    }
}

/// Chat message role
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    /// System message (instructions)
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
    /// Tool result message
    #[serde(rename = "tool")]
    Tool,
}

/// Cache marker for Anthropic prompt caching
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CacheMarker {
    /// Ephemeral cache point
    Ephemeral,
}

/// A chat message in the conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Message role
    pub role: ChatRole,

    /// Message content (text)
    pub content: String,

    /// Optional cache marker (for Anthropic caching)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_marker: Option<CacheMarker>,

    /// Optional model family filtering
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_families: Option<Vec<String>>,

    /// Provider-specific signature for reasoning preservation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl ChatMessage {
    /// Create a new system message
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
            cache_marker: None,
            model_families: None,
            signature: None,
        }
    }

    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
            cache_marker: None,
            model_families: None,
            signature: None,
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
            cache_marker: None,
            model_families: None,
            signature: None,
        }
    }

    /// Add a cache marker to this message
    pub fn with_cache_marker(mut self, marker: CacheMarker) -> Self {
        self.cache_marker = Some(marker);
        self
    }
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Input tokens consumed
    pub input_tokens: u64,

    /// Output tokens generated
    pub output_tokens: u64,

    /// Cached tokens (for Anthropic caching)
    pub cached_tokens: u64,
}

impl TokenUsage {
    /// Create new token usage stats
    pub fn new(input_tokens: u64, output_tokens: u64, cached_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cached_tokens,
        }
    }

    /// Total tokens (input + output)
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// A tool call from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool use
    pub tool_use_id: String,

    /// Name of the tool to call
    pub tool_name: String,

    /// Tool arguments (parsed JSON)
    pub args: serde_json::Value,

    /// Raw argument string (for debugging)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,

    /// Provider signature for reasoning preservation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(tool_name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            tool_use_id: Uuid::new_v4().to_string(),
            tool_name: tool_name.into(),
            args,
            raw_arguments: None,
            signature: None,
        }
    }

    /// Create from provider-specific format
    pub fn from_provider_format(provider_format: serde_json::Value, provider: Provider) -> Result<Self> {
        match provider {
            Provider::Anthropic | Provider::Bedrock => {
                Ok(Self {
                    tool_use_id: provider_format["id"]
                        .as_str()
                        .ok_or_else(|| Error::Parse("Missing 'id' in tool call".to_string()))?
                        .to_string(),
                    tool_name: provider_format["name"]
                        .as_str()
                        .ok_or_else(|| Error::Parse("Missing 'name' in tool call".to_string()))?
                        .to_string(),
                    args: provider_format["input"].clone(),
                    raw_arguments: None,
                    signature: None,
                })
            }
            Provider::OpenAI | Provider::OpenRouter => {
                let function = &provider_format["function"];
                let args_str = function["arguments"]
                    .as_str()
                    .ok_or_else(|| Error::Parse("Missing 'arguments' in tool call".to_string()))?;

                Ok(Self {
                    tool_use_id: provider_format["id"]
                        .as_str()
                        .ok_or_else(|| Error::Parse("Missing 'id' in tool call".to_string()))?
                        .to_string(),
                    tool_name: function["name"]
                        .as_str()
                        .ok_or_else(|| Error::Parse("Missing 'name' in function".to_string()))?
                        .to_string(),
                    args: serde_json::from_str(args_str)?,
                    raw_arguments: Some(args_str.to_string()),
                    signature: None,
                })
            }
            Provider::GoogleGenAI | Provider::GoogleAnthropic => {
                Ok(Self {
                    tool_use_id: Uuid::new_v4().to_string(),
                    tool_name: provider_format["functionCall"]["name"]
                        .as_str()
                        .ok_or_else(|| Error::Parse("Missing 'name' in functionCall".to_string()))?
                        .to_string(),
                    args: provider_format["functionCall"]["args"].clone(),
                    raw_arguments: None,
                    signature: None,
                })
            }
        }
    }

    /// Convert to provider-specific assistant message format
    pub fn to_provider_assistant_message(&self, provider: Provider) -> serde_json::Value {
        match provider {
            Provider::Anthropic | Provider::Bedrock => {
                serde_json::json!({
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": self.tool_use_id,
                        "name": self.tool_name,
                        "input": self.args
                    }]
                })
            }
            Provider::OpenAI | Provider::OpenRouter => {
                serde_json::json!({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": self.tool_use_id,
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": self.raw_arguments.as_ref()
                                .unwrap_or(&serde_json::to_string(&self.args).unwrap())
                        }
                    }]
                })
            }
            Provider::GoogleGenAI | Provider::GoogleAnthropic => {
                serde_json::json!({
                    "role": "model",
                    "parts": [{
                        "functionCall": {
                            "name": self.tool_name,
                            "args": self.args
                        }
                    }]
                })
            }
        }
    }
}

/// Result of a tool execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// ID matching the original tool call
    pub tool_use_id: String,

    /// Name of the tool that was called (needed for Google API)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,

    /// Result content (stringified)
    pub content: String,

    /// Whether the tool execution resulted in an error
    pub is_error: bool,

    /// Provider signature for reasoning preservation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    /// Original tool call object for reference (as JSON)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_obj: Option<serde_json::Value>,
}

impl ToolResult {
    /// Create a successful tool result
    pub fn success(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            tool_name: None,
            content: content.into(),
            is_error: false,
            signature: None,
            tool_obj: None,
        }
    }

    /// Create a successful tool result with tool name (for Google API)
    pub fn success_with_name(
        tool_use_id: impl Into<String>,
        tool_name: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            tool_name: Some(tool_name.into()),
            content: content.into(),
            is_error: false,
            signature: None,
            tool_obj: None,
        }
    }

    /// Create an error tool result
    pub fn error(tool_use_id: impl Into<String>, error: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            tool_name: None,
            content: error.into(),
            is_error: true,
            signature: None,
            tool_obj: None,
        }
    }

    /// Set the tool name
    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = Some(name.into());
        self
    }

    /// Set the tool object (JSON representation)
    pub fn with_tool_obj(mut self, obj: serde_json::Value) -> Self {
        self.tool_obj = Some(obj);
        self
    }

    /// Convert to provider-specific format
    pub fn to_provider_format(&self, provider: Provider) -> serde_json::Value {
        match provider {
            Provider::Anthropic | Provider::Bedrock => {
                serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": self.tool_use_id,
                    "content": self.content,
                    "is_error": self.is_error
                })
            }
            Provider::OpenAI | Provider::OpenRouter => {
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": self.tool_use_id,
                    "content": self.content
                })
            }
            Provider::GoogleGenAI | Provider::GoogleAnthropic => {
                // Try to get tool_name from: 1) tool_name field, 2) tool_obj["_tool_name"], 3) "unknown"
                let name = self.tool_name.clone().unwrap_or_else(|| {
                    self.tool_obj
                        .as_ref()
                        .and_then(|obj| obj.get("_tool_name"))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                });

                serde_json::json!({
                    "functionResponse": {
                        "name": name,
                        "response": {
                            "result": self.content
                        }
                    }
                })
            }
        }
    }
}

/// A reasoning segment (extended thinking)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSegment {
    /// Reasoning content
    pub content: String,

    /// Index in sequence of reasoning segments
    pub segment_index: usize,

    /// Provider signature for reasoning preservation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    /// Provider-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

impl ReasoningSegment {
    /// Create a new reasoning segment with just content (index defaults to 0)
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            segment_index: 0,
            signature: None,
            provider_metadata: None,
        }
    }

    /// Create a new reasoning segment with content and index
    pub fn with_index(content: impl Into<String>, segment_index: usize) -> Self {
        Self {
            content: content.into(),
            segment_index,
            signature: None,
            provider_metadata: None,
        }
    }

    /// Set the signature
    pub fn with_signature(mut self, signature: impl Into<String>) -> Self {
        self.signature = Some(signature.into());
        self
    }

    /// Set provider metadata
    pub fn with_provider_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.provider_metadata = Some(metadata);
        self
    }

    /// Set segment index
    pub fn with_segment_index(mut self, index: usize) -> Self {
        self.segment_index = index;
        self
    }

    /// Convert to provider-specific format
    pub fn to_provider_format(&self, provider: Provider) -> serde_json::Value {
        match provider {
            // Anthropic, Bedrock, and GoogleAnthropic (Vertex AI Claude) all use Anthropic format
            Provider::Anthropic | Provider::Bedrock | Provider::GoogleAnthropic => {
                let mut obj = serde_json::json!({
                    "type": "thinking",
                    "thinking": self.content
                });
                if let Some(ref sig) = self.signature {
                    obj["signature"] = serde_json::Value::String(sig.clone());
                }
                obj
            }
            Provider::OpenAI | Provider::OpenRouter => {
                let mut obj = serde_json::json!({
                    "type": "reasoning",
                    "content": self.content
                });
                if let Some(ref sig) = self.signature {
                    obj["signature"] = serde_json::Value::String(sig.clone());
                }
                obj
            }
            Provider::GoogleGenAI => {
                let mut obj = serde_json::json!({
                    "thought": true,
                    "text": self.content
                });
                if let Some(ref sig) = self.signature {
                    obj["thought_signature"] = serde_json::Value::String(sig.clone());
                }
                obj
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_from_model_string() {
        let (provider, model) = Provider::from_model_string("anthropic:claude-3-opus").unwrap();
        assert_eq!(provider, Provider::Anthropic);
        assert_eq!(model, "claude-3-opus");

        let (provider, model) = Provider::from_model_string("openai:gpt-4").unwrap();
        assert_eq!(provider, Provider::OpenAI);
        assert_eq!(model, "gpt-4");

        assert!(Provider::from_model_string("invalid").is_err());
    }

    #[test]
    fn test_provider_supports_native_tools() {
        assert!(Provider::Anthropic.supports_native_tools());
        assert!(Provider::OpenAI.supports_native_tools());
        assert!(Provider::GoogleAnthropic.supports_native_tools());
        assert!(Provider::GoogleGenAI.supports_native_tools());
        assert!(Provider::OpenRouter.supports_native_tools());
        assert!(Provider::Bedrock.supports_native_tools());
    }

    #[test]
    fn test_chat_message_constructors() {
        let msg = ChatMessage::system("You are helpful");
        assert_eq!(msg.role, ChatRole::System);
        assert_eq!(msg.content, "You are helpful");

        let msg = ChatMessage::user("Hello");
        assert_eq!(msg.role, ChatRole::User);

        let msg = ChatMessage::assistant("Hi there")
            .with_cache_marker(CacheMarker::Ephemeral);
        assert_eq!(msg.role, ChatRole::Assistant);
        assert!(msg.cache_marker.is_some());
    }

    #[test]
    fn test_token_usage() {
        let usage = TokenUsage::new(100, 50, 25);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cached_tokens, 25);
        assert_eq!(usage.total_tokens(), 150);
    }

    #[test]
    fn test_tool_call_creation() {
        let tool_call = ToolCall::new("get_weather", serde_json::json!({"city": "SF"}));
        assert_eq!(tool_call.tool_name, "get_weather");
        assert_eq!(tool_call.args["city"], "SF");
        assert!(!tool_call.tool_use_id.is_empty());
    }

    #[test]
    fn test_tool_call_from_anthropic_format() {
        let format = serde_json::json!({
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"city": "SF"}
        });

        let tool_call = ToolCall::from_provider_format(format, Provider::Anthropic).unwrap();
        assert_eq!(tool_call.tool_use_id, "toolu_123");
        assert_eq!(tool_call.tool_name, "get_weather");
        assert_eq!(tool_call.args["city"], "SF");
    }

    #[test]
    fn test_tool_call_from_openai_format() {
        let format = serde_json::json!({
            "id": "call_123",
            "function": {
                "name": "get_weather",
                "arguments": "{\"city\":\"SF\"}"
            }
        });

        let tool_call = ToolCall::from_provider_format(format, Provider::OpenAI).unwrap();
        assert_eq!(tool_call.tool_use_id, "call_123");
        assert_eq!(tool_call.tool_name, "get_weather");
        assert_eq!(tool_call.args["city"], "SF");
    }

    #[test]
    fn test_tool_result_success() {
        let result = ToolResult::success("toolu_123", "Sunny, 72°F");
        assert_eq!(result.tool_use_id, "toolu_123");
        assert_eq!(result.content, "Sunny, 72°F");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_result_error() {
        let result = ToolResult::error("toolu_123", "API error");
        assert!(result.is_error);
    }

    #[test]
    fn test_tool_result_to_anthropic_format() {
        let result = ToolResult::success("toolu_123", "Result");
        let format = result.to_provider_format(Provider::Anthropic);

        assert_eq!(format["type"], "tool_result");
        assert_eq!(format["tool_use_id"], "toolu_123");
        assert_eq!(format["content"], "Result");
    }

    #[test]
    fn test_tool_result_to_openai_format() {
        let result = ToolResult::success("call_123", "Result");
        let format = result.to_provider_format(Provider::OpenAI);

        assert_eq!(format["role"], "tool");
        assert_eq!(format["tool_call_id"], "call_123");
        assert_eq!(format["content"], "Result");
    }

    #[test]
    fn test_reasoning_segment() {
        let segment = ReasoningSegment::with_index("Thinking...", 0);
        assert_eq!(segment.content, "Thinking...");
        assert_eq!(segment.segment_index, 0);
    }

    #[test]
    fn test_reasoning_segment_to_anthropic_format() {
        let segment = ReasoningSegment::with_index("Thinking...", 0);
        let format = segment.to_provider_format(Provider::Anthropic);

        assert_eq!(format["type"], "thinking");
        assert_eq!(format["thinking"], "Thinking...");
    }
}
