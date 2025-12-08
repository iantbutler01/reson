//! LLM provider clients
//!
//! This module contains the InferenceClient trait and implementations
//! for various LLM providers (Anthropic, OpenAI, Google, etc.).

use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;

use crate::error::Result;
use crate::types::TokenUsage;
use crate::utils::ConversationMessage;

// Re-export Provider from types for convenience
pub use crate::types::Provider;

// Provider implementations
pub mod anthropic;
pub(crate) mod anthropic_streaming;
pub mod bedrock;
pub mod google;
#[cfg(feature = "google-adc")]
pub mod google_anthropic;
pub mod openai;
pub(crate) mod openai_streaming;
pub mod openrouter;
pub mod tracing_client;

pub use anthropic::AnthropicClient;
pub use bedrock::BedrockClient;
pub use google::{GoogleGenAIClient, FileState, UploadedFile};
#[cfg(feature = "google-adc")]
pub use google_anthropic::GoogleAnthropicClient;
pub use openai::OAIClient;
pub use openrouter::OpenRouterClient;
pub use tracing_client::TracingInferenceClient;

/// Configuration for generation requests
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Model name (provider-specific)
    pub model: String,

    /// Maximum tokens to generate
    pub max_tokens: Option<u32>,

    /// Temperature (0.0-1.0)
    pub temperature: Option<f32>,

    /// Top-p sampling
    pub top_p: Option<f32>,

    /// Tool schemas (provider-specific format)
    pub tools: Option<Vec<serde_json::Value>>,

    /// Whether to use native tool calling
    pub native_tools: bool,

    /// Reasoning effort (for o-series models)
    pub reasoning_effort: Option<String>,

    /// Thinking budget tokens (for Claude extended thinking)
    pub thinking_budget: Option<u32>,

    /// Output schema for structured outputs (JSON schema)
    pub output_schema: Option<serde_json::Value>,

    /// Name for the output type (used in schema wrapper)
    pub output_type_name: Option<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            model: String::new(),
            max_tokens: Some(4096),
            temperature: None,
            top_p: None,
            tools: None,
            native_tools: false,
            reasoning_effort: None,
            thinking_budget: None,
            output_schema: None,
            output_type_name: None,
        }
    }
}

impl GenerationConfig {
    /// Create a new configuration with required model
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set tools
    pub fn with_tools(mut self, tools: Vec<serde_json::Value>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Enable native tools
    pub fn with_native_tools(mut self, enabled: bool) -> Self {
        self.native_tools = enabled;
        self
    }

    /// Set reasoning effort (for o-series models)
    pub fn with_reasoning_effort(mut self, effort: impl Into<String>) -> Self {
        self.reasoning_effort = Some(effort.into());
        self
    }

    /// Set thinking budget (for Claude extended thinking)
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Set output schema for structured outputs
    pub fn with_output_schema(mut self, schema: serde_json::Value, type_name: impl Into<String>) -> Self {
        self.output_schema = Some(schema);
        self.output_type_name = Some(type_name.into());
        self
    }
}

/// Response from a generation request
#[derive(Debug, Clone)]
pub struct GenerationResponse {
    /// Generated content (text)
    pub content: String,

    /// Extended reasoning content (if any)
    pub reasoning: Option<String>,

    /// Tool calls made by the model
    pub tool_calls: Vec<serde_json::Value>,

    /// Reasoning segments
    pub reasoning_segments: Vec<serde_json::Value>,

    /// Token usage statistics
    pub usage: TokenUsage,

    /// Raw response (for debugging/tool extraction)
    pub raw: Option<serde_json::Value>,
}

impl GenerationResponse {
    /// Create a simple text response
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            reasoning_segments: Vec::new(),
            usage: TokenUsage::default(),
            raw: None,
        }
    }

    /// Check if response contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

/// Stream chunk types
#[derive(Debug, Clone)]
pub enum StreamChunk {
    /// Reasoning/thinking content
    Reasoning(String),

    /// Provider signature for reasoning preservation
    Signature(String),

    /// Main content chunk
    Content(String),

    /// Partial tool call (incomplete JSON)
    ToolCallPartial(serde_json::Value),

    /// Complete tool call
    ToolCallComplete(serde_json::Value),

    /// Token usage statistics (final message)
    Usage {
        input_tokens: u64,
        output_tokens: u64,
        cached_tokens: u64,
    },
}

/// Trace callback type for monitoring (wrapped in Arc for Clone support)
pub type TraceCallback = std::sync::Arc<
    dyn Fn(u64, Vec<serde_json::Value>, &GenerationResponse) -> Pin<Box<dyn futures::Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

/// Core trait for LLM provider clients
#[async_trait]
pub trait InferenceClient: Send + Sync {
    /// Get a single generation (non-streaming)
    async fn get_generation(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<GenerationResponse>;

    /// Connect and listen for streaming responses
    async fn connect_and_listen(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;

    /// Get provider type
    fn provider(&self) -> Provider;

    /// Check if native tools are supported
    fn supports_native_tools(&self) -> bool {
        self.provider().supports_native_tools()
    }

    /// Set trace callback (optional)
    fn set_trace_callback(&mut self, _callback: TraceCallback) {
        // Default: no-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_config_builder() {
        let config = GenerationConfig::new("gpt-4")
            .with_max_tokens(2048)
            .with_temperature(0.5)
            .with_native_tools(true);

        assert_eq!(config.model, "gpt-4");
        assert_eq!(config.max_tokens, Some(2048));
        assert_eq!(config.temperature, Some(0.5));
        assert!(config.native_tools);
    }

    #[test]
    fn test_generation_response_text() {
        let response = GenerationResponse::text("Hello, world!");
        assert_eq!(response.content, "Hello, world!");
        assert!(!response.has_tool_calls());
    }

    #[test]
    fn test_generation_response_with_tool_calls() {
        let mut response = GenerationResponse::text("Using tools...");
        response.tool_calls.push(serde_json::json!({"name": "get_weather"}));
        assert!(response.has_tool_calls());
    }
}
