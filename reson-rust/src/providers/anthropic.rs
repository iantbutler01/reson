//! Anthropic Claude API client
//!
//! Implements the InferenceClient trait for Anthropic's Claude models.
//! Supports:
//! - Native tool calling with parallel execution
//! - Prompt caching with ephemeral markers
//! - Extended thinking mode
//! - Streaming with SSE

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use reqwest::StatusCode;
use std::pin::Pin;

use crate::error::{Error, Result};
use crate::providers::{
    GenerationConfig, GenerationResponse, InferenceClient, StreamChunk, TraceCallback,
};
use crate::retry::{retry_with_backoff, RetryConfig};
use crate::types::{CacheMarker, ChatRole, Provider, TokenUsage};
use crate::utils::{convert_messages_to_provider_format, ConversationMessage};

/// Anthropic API client
#[derive(Clone)]
pub struct AnthropicClient {
    model: String,
    api_key: String,
    api_url: String,
    thinking_budget: Option<u32>,
    trace_callback: Option<TraceCallback>,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            api_url: "https://api.anthropic.com/v1/messages".to_string(),
            thinking_budget: None,
            trace_callback: None,
        }
    }

    /// Set the thinking budget for extended thinking mode
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Set a custom API URL
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Build the request body for Anthropic API
    fn build_request_body(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
        stream: bool,
    ) -> Result<serde_json::Value> {
        // Extract system message if present
        let (system, messages) = self.extract_system_message(messages)?;

        // Convert messages to provider format
        let formatted_messages =
            convert_messages_to_provider_format(messages, Provider::Anthropic)?;

        // Wrap string content in proper format
        let formatted_messages = self.wrap_string_content(formatted_messages);

        let mut request = serde_json::json!({
            "model": if config.model.is_empty() { &self.model } else { &config.model },
            "max_tokens": config.max_tokens.unwrap_or(4096),
            "messages": formatted_messages,
            "stream": stream,
        });

        // Add temperature and top_p if not in thinking mode
        if self.thinking_budget.is_none() {
            if let Some(temp) = config.temperature {
                request["temperature"] = serde_json::json!(temp);
            }
            if let Some(top_p) = config.top_p {
                request["top_p"] = serde_json::json!(top_p);
            }
        }

        // Add system if present
        if let Some(system) = system {
            request["system"] = system;
        }

        // Add tools if provided
        if let Some(ref tools) = config.tools {
            if !tools.is_empty() {
                request["tools"] = serde_json::json!(tools);
                // Enable parallel tool calling via tool_choice
                request["tool_choice"] = serde_json::json!({
                    "type": "auto",
                    "disable_parallel_tool_use": false
                });
            }
        }

        // Add extended thinking if configured
        if let Some(budget) = self.thinking_budget {
            request["thinking"] = serde_json::json!({
                "type": "enabled",
                "budget_tokens": budget
            });
            // Adjust max_tokens and temperature for thinking mode
            let current_max = request["max_tokens"].as_u64().unwrap_or(4096);
            request["max_tokens"] = serde_json::json!(current_max + budget as u64);
            request["temperature"] = serde_json::json!(1.0);
            if let Some(obj) = request.as_object_mut() {
                obj.remove("top_p");
            }
        }

        // Add structured output schema if provided
        if let Some(ref schema) = config.output_schema {
            request["output_format"] = serde_json::json!({
                "type": "json_schema",
                "schema": schema
            });
        }

        Ok(request)
    }

    /// Extract system message and return (system, remaining_messages)
    fn extract_system_message<'a>(
        &self,
        messages: &'a [ConversationMessage],
    ) -> Result<(Option<serde_json::Value>, &'a [ConversationMessage])> {
        if let Some(ConversationMessage::Chat(first)) = messages.first() {
            if first.role == ChatRole::System {
                let mut system_dict = serde_json::json!({
                    "type": "text",
                    "text": first.content
                });

                // Add cache control if marked
                if first.cache_marker == Some(CacheMarker::Ephemeral) {
                    system_dict["cache_control"] = serde_json::json!({"type": "ephemeral"});
                }

                return Ok((Some(serde_json::json!([system_dict])), &messages[1..]));
            }
        }
        Ok((None, messages))
    }

    /// Wrap string content in [{"type": "text", "text": "..."}] format
    /// Skips empty strings to avoid Anthropic's "text content blocks must be non-empty" error
    fn wrap_string_content(&self, mut messages: Vec<serde_json::Value>) -> Vec<serde_json::Value> {
        for msg in messages.iter_mut() {
            if let Some(content) = msg.get("content") {
                if let Some(text) = content.as_str() {
                    if !text.is_empty() {
                        msg["content"] = serde_json::json!([{
                            "type": "text",
                            "text": text
                        }]);
                    } else {
                        // Empty content - convert to empty array
                        // This can happen for assistant messages that only have tool calls
                        msg["content"] = serde_json::json!([]);
                    }
                }
            }
        }
        messages
    }

    /// Extract text content from response
    fn extract_text_content(&self, content: &serde_json::Value) -> String {
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                if block["type"] == "text" {
                    if let Some(text) = block["text"].as_str() {
                        return text.to_string();
                    }
                }
            }
        }
        String::new()
    }

    /// Extract reasoning/thinking content from response
    fn extract_reasoning(&self, content: &serde_json::Value) -> Option<String> {
        if let Some(blocks) = content.as_array() {
            let thinking: Vec<String> = blocks
                .iter()
                .filter(|block| block["type"] == "thinking")
                .filter_map(|block| block["thinking"].as_str().map(|s| s.to_string()))
                .collect();

            if !thinking.is_empty() {
                return Some(thinking.join("\n"));
            }
        }
        None
    }

    /// Extract tool calls from response content
    fn extract_tool_calls(&self, content: &serde_json::Value) -> Vec<serde_json::Value> {
        let mut tool_calls = Vec::new();
        if let Some(blocks) = content.as_array() {
            for block in blocks {
                if block["type"] == "tool_use" {
                    tool_calls.push(block.clone());
                }
            }
        }
        tool_calls
    }

    /// Parse token usage from response
    fn parse_usage(&self, usage: &serde_json::Value) -> TokenUsage {
        TokenUsage {
            input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
            output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
            cached_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        }
    }

    /// Make HTTP request to Anthropic API
    async fn make_request(
        &self,
        body: serde_json::Value,
        use_structured_outputs: bool,
        timeout: Option<std::time::Duration>,
    ) -> Result<reqwest::Response> {
        let client = reqwest::Client::new();

        // Build beta header - add structured-outputs if needed
        let beta_header = if use_structured_outputs {
            "prompt-caching-2024-07-31,output-128k-2025-02-19,structured-outputs-2025-11-13"
        } else {
            "prompt-caching-2024-07-31,output-128k-2025-02-19"
        };

        let response = client
            .post(&self.api_url)
            .timeout(timeout.unwrap_or(std::time::Duration::from_secs(300)))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", beta_header)
            .json(&body)
            .send()
            .await?;

        Ok(response)
    }

    /// Handle error responses - categorize as retryable or non-retryable
    fn handle_error_response(&self, status: StatusCode, body: String) -> Error {
        match status {
            // Client errors (4xx) are generally not retryable
            StatusCode::BAD_REQUEST | StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Error::NonRetryable(format!("{}: {}", status, body))
            }
            // Rate limit - retryable
            StatusCode::TOO_MANY_REQUESTS => Error::Inference(format!("Rate limited: {}", body)),
            // Server errors (5xx) are retryable
            StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT => Error::Inference(format!("{}: {}", status, body)),
            // Default: assume retryable for unknown errors
            _ => Error::Inference(format!("{}: {}", status, body)),
        }
    }

    /// Make request with retry and exponential backoff
    async fn make_request_with_retry(
        &self,
        body: serde_json::Value,
        use_structured_outputs: bool,
        timeout: Option<std::time::Duration>,
    ) -> Result<serde_json::Value> {
        let config = RetryConfig::default();

        retry_with_backoff(config, || async {
            let response = self
                .make_request(body.clone(), use_structured_outputs, timeout)
                .await?;
            let status = response.status();

            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();
                return Err(self.handle_error_response(status, error_body));
            }

            response.json().await.map_err(Error::from)
        })
        .await
    }
}

#[async_trait]
impl InferenceClient for AnthropicClient {
    async fn get_generation(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<GenerationResponse> {
        let use_structured_outputs = config.output_schema.is_some();
        let request_body = self.build_request_body(messages, config, false)?;
        let body = self
            .make_request_with_retry(request_body, use_structured_outputs, config.timeout)
            .await?;

        // Parse usage statistics
        let usage = self.parse_usage(&body["usage"]);

        // Extract content
        let content_value = &body["content"];
        let text_content = self.extract_text_content(content_value);
        let reasoning = self.extract_reasoning(content_value);
        let tool_calls = self.extract_tool_calls(content_value);

        // If tools were provided, return full response for tool extraction
        let has_tools = config.tools.is_some() && !config.tools.as_ref().unwrap().is_empty();
        let has_tool_calls = !tool_calls.is_empty();

        Ok(GenerationResponse {
            content: text_content,
            reasoning,
            tool_calls,
            reasoning_segments: Vec::new(),
            usage,
            provider_cost_dollars: None,
            raw: if has_tools || has_tool_calls {
                Some(body)
            } else {
                None
            },
        })
    }

    async fn connect_and_listen(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        use crate::providers::anthropic_streaming::{parse_anthropic_chunk, ToolCallAccumulator};
        use crate::utils::parse_sse_stream;

        let use_structured_outputs = config.output_schema.is_some();
        let request_body = self.build_request_body(messages, config, true)?;
        let timeout = config.timeout;

        // Retry the connection establishment with backoff
        let retry_config = RetryConfig::default();
        let response = retry_with_backoff(retry_config, || async {
            let resp = self
                .make_request(request_body.clone(), use_structured_outputs, timeout)
                .await?;
            let status = resp.status();

            if !status.is_success() {
                let error_body = resp.text().await.unwrap_or_default();
                return Err(self.handle_error_response(status, error_body));
            }

            Ok(resp)
        })
        .await?;

        let has_tools = config.tools.is_some() && !config.tools.as_ref().unwrap().is_empty();

        // Parse SSE stream
        let sse_stream = parse_sse_stream(response);

        // Process chunks with tool accumulator
        let chunk_stream = sse_stream.scan(
            ToolCallAccumulator::new(),
            move |accumulator, sse_result| {
                let sse_json = match sse_result {
                    Ok(json) => json,
                    Err(e) => return futures::future::ready(Some(vec![Err(e)])),
                };

                // Parse chunk and emit StreamChunks
                let chunks = parse_anthropic_chunk(&sse_json, accumulator, has_tools);
                futures::future::ready(Some(chunks.into_iter().map(Ok).collect()))
            },
        );

        // Flatten the Vec<Result<StreamChunk>> into individual items
        Ok(Box::pin(chunk_stream.flat_map(futures::stream::iter)))
    }

    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    fn set_trace_callback(&mut self, callback: TraceCallback) {
        self.trace_callback = Some(callback);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[test]
    fn test_client_creation() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        assert_eq!(client.model, "claude-3-opus-20240229");
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.api_url, "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn test_with_thinking_budget() {
        let client =
            AnthropicClient::new("test-key", "claude-3-opus-20240229").with_thinking_budget(1024);
        assert_eq!(client.thinking_budget, Some(1024));
    }

    #[test]
    fn test_extract_system_message() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let messages = vec![
            ConversationMessage::Chat(ChatMessage::system("You are helpful")),
            ConversationMessage::Chat(ChatMessage::user("Hello")),
        ];

        let (system, remaining) = client.extract_system_message(&messages).unwrap();
        assert!(system.is_some());
        assert_eq!(remaining.len(), 1);

        let system_val = system.unwrap();
        assert_eq!(system_val[0]["type"], "text");
        assert_eq!(system_val[0]["text"], "You are helpful");
    }

    #[test]
    fn test_extract_system_with_cache_marker() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let messages = vec![ConversationMessage::Chat(
            ChatMessage::system("Long context").with_cache_marker(CacheMarker::Ephemeral),
        )];

        let (system, _) = client.extract_system_message(&messages).unwrap();
        let system_val = system.unwrap();
        assert!(system_val[0]["cache_control"].is_object());
        assert_eq!(system_val[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn test_wrap_string_content() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let messages = vec![serde_json::json!({
            "role": "user",
            "content": "Hello"
        })];

        let wrapped = client.wrap_string_content(messages);
        assert!(wrapped[0]["content"].is_array());
        assert_eq!(wrapped[0]["content"][0]["type"], "text");
        assert_eq!(wrapped[0]["content"][0]["text"], "Hello");
    }

    #[test]
    fn test_extract_text_content() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let content = serde_json::json!([
            {"type": "text", "text": "Hello, world!"}
        ]);

        let text = client.extract_text_content(&content);
        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_extract_tool_calls() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let content = serde_json::json!([
            {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"city": "SF"}},
            {"type": "text", "text": "Using tool..."}
        ]);

        let tool_calls = client.extract_tool_calls(&content);
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "toolu_123");
        assert_eq!(tool_calls[0]["name"], "get_weather");
    }

    #[test]
    fn test_parse_usage() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let usage = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 25,
            "cache_creation_input_tokens": 10
        });

        let parsed = client.parse_usage(&usage);
        assert_eq!(parsed.input_tokens, 100);
        assert_eq!(parsed.output_tokens, 50);
        assert_eq!(parsed.cached_tokens, 25);
    }

    #[test]
    fn test_build_request_body_basic() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
        let config = GenerationConfig::new("claude-3-opus-20240229")
            .with_max_tokens(1024)
            .with_temperature(0.7);

        let body = client
            .build_request_body(&messages, &config, false)
            .unwrap();

        assert_eq!(body["model"], "claude-3-opus-20240229");
        assert_eq!(body["max_tokens"], 1024);
        assert!((body["temperature"].as_f64().unwrap() - 0.7).abs() < 0.01);
        assert_eq!(body["stream"], false);
        assert!(body["messages"].is_array());
    }

    #[test]
    fn test_build_request_with_tools() {
        let client = AnthropicClient::new("test-key", "claude-3-opus-20240229");
        let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
        let tools = vec![serde_json::json!({"name": "get_weather"})];
        let config = GenerationConfig::new("claude-3-opus-20240229").with_tools(tools);

        let body = client
            .build_request_body(&messages, &config, false)
            .unwrap();

        assert!(body["tools"].is_array());
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], false);
    }

    #[test]
    fn test_build_request_with_thinking() {
        let client =
            AnthropicClient::new("test-key", "claude-3-opus-20240229").with_thinking_budget(1024);
        let messages = vec![ConversationMessage::Chat(ChatMessage::user("Think"))];
        let config = GenerationConfig::new("claude-3-opus-20240229").with_max_tokens(2048);

        let body = client
            .build_request_body(&messages, &config, false)
            .unwrap();

        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["budget_tokens"], 1024);
        assert_eq!(body["max_tokens"], 3072); // 2048 + 1024
        assert_eq!(body["temperature"], 1.0);
        assert!(body.get("top_p").is_none());
    }
}
