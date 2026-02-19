//! OpenAI Responses API client
//!
//! Implements the InferenceClient trait for OpenAI's Responses API.
//! Supports:
//! - Function tool calling (Responses format)
//! - Reasoning configuration (o-series models)
//! - Streaming via SSE events
//! - Usage tracking

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use reqwest::StatusCode;
use std::pin::Pin;

use crate::error::{Error, Result};
use crate::providers::{
    GenerationConfig, GenerationResponse, InferenceClient, StreamChunk, TraceCallback,
};
use crate::retry::{retry_with_backoff, RetryConfig};
use crate::types::{Provider, TokenUsage};
use crate::utils::{
    convert_messages_to_responses_input, parse_json_value_strict_str, parse_sse_stream,
    ConversationMessage,
};

use super::openai_responses_streaming::{parse_openai_responses_event, ResponsesToolAccumulator};

/// OpenAI Responses API client (also serves as base for OpenRouter Responses)
pub struct OpenAIResponsesClient {
    model: String,
    api_key: String,
    api_url: String,
    reasoning: Option<String>,
    ranking_referer: Option<String>,
    ranking_title: Option<String>,
    trace_callback: Option<TraceCallback>,
    provider: Provider,
}

impl Clone for OpenAIResponsesClient {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            api_url: self.api_url.clone(),
            reasoning: self.reasoning.clone(),
            ranking_referer: self.ranking_referer.clone(),
            ranking_title: self.ranking_title.clone(),
            trace_callback: self.trace_callback.clone(),
            provider: self.provider,
        }
    }
}

impl std::fmt::Debug for OpenAIResponsesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesClient")
            .field("model", &self.model)
            .field("api_url", &self.api_url)
            .field("reasoning", &self.reasoning)
            .field("ranking_referer", &self.ranking_referer)
            .field("ranking_title", &self.ranking_title)
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl OpenAIResponsesClient {
    /// Create a new OpenAI Responses client
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            api_key: api_key.into(),
            api_url: "https://api.openai.com/v1/responses".to_string(),
            reasoning: None,
            ranking_referer: None,
            ranking_title: None,
            trace_callback: None,
            provider: Provider::OpenAIResponses,
        }
    }

    /// Set reasoning mode (for o-series models)
    /// - Numeric string (e.g., "1024") → max_tokens
    /// - Text string (e.g., "medium", "high") → effort level
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Set custom API URL (used by OpenRouter Responses)
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Set ranking headers (for OpenRouter)
    pub fn with_ranking_headers(mut self, referer: Option<String>, title: Option<String>) -> Self {
        self.ranking_referer = referer;
        self.ranking_title = title;
        self
    }

    /// Set provider type (used by OpenRouter subclass)
    pub(crate) fn with_provider(mut self, provider: Provider) -> Self {
        self.provider = provider;
        self
    }

    /// Build request body for Responses API
    fn build_request_body(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
        stream: bool,
    ) -> Result<serde_json::Value> {
        let (instructions, input_items) =
            convert_messages_to_responses_input(messages, self.provider)?;

        let mut request = serde_json::json!({
            "model": if config.model.is_empty() { &self.model } else { &config.model },
            "input": input_items,
            "max_output_tokens": config.max_tokens.unwrap_or(4096),
            "temperature": config.temperature.unwrap_or(0.7),
            "top_p": config.top_p.unwrap_or(1.0),
            "stream": stream,
        });

        if let Some(instructions) = instructions {
            if !instructions.is_empty() {
                request["instructions"] = serde_json::json!(instructions);
            }
        }

        if let Some(ref tools) = config.tools {
            if !tools.is_empty() {
                request["tools"] = serde_json::json!(tools);
                request["tool_choice"] = serde_json::json!("auto");
            }
        }

        if let Some(ref reasoning) = self.reasoning {
            if reasoning.chars().all(|c| c.is_ascii_digit()) {
                request["reasoning"] = serde_json::json!({
                    "max_tokens": reasoning.parse::<u32>().unwrap_or(1024)
                });
            } else {
                request["reasoning"] = serde_json::json!({
                    "effort": reasoning
                });
            }
        }

        if let Some(ref schema) = config.output_schema {
            let type_name = config.output_type_name.as_deref().unwrap_or("response");
            request["text"] = serde_json::json!({
                "format": {
                    "type": "json_schema",
                    "name": type_name,
                    "schema": schema,
                    "strict": true
                }
            });
        }

        Ok(request)
    }

    fn parse_usage(&self, usage: &serde_json::Value) -> TokenUsage {
        TokenUsage {
            input_tokens: usage
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            output_tokens: usage
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            cached_tokens: usage
                .get("input_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        }
    }

    fn extract_output_text(&self, body: &serde_json::Value) -> String {
        let mut content = String::new();
        if let Some(output) = body.get("output").and_then(|v| v.as_array()) {
            for item in output {
                if item.get("type").and_then(|v| v.as_str()) == Some("message")
                    && item.get("role").and_then(|v| v.as_str()) == Some("assistant")
                {
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            if part.get("type").and_then(|v| v.as_str()) == Some("output_text") {
                                if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                    content.push_str(text);
                                }
                            }
                        }
                    }
                }
            }
        }
        content
    }

    fn extract_reasoning(&self, body: &serde_json::Value) -> Option<String> {
        let mut reasoning = String::new();
        if let Some(output) = body.get("output").and_then(|v| v.as_array()) {
            for item in output {
                if item.get("type").and_then(|v| v.as_str()) == Some("reasoning") {
                    if let Some(summary) = item.get("summary") {
                        if let Some(arr) = summary.as_array() {
                            for part in arr {
                                if let Some(text) = part.as_str() {
                                    reasoning.push_str(text);
                                }
                            }
                        } else if let Some(text) = summary.as_str() {
                            reasoning.push_str(text);
                        }
                    }
                    if let Some(parts) = item.get("content").and_then(|v| v.as_array()) {
                        for part in parts {
                            if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                                reasoning.push_str(text);
                            }
                        }
                    }
                }
            }
        }
        if reasoning.is_empty() {
            None
        } else {
            Some(reasoning)
        }
    }

    fn extract_tool_calls(&self, body: &serde_json::Value) -> Vec<serde_json::Value> {
        let mut tool_calls = Vec::new();
        if let Some(output) = body.get("output").and_then(|v| v.as_array()) {
            for item in output {
                if item.get("type").and_then(|v| v.as_str()) == Some("function_call") {
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let call_id = item
                        .get("call_id")
                        .and_then(|v| v.as_str())
                        .or_else(|| item.get("id").and_then(|v| v.as_str()))
                        .unwrap_or("");
                    let args = item
                        .get("arguments")
                        .map(|v| match v {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        })
                        .unwrap_or_else(|| "{}".to_string());
                    tool_calls.push(serde_json::json!({
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": args
                        }
                    }));
                }
            }
        }
        tool_calls
    }

    async fn make_request(
        &self,
        body: serde_json::Value,
        timeout: Option<std::time::Duration>,
    ) -> Result<reqwest::Response> {
        let client = reqwest::Client::new();
        let mut req = client
            .post(&self.api_url)
            .timeout(timeout.unwrap_or(std::time::Duration::from_secs(180)))
            .header("Content-Type", "application/json")
            .json(&body);

        if !self.api_key.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", self.api_key));
        }

        if let Some(ref referer) = self.ranking_referer {
            req = req.header("HTTP-Referer", referer);
        }
        if let Some(ref title) = self.ranking_title {
            req = req.header("X-Title", title);
        }

        let response = req.send().await?;
        Ok(response)
    }

    fn handle_error_response(&self, status: StatusCode, body: String) -> Error {
        match status {
            StatusCode::BAD_REQUEST | StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Error::NonRetryable(format!("{}: {}", status, body))
            }
            StatusCode::TOO_MANY_REQUESTS => Error::Inference(format!("Rate limited: {}", body)),
            StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT => Error::Inference(format!("{}: {}", status, body)),
            _ => Error::Inference(format!("{}: {}", status, body)),
        }
    }

    async fn make_request_with_retry(
        &self,
        body: serde_json::Value,
        timeout: Option<std::time::Duration>,
    ) -> Result<String> {
        let config = RetryConfig::default();

        retry_with_backoff(config, || async {
            let response = self.make_request(body.clone(), timeout).await?;
            let status = response.status();
            let response_text = response.text().await.unwrap_or_default();

            if !status.is_success() {
                return Err(self.handle_error_response(status, response_text));
            }

            Ok(response_text)
        })
        .await
    }
}

#[async_trait]
impl InferenceClient for OpenAIResponsesClient {
    async fn get_generation(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<GenerationResponse> {
        let request_body = self.build_request_body(messages, config, false)?;
        let response_text = self
            .make_request_with_retry(request_body, config.timeout)
            .await?;

        let body: serde_json::Value = parse_json_value_strict_str(&response_text).map_err(|e| {
            Error::Inference(format!(
                "Failed to parse response as JSON: {}. Response: {}",
                e,
                if response_text.len() > 500 {
                    &response_text[..500]
                } else {
                    &response_text
                }
            ))
        })?;

        if let Some(error) = body.get("error").filter(|e| !e.is_null()) {
            let error_msg = error["message"]
                .as_str()
                .unwrap_or("Unknown error")
                .to_string();
            return Err(Error::Inference(format!("{} ({:?})", error_msg, error)));
        }

        let usage_json = body.get("usage");
        let usage = usage_json.map(|u| self.parse_usage(u)).unwrap_or_default();

        // Extract provider cost if available (OpenRouter returns usage.cost in dollars)
        let provider_cost_dollars = if matches!(
            self.provider,
            Provider::OpenRouter | Provider::OpenRouterResponses
        ) {
            usage_json
                .and_then(|u| u.get("cost"))
                .and_then(|c| c.as_f64())
        } else {
            None
        };

        let content = self.extract_output_text(&body);
        let reasoning = self.extract_reasoning(&body);
        let tool_calls = self.extract_tool_calls(&body);

        let has_tools = config.tools.is_some() && !config.tools.as_ref().unwrap().is_empty();
        let has_tool_calls = !tool_calls.is_empty();

        Ok(GenerationResponse {
            content,
            reasoning,
            tool_calls,
            reasoning_segments: Vec::new(),
            usage,
            provider_cost_dollars,
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
        let request_body = self.build_request_body(messages, config, true)?;
        let timeout = config.timeout;

        let retry_config = RetryConfig::default();
        let response = retry_with_backoff(retry_config, || async {
            let resp = self.make_request(request_body.clone(), timeout).await?;
            let status = resp.status();

            if !status.is_success() {
                let error_body = resp.text().await.unwrap_or_default();
                return Err(self.handle_error_response(status, error_body));
            }

            Ok(resp)
        })
        .await?;

        let has_tools = config.tools.is_some() && !config.tools.as_ref().unwrap().is_empty();
        let sse_stream = parse_sse_stream(response);

        let debug_sse = std::env::var("RESON_DEBUG_RESPONSES_SSE")
            .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
            .unwrap_or(false);

        let chunk_stream = sse_stream.scan(
            ResponsesToolAccumulator::new(),
            move |accumulator, sse_result| {
                let sse_json = match sse_result {
                    Ok(json) => json,
                    Err(e) => return futures::future::ready(Some(vec![Err(e)])),
                };

                if debug_sse {
                    eprintln!("responses sse event: {}", sse_json);
                }

                let chunks = parse_openai_responses_event(&sse_json, accumulator, has_tools);
                futures::future::ready(Some(chunks.into_iter().map(Ok).collect()))
            },
        );

        Ok(Box::pin(chunk_stream.flat_map(futures::stream::iter)))
    }

    fn provider(&self) -> Provider {
        self.provider
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
        let client = OpenAIResponsesClient::new("test-key", "gpt-4");
        assert_eq!(client.model, "gpt-4");
        assert_eq!(client.api_key, "test-key");
        assert_eq!(client.api_url, "https://api.openai.com/v1/responses");
        assert_eq!(client.provider, Provider::OpenAIResponses);
    }

    #[test]
    fn test_build_request_basic() {
        let client = OpenAIResponsesClient::new("test-key", "gpt-4");
        let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
        let config = GenerationConfig::new("gpt-4").with_max_tokens(2048);

        let body = client
            .build_request_body(&messages, &config, false)
            .unwrap();

        assert_eq!(body["model"], "gpt-4");
        assert_eq!(body["max_output_tokens"], 2048);
        assert_eq!(body["stream"], false);
        assert!(body["input"].is_array());
    }

    #[test]
    fn test_build_request_with_tools() {
        let client = OpenAIResponsesClient::new("test-key", "gpt-4");
        let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
        let tools = vec![serde_json::json!({"type": "function", "name": "get_weather"})];
        let config = GenerationConfig::new("gpt-4").with_tools(tools);

        let body = client
            .build_request_body(&messages, &config, false)
            .unwrap();

        assert!(body["tools"].is_array());
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_parse_usage() {
        let client = OpenAIResponsesClient::new("test-key", "gpt-4");
        let usage = serde_json::json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "input_tokens_details": {
                "cached_tokens": 25
            }
        });

        let parsed = client.parse_usage(&usage);
        assert_eq!(parsed.input_tokens, 100);
        assert_eq!(parsed.output_tokens, 50);
        assert_eq!(parsed.cached_tokens, 25);
    }

    #[test]
    fn test_extract_tool_calls() {
        let client = OpenAIResponsesClient::new("test-key", "gpt-4");
        let body = serde_json::json!({
            "output": [{
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": "{\"city\":\"SF\"}"
            }]
        });

        let calls = client.extract_tool_calls(&body);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_123");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
    }
}
