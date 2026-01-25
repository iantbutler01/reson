//! Google Anthropic (Vertex AI with Claude) client implementation
//!
//! This client allows running Anthropic Claude models through Google Cloud's
//! Vertex AI platform using Application Default Credentials (ADC).
//!
//! Uses the same request/response format as direct Anthropic API, but with:
//! - Vertex AI endpoint: `https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:streamRawPredict`
//! - ADC/OAuth2 authentication instead of API key
//! - `anthropic_version: "vertex-2023-10-16"` header in request body
//!
//! Requires the `google-adc` feature to be enabled.

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use reqwest::StatusCode;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Error, Result};
use crate::providers::{
    GenerationConfig, GenerationResponse, InferenceClient, StreamChunk, TraceCallback,
};
use crate::retry::{retry_with_backoff, RetryConfig};
use crate::types::{CacheMarker, ChatRole, Provider, TokenUsage};
use crate::utils::{
    convert_messages_to_provider_format, parse_json_value_strict_str, ConversationMessage,
};

/// Google Anthropic (Vertex AI with Claude) client
///
/// This client runs Claude models via Google Cloud's Vertex AI platform.
/// Authentication is done via Application Default Credentials (ADC).
#[cfg(feature = "google-adc")]
#[derive(Clone)]
pub struct GoogleAnthropicClient {
    model: String,
    project_id: String,
    region: String,
    token_provider: Arc<RwLock<Option<Arc<dyn gcp_auth::TokenProvider>>>>,
    thinking_budget: Option<u32>,
    trace_callback: Option<TraceCallback>,
}

#[cfg(feature = "google-adc")]
impl GoogleAnthropicClient {
    /// Create a new Google Anthropic client with Application Default Credentials (ADC)
    ///
    /// The project ID is automatically extracted from the service account JSON file
    /// referenced by `GOOGLE_APPLICATION_CREDENTIALS`.
    ///
    /// # Arguments
    /// * `model` - The Claude model name (e.g., "claude-3-5-sonnet-v2@20241022", "claude-sonnet-4@20250514")
    /// * `region` - The Google Cloud region (e.g., "us-east5", "europe-west1")
    ///
    /// # Panics
    /// Panics if `GOOGLE_APPLICATION_CREDENTIALS` is not set or if the credentials
    /// file cannot be read or doesn't contain a `project_id`.
    pub fn from_adc(model: impl Into<String>, region: impl Into<String>) -> Self {
        // Read project_id from the service account JSON file
        let creds_path = std::env::var("GOOGLE_APPLICATION_CREDENTIALS")
            .expect("GOOGLE_APPLICATION_CREDENTIALS environment variable must be set");

        let creds_content = std::fs::read_to_string(&creds_path)
            .unwrap_or_else(|_| panic!("Failed to read credentials file: {}", creds_path));

        let creds_json: serde_json::Value = parse_json_value_strict_str(&creds_content)
            .expect("Failed to parse credentials file as JSON");

        let project_id = creds_json["project_id"]
            .as_str()
            .expect("Credentials file must contain project_id")
            .to_string();

        Self {
            model: model.into(),
            project_id,
            region: region.into(),
            token_provider: Arc::new(RwLock::new(None)),
            thinking_budget: None,
            trace_callback: None,
        }
    }

    /// Create a new Google Anthropic client with explicit project ID
    ///
    /// Use this if you want to override the project_id from the credentials file.
    ///
    /// # Arguments
    /// * `model` - The Claude model name
    /// * `project_id` - The Google Cloud project ID
    /// * `region` - The Google Cloud region
    pub fn from_adc_with_project(
        model: impl Into<String>,
        project_id: impl Into<String>,
        region: impl Into<String>,
    ) -> Self {
        Self {
            model: model.into(),
            project_id: project_id.into(),
            region: region.into(),
            token_provider: Arc::new(RwLock::new(None)),
            thinking_budget: None,
            trace_callback: None,
        }
    }

    /// Set the thinking budget for extended thinking mode
    pub fn with_thinking_budget(mut self, budget: u32) -> Self {
        self.thinking_budget = Some(budget);
        self
    }

    /// Get the Vertex AI endpoint URL for this model
    fn get_endpoint_url(&self) -> String {
        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/anthropic/models/{}:streamRawPredict",
            self.region, self.project_id, self.region, self.model
        )
    }

    /// Get a valid access token for ADC authentication
    async fn get_token(&self) -> Result<String> {
        let mut provider = self.token_provider.write().await;

        // Initialize if needed
        if provider.is_none() {
            let tp = gcp_auth::provider()
                .await
                .map_err(|e| Error::NonRetryable(format!("Failed to initialize ADC: {}", e)))?;
            *provider = Some(tp);
        }

        // Get token with cloud-platform scope
        let tp = provider.as_ref().unwrap();
        let scopes = &["https://www.googleapis.com/auth/cloud-platform"];
        let token = tp
            .token(scopes)
            .await
            .map_err(|e| Error::NonRetryable(format!("Failed to get ADC token: {}", e)))?;

        Ok(token.as_str().to_string())
    }

    /// Build the request body for Vertex AI Anthropic endpoint
    ///
    /// Uses Anthropic request format but with:
    /// - No "model" field (model is in the URL)
    /// - `anthropic_version: "vertex-2023-10-16"` instead of API version header
    fn build_request_body(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
        stream: bool,
    ) -> Result<serde_json::Value> {
        // Extract system message if present
        let (system, messages) = self.extract_system_message(messages)?;

        // Convert messages to Anthropic format
        let formatted_messages =
            convert_messages_to_provider_format(messages, Provider::Anthropic)?;

        // Wrap string content in proper format
        let formatted_messages = self.wrap_string_content(formatted_messages);

        // Use config.model if provided, otherwise use self.model
        let _model = if config.model.is_empty() {
            &self.model
        } else {
            &config.model
        };

        let mut request = serde_json::json!({
            // Note: model is NOT included - it's in the URL for Vertex AI
            "anthropic_version": "vertex-2023-10-16",
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

    /// Make HTTP request to Vertex AI endpoint
    async fn make_request(&self, body: serde_json::Value) -> Result<reqwest::Response> {
        let token = self.get_token().await?;
        let client = reqwest::Client::new();

        let response = client
            .post(self.get_endpoint_url())
            .timeout(std::time::Duration::from_secs(300))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
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
    #[cfg(feature = "google-adc")]
    async fn make_request_with_retry(&self, body: serde_json::Value) -> Result<serde_json::Value> {
        let config = RetryConfig::default();

        retry_with_backoff(config, || async {
            let response = self.make_request(body.clone()).await?;
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

#[cfg(feature = "google-adc")]
#[async_trait]
impl InferenceClient for GoogleAnthropicClient {
    async fn get_generation(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<GenerationResponse> {
        let request_body = self.build_request_body(messages, config, false)?;
        let body = self.make_request_with_retry(request_body).await?;

        // Parse usage statistics
        let usage = self.parse_usage(&body["usage"]);

        // Extract content
        let content_value = &body["content"];
        let text_content = self.extract_text_content(content_value);
        let tool_calls = self.extract_tool_calls(content_value);

        // If tools were provided, return full response for tool extraction
        let has_tools = config.tools.is_some() && !config.tools.as_ref().unwrap().is_empty();
        let has_tool_calls = !tool_calls.is_empty();

        Ok(GenerationResponse {
            content: text_content,
            reasoning: None,
            tool_calls,
            reasoning_segments: Vec::new(),
            usage,
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

        let request_body = self.build_request_body(messages, config, true)?;

        // Retry the connection establishment with backoff
        let retry_config = RetryConfig::default();
        let response = retry_with_backoff(retry_config, || async {
            let resp = self.make_request(request_body.clone()).await?;
            let status = resp.status();

            if !status.is_success() {
                let error_body = resp.text().await.unwrap_or_default();
                return Err(self.handle_error_response(status, error_body));
            }

            Ok(resp)
        })
        .await?;

        let has_tools = config.tools.is_some() && !config.tools.as_ref().unwrap().is_empty();

        // Parse SSE stream - reuse Anthropic's SSE parsing since format is identical
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
        Provider::GoogleAnthropic
    }

    fn set_trace_callback(&mut self, callback: TraceCallback) {
        self.trace_callback = Some(callback);
    }
}

#[cfg(all(test, feature = "google-adc"))]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_url() {
        let client = GoogleAnthropicClient::from_adc_with_project(
            "claude-3-5-sonnet-v2@20241022",
            "my-project",
            "us-east5",
        );

        assert_eq!(
            client.get_endpoint_url(),
            "https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-3-5-sonnet-v2@20241022:streamRawPredict"
        );
    }

    #[test]
    fn test_with_thinking_budget() {
        let client = GoogleAnthropicClient::from_adc_with_project(
            "claude-3-5-sonnet-v2@20241022",
            "my-project",
            "us-east5",
        )
        .with_thinking_budget(1024);

        assert_eq!(client.thinking_budget, Some(1024));
    }
}
