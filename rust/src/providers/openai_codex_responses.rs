//! ChatGPT Codex subscription Responses client.
//!
//! This provider targets the ChatGPT Codex backend used by subscription
//! accounts. It intentionally reuses the OpenAI Responses message/tool format
//! and streaming parser, but owns the ChatGPT-specific URL and auth headers.

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use futures::{SinkExt, StreamExt, channel::mpsc};
use http::{HeaderName, HeaderValue, Uri};
use reqwest::{StatusCode, header};
use serde_json::Value;
use std::pin::Pin;
use std::time::Duration;

use crate::error::{Error, Result};
use crate::providers::{
    CodexSubscriptionProviderConfig, CodexSubscriptionTransport, GenerationConfig,
    GenerationResponse, InferenceClient, StreamChunk, TraceCallback,
};
use crate::retry::{RetryConfig, retry_with_backoff};
use crate::schema::fix_tool_schema_for_provider;
use crate::types::{AssistantResponse, Provider, ResponsePart, TokenUsage, ToolCall};
use crate::utils::{
    ConversationMessage, convert_messages_to_responses_input, parse_json_value_strict_str,
    parse_sse_stream, validate_image_input_supported,
};

use super::openai_responses_streaming::{ResponsesToolAccumulator, parse_openai_responses_event};

const DEFAULT_CODEX_BASE_URL: &str = "https://chatgpt.com/backend-api";
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";
const OPENAI_BETA_RESPONSES: &str = "responses=experimental";
const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-06";
const DEFAULT_SSE_HEADER_TIMEOUT: Duration = Duration::from_secs(20);
const DEFAULT_WEBSOCKET_CONNECT_TIMEOUT: Duration = Duration::from_secs(15);

/// ChatGPT Codex Responses client.
pub struct OpenAICodexResponsesClient {
    model: String,
    token: String,
    account_id: String,
    api_url: String,
    websocket_url: String,
    reasoning: Option<String>,
    reasoning_summary: Option<String>,
    text_verbosity: Option<String>,
    service_tier: Option<String>,
    transport: CodexSubscriptionTransport,
    sse_header_timeout: Duration,
    websocket_connect_timeout: Duration,
    trace_callback: Option<TraceCallback>,
}

impl Clone for OpenAICodexResponsesClient {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            token: self.token.clone(),
            account_id: self.account_id.clone(),
            api_url: self.api_url.clone(),
            websocket_url: self.websocket_url.clone(),
            reasoning: self.reasoning.clone(),
            reasoning_summary: self.reasoning_summary.clone(),
            text_verbosity: self.text_verbosity.clone(),
            service_tier: self.service_tier.clone(),
            transport: self.transport,
            sse_header_timeout: self.sse_header_timeout,
            websocket_connect_timeout: self.websocket_connect_timeout,
            trace_callback: self.trace_callback.clone(),
        }
    }
}

impl std::fmt::Debug for OpenAICodexResponsesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICodexResponsesClient")
            .field("model", &self.model)
            .field("api_url", &self.api_url)
            .field("websocket_url", &self.websocket_url)
            .field("reasoning", &self.reasoning)
            .field("reasoning_summary", &self.reasoning_summary)
            .field("text_verbosity", &self.text_verbosity)
            .field("service_tier", &self.service_tier)
            .field("transport", &self.transport)
            .field("sse_header_timeout", &self.sse_header_timeout)
            .field("websocket_connect_timeout", &self.websocket_connect_timeout)
            .finish_non_exhaustive()
    }
}

impl OpenAICodexResponsesClient {
    /// Create a Codex subscription client from typed provider config.
    pub fn new(config: CodexSubscriptionProviderConfig, model: impl Into<String>) -> Result<Self> {
        if config.token.trim().is_empty() {
            return Err(Error::NonRetryable(
                "Codex subscription token is empty".to_string(),
            ));
        }

        let account_id = match config.account_id.filter(|id| !id.trim().is_empty()) {
            Some(id) => id,
            None => extract_account_id(&config.token)?,
        };
        let api_url = resolve_codex_url(config.base_url.as_deref());
        let websocket_url = resolve_codex_websocket_url(&api_url)?;

        Ok(Self {
            model: model.into(),
            token: config.token,
            account_id,
            api_url,
            websocket_url,
            reasoning: config.reasoning_effort,
            reasoning_summary: config.reasoning_summary,
            text_verbosity: config.text_verbosity,
            service_tier: config.service_tier,
            transport: config.transport.unwrap_or(CodexSubscriptionTransport::Auto),
            sse_header_timeout: config
                .sse_header_timeout
                .unwrap_or(DEFAULT_SSE_HEADER_TIMEOUT),
            websocket_connect_timeout: config
                .websocket_connect_timeout
                .unwrap_or(DEFAULT_WEBSOCKET_CONNECT_TIMEOUT),
            trace_callback: None,
        })
    }

    /// Set reasoning effort.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    fn normalized_tools(&self, tools: &[Value]) -> Vec<Value> {
        tools
            .iter()
            .cloned()
            .map(|mut tool| {
                fix_tool_schema_for_provider(&mut tool, "openai-codex-responses");
                tool
            })
            .collect()
    }

    fn build_request_body(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
        stream: bool,
    ) -> Result<Value> {
        let model = config.effective_model(&self.model);
        validate_image_input_supported(messages, Provider::OpenAIResponses, model)?;

        let (instructions, input_items) =
            convert_messages_to_responses_input(messages, Provider::OpenAIResponses)?;

        let mut request = serde_json::json!({
            "model": model,
            "store": false,
            "stream": stream,
            "input": input_items,
            "text": { "verbosity": self.text_verbosity.as_deref().unwrap_or("medium") },
            "include": ["reasoning.encrypted_content"],
            "parallel_tool_calls": true,
        });

        if let Some(instructions) = instructions
            && !instructions.is_empty()
        {
            request["instructions"] = serde_json::json!(instructions);
        }

        if let Some(temperature) = config.temperature {
            request["temperature"] = serde_json::json!(temperature);
        }

        if let Some(ref tools) = config.tools
            && !tools.is_empty()
        {
            request["tools"] = serde_json::json!(self.normalized_tools(tools));
            request["tool_choice"] = serde_json::json!("auto");
        }

        if self.reasoning.is_some() || self.reasoning_summary.is_some() {
            let mut reasoning = serde_json::Map::new();
            if let Some(ref effort) = self.reasoning {
                reasoning.insert("effort".to_string(), serde_json::json!(effort));
            }
            reasoning.insert(
                "summary".to_string(),
                serde_json::json!(self.reasoning_summary.as_deref().unwrap_or("auto")),
            );
            request["reasoning"] = serde_json::Value::Object(reasoning);
        }

        if let Some(ref service_tier) = self.service_tier {
            request["service_tier"] = serde_json::json!(service_tier);
        }

        if let Some(ref schema) = config.output_schema {
            let type_name = config.output_type_name.as_deref().unwrap_or("response");
            request["text"] = serde_json::json!({
                "format": {
                    "type": "json_schema",
                    "name": type_name,
                    "schema": schema,
                    "strict": true
                },
                "verbosity": self.text_verbosity.as_deref().unwrap_or("medium")
            });
        }

        Ok(request)
    }

    fn sse_request_builder(
        &self,
        body: &Value,
        timeout: Option<Duration>,
    ) -> reqwest::RequestBuilder {
        reqwest::Client::new()
            .post(&self.api_url)
            .timeout(timeout.unwrap_or(Duration::from_secs(180)))
            .header(header::AUTHORIZATION, format!("Bearer {}", self.token))
            .header("chatgpt-account-id", &self.account_id)
            .header("originator", "pi")
            .header(header::USER_AGENT, "openbracket-chevalier (rust)")
            .header("OpenAI-Beta", OPENAI_BETA_RESPONSES)
            .header(header::ACCEPT, "text/event-stream")
            .header(header::CONTENT_TYPE, "application/json")
            .json(body)
    }

    async fn make_sse_request(
        &self,
        body: Value,
        timeout: Option<Duration>,
    ) -> Result<reqwest::Response> {
        let response = tokio::time::timeout(
            self.sse_header_timeout,
            self.sse_request_builder(&body, timeout).send(),
        )
        .await
        .map_err(|_| {
            Error::Inference(format!(
                "Codex SSE response headers timed out after {}ms",
                self.sse_header_timeout.as_millis()
            ))
        })??;

        Ok(response)
    }

    fn websocket_builder(&self) -> Result<tokio_websockets::ClientBuilder<'static>> {
        let uri: Uri = self.websocket_url.parse().map_err(|e| {
            Error::NonRetryable(format!(
                "Invalid Codex websocket URL '{}': {}",
                self.websocket_url, e
            ))
        })?;

        let mut builder = tokio_websockets::ClientBuilder::from_uri(uri);
        for (name, value) in [
            ("Authorization", format!("Bearer {}", self.token)),
            ("chatgpt-account-id", self.account_id.clone()),
            ("originator", "pi".to_string()),
            ("User-Agent", "openbracket-chevalier (rust)".to_string()),
            ("OpenAI-Beta", OPENAI_BETA_RESPONSES_WEBSOCKETS.to_string()),
        ] {
            builder = builder.add_header(
                HeaderName::from_bytes(name.as_bytes()).map_err(|e| {
                    Error::NonRetryable(format!("Invalid Codex websocket header '{}': {}", name, e))
                })?,
                HeaderValue::from_str(&value).map_err(|e| {
                    Error::NonRetryable(format!("Invalid Codex websocket header '{}': {}", name, e))
                })?,
            );
        }

        Ok(builder)
    }

    async fn connect_websocket_stream(
        &self,
        body: Value,
        has_tools: bool,
    ) -> Result<Pin<Box<dyn futures::stream::Stream<Item = Result<StreamChunk>> + Send>>> {
        let websocket_request = wrap_websocket_request_body(body)?;
        let request_text = serde_json::to_string(&websocket_request)?;
        let builder = self.websocket_builder()?;

        let connect_result =
            tokio::time::timeout(self.websocket_connect_timeout, builder.connect())
                .await
                .map_err(|_| {
                    Error::Inference(format!(
                        "Codex websocket connect timed out after {}ms",
                        self.websocket_connect_timeout.as_millis()
                    ))
                })?;
        let (mut client, _) = connect_result
            .map_err(|e| Error::Inference(format!("Codex websocket connect failed: {}", e)))?;

        client
            .send(tokio_websockets::Message::text(request_text))
            .await
            .map_err(|e| Error::Inference(format!("Codex websocket send failed: {}", e)))?;

        let debug = debug_codex_stream();
        let (tx, rx) = mpsc::unbounded::<Result<StreamChunk>>();
        tokio::spawn(async move {
            let mut accumulator = ResponsesToolAccumulator::new();

            while let Some(item) = client.next().await {
                let message = match item {
                    Ok(message) => message,
                    Err(error) => {
                        let _ = tx.unbounded_send(Err(Error::Inference(format!(
                            "Codex websocket read failed: {}",
                            error
                        ))));
                        return;
                    }
                };

                if message.is_close() {
                    return;
                }

                let Some(text) = message.as_text() else {
                    continue;
                };

                let event_json = match parse_json_value_strict_str(text) {
                    Ok(value) => value,
                    Err(error) => {
                        let _ = tx.unbounded_send(Err(Error::Inference(format!(
                            "Invalid Codex websocket JSON: {}",
                            error
                        ))));
                        return;
                    }
                };

                if debug {
                    eprintln!("codex websocket event: {}", event_json);
                }

                if let Err(error) = reject_codex_error_event(&event_json) {
                    let _ = tx.unbounded_send(Err(error));
                    return;
                }

                let done = is_completion_event(&event_json);
                let event_json = normalize_completion_event(event_json);
                for chunk in parse_openai_responses_event(&event_json, &mut accumulator, has_tools)
                {
                    if tx.unbounded_send(Ok(chunk)).is_err() {
                        return;
                    }
                }

                if done {
                    return;
                }
            }
        });

        Ok(Box::pin(rx))
    }

    fn handle_error_response(&self, status: StatusCode, body: String) -> Error {
        if let Some(friendly) = codex_friendly_error(status, &body) {
            return friendly;
        }

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

    async fn connect_sse_stream(
        &self,
        request_body: Value,
        timeout: Option<Duration>,
        retry_config: Option<RetryConfig>,
        has_tools: bool,
    ) -> Result<Pin<Box<dyn futures::stream::Stream<Item = Result<StreamChunk>> + Send>>> {
        let config = retry_config.unwrap_or_default();
        let response = retry_with_backoff(config, || async {
            let resp = self.make_sse_request(request_body.clone(), timeout).await?;
            let status = resp.status();
            if !status.is_success() {
                let error_body = resp.text().await.unwrap_or_default();
                return Err(self.handle_error_response(status, error_body));
            }
            Ok(resp)
        })
        .await?;

        let debug = debug_codex_stream();
        let sse_stream = parse_sse_stream(response);
        let chunk_stream = sse_stream.scan(
            ResponsesToolAccumulator::new(),
            move |accumulator, sse_result| {
                let sse_json = match sse_result {
                    Ok(json) => json,
                    Err(e) => return futures::future::ready(Some(vec![Err(e)])),
                };

                if debug {
                    eprintln!("codex sse event: {}", sse_json);
                }

                if let Err(error) = reject_codex_error_event(&sse_json) {
                    return futures::future::ready(Some(vec![Err(error)]));
                }

                let event_json = normalize_completion_event(sse_json);
                let chunks = parse_openai_responses_event(&event_json, accumulator, has_tools);
                futures::future::ready(Some(chunks.into_iter().map(Ok).collect()))
            },
        );

        Ok(Box::pin(chunk_stream.flat_map(futures::stream::iter)))
    }

    async fn connect_auto_stream(
        &self,
        request_body: Value,
        timeout: Option<Duration>,
        retry_config: Option<RetryConfig>,
        has_tools: bool,
    ) -> Result<Pin<Box<dyn futures::stream::Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut websocket_stream = match self
            .connect_websocket_stream(request_body.clone(), has_tools)
            .await
        {
            Ok(stream) => stream,
            Err(websocket_error) => {
                if debug_codex_stream() {
                    eprintln!(
                        "codex websocket failed before stream start; falling back to SSE: {}",
                        websocket_error
                    );
                }
                return self
                    .connect_sse_stream(request_body, timeout, retry_config, has_tools)
                    .await;
            }
        };

        match websocket_stream.next().await {
            Some(Ok(first_chunk)) => Ok(Box::pin(
                futures::stream::once(async move { Ok(first_chunk) }).chain(websocket_stream),
            )),
            Some(Err(websocket_error)) => {
                if debug_codex_stream() {
                    eprintln!(
                        "codex websocket failed before first chunk; falling back to SSE: {}",
                        websocket_error
                    );
                }
                self.connect_sse_stream(request_body, timeout, retry_config, has_tools)
                    .await
            }
            None => {
                if debug_codex_stream() {
                    eprintln!("codex websocket closed before first chunk; falling back to SSE");
                }
                self.connect_sse_stream(request_body, timeout, retry_config, has_tools)
                    .await
            }
        }
    }
}

#[async_trait]
impl InferenceClient for OpenAICodexResponsesClient {
    async fn get_generation(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<GenerationResponse> {
        let mut stream = self.connect_and_listen(messages, config).await?;
        let mut response = AssistantResponse::default();
        let mut usage = TokenUsage::default();

        while let Some(chunk) = stream.next().await {
            match chunk? {
                StreamChunk::Content(text) => {
                    response.push_output(ResponsePart::Text { text });
                }
                StreamChunk::Reasoning(text) => {
                    response.push_output(ResponsePart::Reasoning { text });
                }
                StreamChunk::Signature(value) => {
                    response.push_output(ResponsePart::Signature { value });
                }
                StreamChunk::ToolCallComplete(tool) => {
                    response.push_output(ResponsePart::Tool {
                        call: ToolCall::from_provider_format(tool, Provider::OpenAIResponses)?,
                    });
                }
                StreamChunk::ToolCallPartial(_) => {}
                StreamChunk::Usage {
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                    cache_write_input_tokens,
                } => {
                    usage = TokenUsage {
                        input_tokens,
                        output_tokens,
                        cached_tokens,
                        cache_write_input_tokens,
                    };
                }
            }
        }

        Ok(GenerationResponse::from_assistant_response(
            response, usage, None, None,
        ))
    }

    async fn connect_and_listen(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn futures::stream::Stream<Item = Result<StreamChunk>> + Send>>> {
        let request_body = self.build_request_body(messages, config, true)?;
        let has_tools = config.tools.as_ref().is_some_and(|tools| !tools.is_empty());

        match self.transport {
            CodexSubscriptionTransport::WebSocket => {
                self.connect_websocket_stream(request_body, has_tools).await
            }
            CodexSubscriptionTransport::Sse => {
                self.connect_sse_stream(
                    request_body,
                    config.timeout,
                    config.retry_config.clone(),
                    has_tools,
                )
                .await
            }
            CodexSubscriptionTransport::Auto => {
                self.connect_auto_stream(
                    request_body,
                    config.timeout,
                    config.retry_config.clone(),
                    has_tools,
                )
                .await
            }
        }
    }

    fn provider(&self) -> Provider {
        Provider::OpenAIResponses
    }

    fn set_trace_callback(&mut self, callback: TraceCallback) {
        self.trace_callback = Some(callback);
    }
}

fn resolve_codex_url(base_url: Option<&str>) -> String {
    let raw = base_url
        .filter(|url| !url.trim().is_empty())
        .unwrap_or(DEFAULT_CODEX_BASE_URL)
        .trim()
        .trim_end_matches('/')
        .to_string();

    if raw.ends_with("/codex/responses") {
        raw
    } else if raw.ends_with("/codex") {
        format!("{raw}/responses")
    } else {
        format!("{raw}/codex/responses")
    }
}

fn resolve_codex_websocket_url(api_url: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(api_url)
        .map_err(|e| Error::NonRetryable(format!("Invalid Codex URL '{}': {}", api_url, e)))?;
    match url.scheme() {
        "https" => {
            url.set_scheme("wss")
                .map_err(|_| Error::NonRetryable("Invalid Codex websocket scheme".to_string()))?;
        }
        "http" => {
            url.set_scheme("ws")
                .map_err(|_| Error::NonRetryable("Invalid Codex websocket scheme".to_string()))?;
        }
        "wss" | "ws" => {}
        scheme => {
            return Err(Error::NonRetryable(format!(
                "Unsupported Codex URL scheme '{}'",
                scheme
            )));
        }
    }
    Ok(url.to_string())
}

fn extract_account_id(token: &str) -> Result<String> {
    let payload = token.split('.').nth(1).ok_or_else(|| {
        Error::NonRetryable("Failed to extract accountId from Codex token".to_string())
    })?;
    let decoded = URL_SAFE_NO_PAD.decode(payload).map_err(|_| {
        Error::NonRetryable("Failed to decode accountId from Codex token".to_string())
    })?;
    let body: Value = serde_json::from_slice(&decoded).map_err(|_| {
        Error::NonRetryable("Failed to parse accountId from Codex token".to_string())
    })?;
    body.get(JWT_CLAIM_PATH)
        .and_then(|claim| claim.get("chatgpt_account_id"))
        .and_then(|value| value.as_str())
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .ok_or_else(|| Error::NonRetryable("No chatgpt accountId in Codex token".to_string()))
}

fn wrap_websocket_request_body(body: Value) -> Result<Value> {
    let Value::Object(mut object) = body else {
        return Err(Error::NonRetryable(
            "Codex websocket request body must be an object".to_string(),
        ));
    };
    object.insert(
        "type".to_string(),
        Value::String("response.create".to_string()),
    );
    Ok(Value::Object(object))
}

fn is_completion_event(event_json: &Value) -> bool {
    matches!(
        event_json.get("type").and_then(|value| value.as_str()),
        Some("response.done" | "response.completed" | "response.incomplete")
    )
}

fn normalize_completion_event(mut event_json: Value) -> Value {
    if matches!(
        event_json.get("type").and_then(|value| value.as_str()),
        Some("response.completed" | "response.incomplete")
    ) && let Value::Object(ref mut object) = event_json
    {
        object.insert(
            "type".to_string(),
            Value::String("response.done".to_string()),
        );
    }
    event_json
}

fn reject_codex_error_event(event_json: &Value) -> Result<()> {
    match event_json.get("type").and_then(|value| value.as_str()) {
        Some("error") => {
            let detail = codex_error_detail(event_json);
            Err(Error::Inference(format!("Codex error: {detail}")))
        }
        Some("response.failed") => {
            let error = event_json
                .get("response")
                .and_then(|response| response.get("error"));
            let message = error
                .and_then(|value| value.get("message"))
                .and_then(|value| value.as_str())
                .unwrap_or("Codex response failed");
            Err(Error::Inference(message.to_string()))
        }
        _ => Ok(()),
    }
}

fn codex_error_detail(event_json: &Value) -> String {
    let nested = event_json.get("error");
    let message = event_json
        .get("message")
        .or_else(|| nested.and_then(|error| error.get("message")))
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty());
    if let Some(message) = message {
        return message.to_string();
    }
    let code = event_json
        .get("code")
        .or_else(|| nested.and_then(|error| error.get("code")))
        .or_else(|| nested.and_then(|error| error.get("type")))
        .and_then(|value| value.as_str())
        .filter(|value| !value.trim().is_empty());
    if let Some(code) = code {
        return code.to_string();
    }
    event_json.to_string()
}

fn codex_friendly_error(status: StatusCode, body: &str) -> Option<Error> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    let err = parsed.get("error")?;
    let code = err
        .get("code")
        .or_else(|| err.get("type"))
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if status != StatusCode::TOO_MANY_REQUESTS
        && !code.contains("usage_limit")
        && !code.contains("rate_limit")
    {
        return None;
    }

    let plan = err
        .get("plan_type")
        .and_then(|value| value.as_str())
        .map(|plan| format!(" ({})", plan.to_ascii_lowercase()))
        .unwrap_or_default();
    Some(Error::Inference(format!(
        "You have hit your ChatGPT usage limit{}.",
        plan
    )))
}

fn debug_codex_stream() -> bool {
    std::env::var("CHEVALIER_DEBUG_CODEX_STREAM")
        .map(|v| matches!(v.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_codex_url() {
        assert_eq!(
            resolve_codex_url(None),
            "https://chatgpt.com/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://example.test/backend-api/codex")),
            "https://example.test/backend-api/codex/responses"
        );
        assert_eq!(
            resolve_codex_url(Some("https://example.test/backend-api/codex/responses")),
            "https://example.test/backend-api/codex/responses"
        );
    }

    #[test]
    fn test_wrap_websocket_request_body() {
        let body = serde_json::json!({"model": "gpt-5.1-codex", "stream": true});
        let wrapped = wrap_websocket_request_body(body).unwrap();
        assert_eq!(wrapped["type"], "response.create");
        assert_eq!(wrapped["model"], "gpt-5.1-codex");
    }

    #[test]
    fn test_extract_account_id_from_jwt() {
        let payload = URL_SAFE_NO_PAD.encode(
            serde_json::json!({
                JWT_CLAIM_PATH: {
                    "chatgpt_account_id": "acct_123"
                }
            })
            .to_string(),
        );
        let token = format!("header.{payload}.signature");
        assert_eq!(extract_account_id(&token).unwrap(), "acct_123");
    }

    #[test]
    fn test_build_codex_request_body() {
        let client = OpenAICodexResponsesClient::new(
            CodexSubscriptionProviderConfig {
                token: "header.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF8xMjMifX0.signature".to_string(),
                account_id: Some("acct_123".to_string()),
                base_url: None,
                transport: Some(CodexSubscriptionTransport::Sse),
                sse_header_timeout: None,
                websocket_connect_timeout: None,
                reasoning_effort: Some("high".to_string()),
                reasoning_summary: Some("concise".to_string()),
                text_verbosity: Some("medium".to_string()),
                service_tier: Some("priority".to_string()),
            },
            "gpt-5.1-codex",
        )
        .unwrap();
        let config = GenerationConfig::new("gpt-5.1-codex");
        let messages = vec![ConversationMessage::Chat(crate::types::ChatMessage::user(
            "Hello",
        ))];
        let body = client.build_request_body(&messages, &config, true).unwrap();

        assert_eq!(body["model"], "gpt-5.1-codex");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert!(body["input"].is_array());
        assert_eq!(body["text"]["verbosity"], "medium");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["summary"], "concise");
        assert_eq!(body["service_tier"], "priority");
    }

    #[test]
    fn test_build_codex_request_body_with_summary_only_reasoning() {
        let client = OpenAICodexResponsesClient::new(
            CodexSubscriptionProviderConfig {
                token: "header.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF8xMjMifX0.signature".to_string(),
                account_id: Some("acct_123".to_string()),
                base_url: None,
                transport: Some(CodexSubscriptionTransport::Sse),
                sse_header_timeout: None,
                websocket_connect_timeout: None,
                reasoning_effort: None,
                reasoning_summary: Some("detailed".to_string()),
                text_verbosity: None,
                service_tier: None,
            },
            "gpt-5.1-codex",
        )
        .unwrap();
        let config = GenerationConfig::new("gpt-5.1-codex");
        let messages = vec![ConversationMessage::Chat(crate::types::ChatMessage::user(
            "Hello",
        ))];
        let body = client.build_request_body(&messages, &config, true).unwrap();

        assert!(body["reasoning"].get("effort").is_none());
        assert_eq!(body["reasoning"]["summary"], "detailed");
        assert_eq!(body["text"]["verbosity"], "medium");
    }
}
