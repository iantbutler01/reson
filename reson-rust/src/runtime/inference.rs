//! Inference orchestration utilities
//!
//! Handles LLM API calls with native tools or parser-based approaches.

use futures::stream::{Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Error, Result};
use crate::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient, OAIClient,
    OpenAIResponsesClient, OpenRouterClient, OpenRouterResponsesClient, StreamChunk,
};
use crate::schema::fix_output_schema_for_provider;
use crate::types::{
    AssistantResponse, ChatMessage, CreateResult, ReasoningSegment, ResponsePart,
    ResponseStreamEvent, TokenUsage, ToolCall,
};
use crate::utils::ConversationMessage;

use super::{Accumulators, ToolFunction, ToolSchemaInfo};

/// Result from non-streaming LLM call
pub struct CallResult {
    pub response: AssistantResponse,
}

async fn stream_chunk_to_runtime_events(
    chunk: StreamChunk,
    accumulators: Arc<RwLock<Accumulators>>,
    response: Arc<RwLock<AssistantResponse>>,
) -> Result<Vec<ResponseStreamEvent>> {
    match chunk {
        StreamChunk::Content(text) => {
            {
                let mut acc = accumulators.write().await;
                acc.raw_response.push(text.clone());
            }
            {
                let mut response = response.write().await;
                response.push_output(ResponsePart::Text { text: text.clone() });
            }
            Ok(vec![ResponseStreamEvent::Output(ResponsePart::Text { text })])
        }
        StreamChunk::Reasoning(text) => {
            let mut acc = accumulators.write().await;
            acc.reasoning.push(text.clone());

            if let Some(current) = acc.current_reasoning_segment.as_mut() {
                current.content.push_str(&text);
                let updated_content = current.content.clone();
                if let Some(last) = acc.reasoning_segments.last_mut() {
                    last.content = updated_content;
                }
            } else {
                let segment = ReasoningSegment::with_index(text.clone(), acc.reasoning_segments.len());
                acc.reasoning_segments.push(segment.clone());
                acc.current_reasoning_segment = Some(segment);
            }
            drop(acc);

            {
                let mut response = response.write().await;
                response.push_output(ResponsePart::Reasoning { text: text.clone() });
            }

            Ok(vec![ResponseStreamEvent::Output(ResponsePart::Reasoning { text })])
        }
        StreamChunk::Signature(sig) => {
            let mut acc = accumulators.write().await;
            if let Some(current) = acc.current_reasoning_segment.as_mut() {
                current.signature = Some(sig.clone());
                if let Some(last) = acc.reasoning_segments.last_mut() {
                    last.signature = Some(sig.clone());
                }
            }
            drop(acc);

            {
                let mut response = response.write().await;
                response.push_output(ResponsePart::Signature { value: sig.clone() });
            }

            Ok(vec![ResponseStreamEvent::Output(ResponsePart::Signature {
                value: sig,
            })])
        }
        StreamChunk::ToolCallComplete(tool) => {
            let tool_calls = match ToolCall::create(tool)? {
                CreateResult::Single(call) => vec![call],
                CreateResult::Multiple(calls) => calls,
                CreateResult::Empty => Vec::new(),
            };

            let mut events = Vec::new();
            let mut response_guard = response.write().await;
            for call in tool_calls {
                response_guard.push_output(ResponsePart::Tool { call: call.clone() });
                events.push(ResponseStreamEvent::Output(ResponsePart::Tool { call }));
            }
            Ok(events)
        }
        StreamChunk::ToolCallPartial(tool) => Ok(vec![ResponseStreamEvent::ToolPartial(tool)]),
        StreamChunk::Usage {
            input_tokens,
            output_tokens,
            cached_tokens,
        } => Ok(vec![ResponseStreamEvent::Usage(TokenUsage {
            input_tokens,
            output_tokens,
            cached_tokens,
        })]),
    }
}

/// Convert a field type to JSON schema type
fn field_type_to_json_type(field_type: &str) -> serde_json::Value {
    serde_json::json!(infer_json_type(field_type))
}

/// Infer JSON schema type name from a Rust type string
fn infer_json_type(field_type: &str) -> &'static str {
    let ty = strip_type_modifiers(field_type);
    if ty.is_empty() {
        return "string";
    }

    // Handle slices/arrays like &[T] or [T; N]
    if ty.starts_with('[') {
        return "array";
    }

    // Handle manual descriptors (string, integer, etc)
    let ty_lower = ty.to_ascii_lowercase();
    match ty_lower.as_str() {
        "string" | "&str" | "str" => return "string",
        "number" | "f32" | "f64" | "float" => return "number",
        "integer" | "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "isize" => return "integer",
        "boolean" | "bool" => return "boolean",
        "array" => return "array",
        "object" => return "object",
        _ => {}
    }

    let (base, generics) = split_type_and_generics(ty);
    let ident = base.rsplit("::").next().unwrap_or(base);

    if ident == "Option" {
        if let Some(inner) = generics {
            return infer_json_type(inner);
        }
        return "string";
    }

    if ident == "Vec" || ident == "VecDeque" || ident == "LinkedList" {
        return "array";
    }

    if ident == "HashMap" || ident == "BTreeMap" || ident == "IndexMap" {
        return "object";
    }

    if ident == "Value" && base.contains("serde_json") {
        return "object";
    }

    match ident {
        "String" => "string",
        "str" => "string",
        "f32" | "f64" => "number",
        "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "usize" | "isize" => {
            "integer"
        }
        "bool" => "boolean",
        _ => "string",
    }
}

/// Determine the `items` schema for array-like fields
fn field_type_array_items(field_type: &str) -> Option<serde_json::Value> {
    let ty = strip_type_modifiers(field_type);
    if ty.is_empty() {
        return None;
    }

    // Handle manual "array" descriptor with unknown item type
    if ty.eq_ignore_ascii_case("array") {
        return Some(serde_json::json!({ "type": "string" }));
    }

    // Handle slices like &[T] or [T; N]
    if ty.starts_with('[') {
        if let Some(end) = ty.find(';') {
            let inner = ty[1..end].trim();
            if !inner.is_empty() {
                return Some(serde_json::json!({ "type": infer_json_type(inner) }));
            }
        } else if let Some(end) = ty.rfind(']') {
            let inner = ty[1..end].trim();
            if !inner.is_empty() {
                return Some(serde_json::json!({ "type": infer_json_type(inner) }));
            }
        }
        return Some(serde_json::json!({ "type": "string" }));
    }

    let (base, generics) = split_type_and_generics(ty);
    let ident = base.rsplit("::").next().unwrap_or(base);

    if ident == "Option" {
        if let Some(inner) = generics {
            return field_type_array_items(inner);
        }
        return None;
    }

    if ident == "Vec" || ident == "VecDeque" || ident == "LinkedList" {
        if let Some(inner) = generics {
            return Some(serde_json::json!({ "type": infer_json_type(inner) }));
        }
        return Some(serde_json::json!({ "type": "string" }));
    }

    None
}

/// Remove reference/mut modifiers from type strings
fn strip_type_modifiers(field_type: &str) -> &str {
    let mut ty = field_type.trim();
    loop {
        if let Some(stripped) = ty.strip_prefix('&') {
            ty = stripped.trim_start();
            if let Some(stripped_mut) = ty.strip_prefix("mut") {
                ty = stripped_mut.trim_start();
            }
            continue;
        }
        break;
    }
    ty
}

/// Split a type string into the base identifier and its first generic argument
fn split_type_and_generics(ty: &str) -> (&str, Option<&str>) {
    if let Some(start) = ty.find('<') {
        let base = &ty[..start];
        let mut depth = 0usize;
        for (offset, ch) in ty[start..].char_indices() {
            match ch {
                '<' => depth += 1,
                '>' => {
                    if depth == 0 {
                        break;
                    }
                    depth -= 1;
                    if depth == 0 {
                        let inner = &ty[start + 1..start + offset];
                        return (base, Some(inner));
                    }
                }
                _ => {}
            }
        }
        (base, None)
    } else {
        (ty, None)
    }
}

/// Generate native tool schemas for the given model/provider
///
/// Creates provider-specific tool schemas from registered tools.
/// Uses the tool_schemas mapping for tools with explicit schema info.
fn generate_tool_schemas(
    tools: &HashMap<String, ToolFunction>,
    tool_schemas: &HashMap<String, ToolSchemaInfo>,
    model: &str,
) -> Result<Vec<serde_json::Value>> {
    if tools.is_empty() {
        return Ok(Vec::new());
    }

    // Parse provider from model string
    let provider = resolve_provider_key(model);

    // Get appropriate schema generator
    let generator = crate::schema::get_schema_generator(&provider)?;

    // For each tool, generate schema
    let mut schemas = Vec::new();

    for (tool_name, _tool_fn) in tools.iter() {
        // Check if we have schema info for this tool
        let tool_schema = if let Some(schema_info) = tool_schemas.get(tool_name) {
            // Build properties and required from field descriptions
            let mut properties = serde_json::Map::new();
            let mut required = Vec::new();

            for field in &schema_info.fields {
                let mut field_schema = serde_json::Map::new();
                let field_type_json = field_type_to_json_type(&field.field_type);
                let is_array = field_type_json == serde_json::json!("array");
                field_schema.insert("type".to_string(), field_type_json);
                if is_array {
                    let items_schema = field_type_array_items(&field.field_type)
                        .unwrap_or_else(|| serde_json::json!({ "type": "string" }));
                    field_schema.insert("items".to_string(), items_schema);
                }
                if !field.description.is_empty() {
                    field_schema.insert(
                        "description".to_string(),
                        serde_json::json!(field.description),
                    );
                }
                properties.insert(field.name.clone(), serde_json::Value::Object(field_schema));

                if field.required {
                    required.push(serde_json::json!(field.name));
                }
            }

            generator.generate_schema(
                tool_name,
                &schema_info.description,
                serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required
                }),
            )
        } else {
            // No schema info - generate minimal schema
            generator.generate_schema(
                tool_name,
                &format!("Tool: {}", tool_name),
                serde_json::json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }),
            )
        };

        schemas.push(tool_schema);
    }

    Ok(schemas)
}

/// Resolve provider key for model strings, including responses modifiers.
fn resolve_provider_key(model: &str) -> String {
    let parts: Vec<&str> = model.split(':').collect();
    if parts.len() >= 3 && parts[1] == "resp" {
        match parts[0] {
            "openai" => "openai-responses".to_string(),
            "openrouter" => "openrouter-responses".to_string(),
            other => other.to_string(),
        }
    } else {
        parts.first().unwrap_or(&model).to_string()
    }
}

/// Create an inference client from model string
///
/// Format: `provider:model@param=value` or `provider:resp:model@param=value`
///
/// Parameters:
/// - `@reasoning=<value>` - reasoning effort (numeric budget or level like `high`)
/// - `@server_url=<url>` - custom API endpoint (required for `custom-openai`, optional for `openai`)
/// - `@api_key=<key>` - inline API key (overrides env var and `api_key` parameter)
///
/// Examples:
/// - `anthropic:claude-3-5-sonnet-20241022`
/// - `openrouter:openai/gpt-4o@reasoning=high`
/// - `openai:gpt-4o`
/// - `openai:resp:gpt-4o`
/// - `openrouter:resp:openai/o4-mini`
/// - `custom-openai:llama-3@server_url=http://localhost:8000/v1`
/// - `custom-openai:my-model@server_url=http://localhost:11434/v1@api_key=ollama`
pub fn create_inference_client(
    model_str: &str,
    api_key: Option<&str>,
) -> Result<Box<dyn InferenceClient>> {
    // Parse model string
    let parts: Vec<&str> = model_str.split(':').collect();
    if parts.len() < 2 {
        return Err(Error::NonRetryable(format!(
            "Invalid model string format: {}. Expected 'provider:model'",
            model_str
        )));
    }
    let (provider, model_part) = if parts.len() >= 3 && parts[1] == "resp" {
        (format!("{}-responses", parts[0]), parts[2..].join(":"))
    } else {
        (parts[0].to_string(), parts[1..].join(":"))
    };

    // Parse parameters (e.g., @reasoning=1024, @server_url=http://localhost:8000/v1, @api_key=sk-...)
    let mut reasoning = None;
    let mut server_url = None;
    let mut inline_api_key = None;
    let model_name = if model_part.contains('@') {
        let model_parts: Vec<&str> = model_part.split('@').collect();

        for param in &model_parts[1..] {
            if param.starts_with("reasoning=") {
                reasoning = Some(param.strip_prefix("reasoning=").unwrap().to_string());
            } else if param.starts_with("server_url=") {
                server_url = Some(param.strip_prefix("server_url=").unwrap().to_string());
            } else if param.starts_with("api_key=") {
                inline_api_key = Some(param.strip_prefix("api_key=").unwrap().to_string());
            }
        }

        model_parts[0].to_string()
    } else {
        model_part.to_string()
    };

    // Resolve API key: @api_key= > api_key parameter > env var
    let key = if let Some(k) = inline_api_key {
        k
    } else if let Some(k) = api_key {
        k.to_string()
    } else {
        match provider.as_str() {
            "anthropic" => std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| Error::NonRetryable("ANTHROPIC_API_KEY not set".to_string()))?,
            "openai" => std::env::var("OPENAI_API_KEY")
                .map_err(|_| Error::NonRetryable("OPENAI_API_KEY not set".to_string()))?,
            "openai-responses" => std::env::var("OPENAI_API_KEY")
                .map_err(|_| Error::NonRetryable("OPENAI_API_KEY not set".to_string()))?,
            "openrouter" => std::env::var("OPENROUTER_API_KEY")
                .or_else(|_| std::env::var("OPENROUTER_KEY"))
                .map_err(|_| Error::NonRetryable("OPENROUTER_API_KEY not set".to_string()))?,
            "openrouter-responses" => std::env::var("OPENROUTER_API_KEY")
                .or_else(|_| std::env::var("OPENROUTER_KEY"))
                .map_err(|_| Error::NonRetryable("OPENROUTER_API_KEY not set".to_string()))?,
            "google-gemini" | "google-genai" | "gemini" => {
                std::env::var("GOOGLE_GEMINI_API_KEY")
                    .map_err(|_| Error::NonRetryable("GOOGLE_GEMINI_API_KEY not set".to_string()))?
            }
            "custom-openai" => {
                std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "not-needed".to_string())
            }
            _ => {
                return Err(Error::NonRetryable(format!(
                    "Unknown provider: {}",
                    provider
                )))
            }
        }
    };

    // Create client
    let client: Box<dyn InferenceClient> = match provider.as_str() {
        "anthropic" => {
            let mut client = AnthropicClient::new(key, model_name);
            if let Some(r) = reasoning {
                if let Ok(budget) = r.parse::<u32>() {
                    client = client.with_thinking_budget(budget);
                }
            }
            Box::new(client)
        }
        "openai" => {
            let mut client = OAIClient::new(key, model_name);
            if let Some(url) = server_url {
                client = client.with_api_url(url);
            } else if let Ok(base_url) = std::env::var("OPENAI_BASE_URL") {
                client = client.with_api_url(base_url);
            }
            if let Some(r) = reasoning {
                client = client.with_reasoning(r);
            }
            Box::new(client)
        }
        "openai-responses" => {
            let mut client = OpenAIResponsesClient::new(key, model_name);
            if let Some(r) = reasoning {
                client = client.with_reasoning(r);
            }
            Box::new(client)
        }
        "openrouter" => {
            let mut client = OpenRouterClient::new(key, model_name, None, None);
            if let Some(r) = reasoning {
                client = client.with_reasoning(r);
            }
            Box::new(client)
        }
        "openrouter-responses" => {
            let mut client = OpenRouterResponsesClient::new(key, model_name, None, None);
            if let Some(r) = reasoning {
                client = client.with_reasoning(r);
            }
            Box::new(client)
        }
        "google-gemini" | "google-genai" | "gemini" => {
            let mut client = GoogleGenAIClient::new(key, model_name);
            if let Some(r) = reasoning {
                if let Ok(budget) = r.parse::<u32>() {
                    client = client.with_thinking_budget(budget);
                }
            }
            Box::new(client)
        }
        "custom-openai" => {
            let url = server_url.ok_or_else(|| {
                Error::NonRetryable(
                    "custom-openai requires @server_url= parameter (e.g., custom-openai:model@server_url=http://localhost:8000/v1)".to_string(),
                )
            })?;
            let mut client = OAIClient::new(key, model_name).with_api_url(url);
            if let Some(r) = reasoning {
                client = client.with_reasoning(r);
            }
            Box::new(client)
        }
        _ => {
            return Err(Error::NonRetryable(format!(
                "Unsupported provider: {}",
                provider
            )))
        }
    };

    Ok(client)
}

/// Execute a non-streaming LLM call
#[allow(clippy::too_many_arguments)]
pub async fn call_llm(
    prompt: Option<&str>,
    model: &str,
    tools: Arc<RwLock<HashMap<String, ToolFunction>>>,
    tool_schema_info: Arc<RwLock<HashMap<String, ToolSchemaInfo>>>,
    output_type_name: Option<String>,
    output_schema: Option<serde_json::Value>,
    api_key: Option<&str>,
    system: Option<&str>,
    history: Option<Vec<ConversationMessage>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    timeout: Option<std::time::Duration>,
    _call_context: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>>,
) -> Result<CallResult> {
    // Create client
    let client = create_inference_client(model, api_key)?;

    // Build messages
    let mut messages = Vec::new();

    if let Some(sys) = system {
        messages.push(ConversationMessage::Chat(ChatMessage::system(sys)));
    }

    if let Some(hist) = history {
        messages.extend(hist);
    }

    if let Some(p) = prompt {
        messages.push(ConversationMessage::Chat(ChatMessage::user(p)));
    }

    if messages.is_empty() {
        return Err(Error::NonRetryable(
            "No messages to send to LLM".to_string(),
        ));
    }

    // Generate tool schemas (always enabled - native tools only)
    let tool_schemas = {
        let tools_guard = tools.read().await;
        let schemas_guard = tool_schema_info.read().await;
        if !tools_guard.is_empty() {
            Some(generate_tool_schemas(&tools_guard, &schemas_guard, model)?)
        } else {
            None
        }
    };

    // Fix output schema for provider-specific requirements
    let provider = resolve_provider_key(model);
    let fixed_output_schema = output_schema.map(|mut schema| {
        fix_output_schema_for_provider(&mut schema, &provider);
        schema
    });

    // Build config
    let config = GenerationConfig {
        model: String::new(), // Use client's default
        max_tokens,
        temperature,
        top_p,
        tools: tool_schemas,
        native_tools: true, // Always true - we only support native tools
        reasoning_effort: None,
        thinking_budget: None,
        output_schema: fixed_output_schema,
        output_type_name,
        timeout,
    };

    // Make API call
    let response = client.get_generation(&messages, &config).await?;

    if response.response.output.is_empty() {
        return Err(Error::Inference(
            "Provider returned an empty assistant response".to_string(),
        ));
    }

    Ok(CallResult {
        response: response.response,
    })
}

/// Execute a streaming LLM call
#[allow(clippy::too_many_arguments)]
pub async fn call_llm_stream(
    prompt: Option<&str>,
    model: &str,
    tools: Arc<RwLock<HashMap<String, ToolFunction>>>,
    tool_schema_info: Arc<RwLock<HashMap<String, ToolSchemaInfo>>>,
    output_type_name: Option<String>,
    output_schema: Option<serde_json::Value>,
    api_key: Option<&str>,
    system: Option<&str>,
    history: Option<Vec<ConversationMessage>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    timeout: Option<std::time::Duration>,
    _call_context: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>>,
    accumulators: Arc<RwLock<Accumulators>>,
) -> Result<Pin<Box<dyn Stream<Item = Result<ResponseStreamEvent>> + Send>>> {
    // Create client
    let client = create_inference_client(model, api_key)?;

    // Build messages
    let mut messages = Vec::new();

    if let Some(sys) = system {
        messages.push(ConversationMessage::Chat(ChatMessage::system(sys)));
    }

    if let Some(hist) = history {
        messages.extend(hist);
    }

    if let Some(p) = prompt {
        messages.push(ConversationMessage::Chat(ChatMessage::user(p)));
    }

    if messages.is_empty() {
        return Err(Error::NonRetryable(
            "No messages to send to LLM".to_string(),
        ));
    }

    // Generate tool schemas (always enabled - native tools only)
    let tool_schemas = {
        let tools_guard = tools.read().await;
        let schemas_guard = tool_schema_info.read().await;
        if !tools_guard.is_empty() {
            Some(generate_tool_schemas(&tools_guard, &schemas_guard, model)?)
        } else {
            None
        }
    };

    // Fix output schema for provider-specific requirements
    let provider = resolve_provider_key(model);
    let fixed_output_schema = output_schema.map(|mut schema| {
        fix_output_schema_for_provider(&mut schema, &provider);
        schema
    });

    // Build config
    let config = GenerationConfig {
        model: String::new(),
        max_tokens,
        temperature,
        top_p,
        tools: tool_schemas,
        native_tools: true, // Always true - we only support native tools
        reasoning_effort: None,
        thinking_budget: None,
        output_schema: fixed_output_schema,
        output_type_name,
        timeout,
    };

    // Get streaming response
    let stream = client.connect_and_listen(&messages, &config).await?;

    let response = Arc::new(RwLock::new(AssistantResponse::default()));
    let stream_response = response.clone();

    let transformed = stream.then(move |chunk_result| {
        let accumulators = accumulators.clone();
        let response = stream_response.clone();
        async move {
            match chunk_result {
                Ok(chunk) => match stream_chunk_to_runtime_events(chunk, accumulators, response).await
                {
                    Ok(events) => events.into_iter().map(Ok).collect(),
                    Err(e) => vec![Err(e)],
                },
                Err(e) => vec![Err(e)],
            }
        }
    })
    .flat_map(futures::stream::iter);

    let final_response = response.clone();
    let completed = futures::stream::once(async move {
        let response = final_response.read().await.clone();
        if response.output.is_empty() {
            Err(Error::Inference(
                "Provider returned an empty assistant response".to_string(),
            ))
        } else {
            Ok(ResponseStreamEvent::Complete(response))
        }
    });

    Ok(Box::pin(transformed.chain(completed)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_inference_client_anthropic() {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        let result = create_inference_client("anthropic:claude-3-5-sonnet-20241022", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_inference_client_with_reasoning() {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key");
        let result =
            create_inference_client("anthropic:claude-3-5-sonnet-20241022@reasoning=1024", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_inference_client_invalid_format() {
        let result = create_inference_client("invalid-format", None);
        assert!(result.is_err());
    }

    #[test]
    fn test_create_inference_client_openai() {
        std::env::set_var("OPENAI_API_KEY", "test-key");
        let result = create_inference_client("openai:gpt-4o", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_inference_client_openai_responses() {
        std::env::set_var("OPENAI_API_KEY", "test-key");
        let result = create_inference_client("openai:resp:gpt-4o", None);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_stream_content_updates_raw_response_accumulator() {
        let accumulators = Arc::new(RwLock::new(Accumulators::default()));
        let response = Arc::new(RwLock::new(AssistantResponse::default()));
        let events = stream_chunk_to_runtime_events(
            StreamChunk::Content("hello".to_string()),
            accumulators.clone(),
            response.clone(),
        )
        .await
        .unwrap();

        assert_eq!(events.len(), 1);
        assert!(matches!(
            &events[0],
            ResponseStreamEvent::Output(ResponsePart::Text { text }) if text == "hello"
        ));
        assert_eq!(
            accumulators.read().await.raw_response,
            vec!["hello".to_string()]
        );
        assert_eq!(response.read().await.text(), "hello");
    }

    #[test]
    fn test_create_inference_client_openrouter() {
        std::env::set_var("OPENROUTER_API_KEY", "test-key");
        let result = create_inference_client("openrouter:openai/gpt-4o", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_create_inference_client_openrouter_responses() {
        std::env::set_var("OPENROUTER_API_KEY", "test-key");
        let result = create_inference_client("openrouter:resp:openai/o4-mini", None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_tool_schemas_empty() {
        let tools = HashMap::new();
        let tool_schemas = HashMap::new();
        let result = generate_tool_schemas(&tools, &tool_schemas, "anthropic:claude-3");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);
    }

    #[test]
    fn test_generate_tool_schemas_anthropic() {
        use crate::parsers::FieldDescription;

        let mut tools = HashMap::new();
        tools.insert(
            "get_weather".to_string(),
            ToolFunction::Sync(Box::new(|_args| Ok("sunny".to_string()))),
        );

        let mut tool_schemas = HashMap::new();
        tool_schemas.insert(
            "get_weather".to_string(),
            ToolSchemaInfo {
                name: "get_weather".to_string(),
                description: "Get the weather for a location".to_string(),
                fields: vec![FieldDescription {
                    name: "location".to_string(),
                    field_type: "string".to_string(),
                    description: "The city name".to_string(),
                    required: true,
                }],
            },
        );

        let result = generate_tool_schemas(&tools, &tool_schemas, "anthropic:claude-3");
        assert!(result.is_ok());

        let schemas = result.unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["name"], "get_weather");
        assert!(schemas[0]["input_schema"].is_object());
        assert_eq!(
            schemas[0]["input_schema"]["properties"]["location"]["type"],
            "string"
        );
    }

    #[test]
    fn test_generate_tool_schemas_openai() {
        use crate::parsers::FieldDescription;

        let mut tools = HashMap::new();
        tools.insert(
            "calculate".to_string(),
            ToolFunction::Sync(Box::new(|_args| Ok("42".to_string()))),
        );

        let mut tool_schemas = HashMap::new();
        tool_schemas.insert(
            "calculate".to_string(),
            ToolSchemaInfo {
                name: "calculate".to_string(),
                description: "Calculate a math expression".to_string(),
                fields: vec![FieldDescription {
                    name: "expression".to_string(),
                    field_type: "string".to_string(),
                    description: "Math expression to evaluate".to_string(),
                    required: true,
                }],
            },
        );

        let result = generate_tool_schemas(&tools, &tool_schemas, "openai:gpt-4o");
        assert!(result.is_ok());

        let schemas = result.unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["type"], "function");
        assert_eq!(schemas[0]["function"]["name"], "calculate");
        assert_eq!(
            schemas[0]["function"]["parameters"]["properties"]["expression"]["type"],
            "string"
        );
    }

    #[test]
    fn test_generate_tool_schemas_with_schema_info() {
        use crate::parsers::FieldDescription;

        let mut tools = HashMap::new();
        tools.insert(
            "search".to_string(),
            ToolFunction::Sync(Box::new(|_args| Ok("results".to_string()))),
        );

        let mut tool_schemas = HashMap::new();
        tool_schemas.insert(
            "search".to_string(),
            ToolSchemaInfo {
                name: "search".to_string(),
                description: "Search for documents".to_string(),
                fields: vec![
                    FieldDescription {
                        name: "query".to_string(),
                        field_type: "string".to_string(),
                        description: "Search query".to_string(),
                        required: true,
                    },
                    FieldDescription {
                        name: "limit".to_string(),
                        field_type: "integer".to_string(),
                        description: "Max results".to_string(),
                        required: false,
                    },
                    FieldDescription {
                        name: "ids".to_string(),
                        field_type: "Vec<i64>".to_string(),
                        description: "Optional document ids".to_string(),
                        required: false,
                    },
                ],
            },
        );

        let result = generate_tool_schemas(&tools, &tool_schemas, "anthropic:claude-3");
        assert!(result.is_ok());

        let schemas = result.unwrap();
        assert_eq!(schemas.len(), 1);
        assert_eq!(schemas[0]["name"], "search");
        assert_eq!(
            schemas[0]["input_schema"]["properties"]["ids"]["items"]["type"],
            "integer"
        );
    }

    #[test]
    fn test_generate_tool_schemas_unsupported_provider() {
        let mut tools = HashMap::new();
        tools.insert(
            "test".to_string(),
            ToolFunction::Sync(Box::new(|_args| Ok("ok".to_string()))),
        );
        let tool_schemas = HashMap::new();

        let result = generate_tool_schemas(&tools, &tool_schemas, "unsupported:model");
        assert!(result.is_err());
    }

    #[test]
    fn test_field_type_to_json_type() {
        assert_eq!(
            field_type_to_json_type("String"),
            serde_json::json!("string")
        );
        assert_eq!(field_type_to_json_type("i32"), serde_json::json!("integer"));
        assert_eq!(field_type_to_json_type("f64"), serde_json::json!("number"));
        assert_eq!(
            field_type_to_json_type("bool"),
            serde_json::json!("boolean")
        );
        assert_eq!(
            field_type_to_json_type("Vec<String>"),
            serde_json::json!("array")
        );
        assert_eq!(
            field_type_to_json_type("Option<i32>"),
            serde_json::json!("integer")
        );
        assert_eq!(
            field_type_to_json_type("alloc::vec::Vec<i64>"),
            serde_json::json!("array")
        );
        assert_eq!(
            field_type_to_json_type("CustomType"),
            serde_json::json!("string")
        );
    }

    #[test]
    fn test_field_type_array_items_detection() {
        let array_items = field_type_array_items("Vec<i64>").unwrap();
        assert_eq!(array_items["type"], "integer");

        let option_items = field_type_array_items("Option<Vec<String>>").unwrap();
        assert_eq!(option_items["type"], "string");

        let slice_items = field_type_array_items("&[bool]").unwrap();
        assert_eq!(slice_items["type"], "boolean");
    }
}
