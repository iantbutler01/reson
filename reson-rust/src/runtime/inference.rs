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
    OpenRouterClient,
};
use crate::schema::fix_output_schema_for_provider;
use crate::storage::Storage;
use crate::types::ChatMessage;
use crate::utils::ConversationMessage;

use super::{Accumulators, ToolFunction, ToolSchemaInfo};

/// Result from non-streaming LLM call
pub struct CallResult {
    pub parsed_value: serde_json::Value,
    pub raw_response: Option<String>,
    pub reasoning: Option<String>,
    /// Tool calls from the response (if any)
    pub tool_calls: Vec<serde_json::Value>,
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
    let provider = model.split(':').next().unwrap_or(model);

    // Get appropriate schema generator
    let generator = crate::schema::get_schema_generator(provider)?;

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

/// Create an inference client from model string
///
/// Format: `provider:model@param=value`
/// Examples:
/// - `anthropic:claude-3-5-sonnet-20241022`
/// - `openrouter:openai/gpt-4o@reasoning=high`
/// - `openai:gpt-4o`
pub fn create_inference_client(
    model_str: &str,
    api_key: Option<&str>,
) -> Result<Box<dyn InferenceClient>> {
    // Parse model string
    let parts: Vec<&str> = model_str.split(':').collect();
    if parts.len() != 2 {
        return Err(Error::NonRetryable(format!(
            "Invalid model string format: {}. Expected 'provider:model'",
            model_str
        )));
    }

    let provider = parts[0];
    let model_part = parts[1];

    // Parse parameters (e.g., @reasoning=1024)
    let mut reasoning = None;
    let model_name = if model_part.contains('@') {
        let model_parts: Vec<&str> = model_part.split('@').collect();

        for param in &model_parts[1..] {
            if param.starts_with("reasoning=") {
                reasoning = Some(param.strip_prefix("reasoning=").unwrap().to_string());
            }
        }

        model_parts[0].to_string()
    } else {
        model_part.to_string()
    };

    // Get API key from env if not provided
    let key = match api_key {
        Some(k) => k.to_string(),
        None => match provider {
            "anthropic" => std::env::var("ANTHROPIC_API_KEY")
                .map_err(|_| Error::NonRetryable("ANTHROPIC_API_KEY not set".to_string()))?,
            "openai" => std::env::var("OPENAI_API_KEY")
                .map_err(|_| Error::NonRetryable("OPENAI_API_KEY not set".to_string()))?,
            "openrouter" => std::env::var("OPENROUTER_API_KEY")
                .or_else(|_| std::env::var("OPENROUTER_KEY"))
                .map_err(|_| Error::NonRetryable("OPENROUTER_API_KEY not set".to_string()))?,
            "google-gemini" | "google-genai" | "gemini" => {
                std::env::var("GOOGLE_GEMINI_API_KEY")
                    .map_err(|_| Error::NonRetryable("GOOGLE_GEMINI_API_KEY not set".to_string()))?
            }
            _ => {
                return Err(Error::NonRetryable(format!(
                    "Unknown provider: {}",
                    provider
                )))
            }
        },
    };

    // Create client
    let client: Box<dyn InferenceClient> = match provider {
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
        "google-gemini" | "google-genai" | "gemini" => {
            let mut client = GoogleGenAIClient::new(key, model_name);
            if let Some(r) = reasoning {
                if let Ok(budget) = r.parse::<u32>() {
                    client = client.with_thinking_budget(budget);
                }
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
    _store: Arc<dyn Storage>,
    api_key: Option<&str>,
    system: Option<&str>,
    history: Option<Vec<ConversationMessage>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
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
    let provider = model.split(':').next().unwrap_or(model);
    let fixed_output_schema = output_schema.map(|mut schema| {
        fix_output_schema_for_provider(&mut schema, provider);
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
    };

    // Make API call
    let response = client.get_generation(&messages, &config).await?;

    // Check if response contains tool calls
    if response.has_tool_calls() {
        // Return tool calls - we need to format them for the runtime
        // Add _tool_name field so runtime.is_tool_call() works
        let formatted_tool_calls: Vec<serde_json::Value> = response
            .tool_calls
            .iter()
            .map(|tc| {
                let mut tool_call = tc.clone();
                // For Anthropic format: {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
                if let Some(name) = tc.get("name").and_then(|n| n.as_str()) {
                    if let Some(obj) = tool_call.as_object_mut() {
                        obj.insert("_tool_name".to_string(), serde_json::json!(name));
                        // Also copy input fields to top level for easier access
                        if let Some(input) = tc.get("input").cloned() {
                            if let Some(input_obj) = input.as_object() {
                                for (k, v) in input_obj {
                                    obj.insert(k.clone(), v.clone());
                                }
                            }
                        }
                    }
                }
                // For OpenAI format: {"id": "...", "function": {"name": "...", "arguments": "..."}}
                else if let Some(func) = tc.get("function") {
                    if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                        if let Some(obj) = tool_call.as_object_mut() {
                            obj.insert("_tool_name".to_string(), serde_json::json!(name));
                            // Parse arguments JSON and copy to top level
                            if let Some(args_str) = func.get("arguments").and_then(|a| a.as_str()) {
                                if let Ok(args) =
                                    serde_json::from_str::<serde_json::Value>(args_str)
                                {
                                    if let Some(args_obj) = args.as_object() {
                                        for (k, v) in args_obj {
                                            obj.insert(k.clone(), v.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                tool_call
            })
            .collect();

        // Return first tool call as parsed_value (for single tool call case)
        // and all tool calls in the tool_calls vec
        let parsed_value = if formatted_tool_calls.len() == 1 {
            formatted_tool_calls[0].clone()
        } else {
            serde_json::json!(formatted_tool_calls)
        };

        return Ok(CallResult {
            parsed_value,
            raw_response: Some(response.content),
            reasoning: response.reasoning,
            tool_calls: formatted_tool_calls,
        });
    }

    // Parse response as text/JSON
    let parsed_value = if !response.content.is_empty() {
        // Try to parse as JSON, fallback to string
        serde_json::from_str(&response.content)
            .unwrap_or_else(|_| serde_json::json!(response.content))
    } else {
        serde_json::Value::Null
    };

    Ok(CallResult {
        parsed_value,
        raw_response: Some(response.content),
        reasoning: response.reasoning,
        tool_calls: Vec::new(),
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
    _store: Arc<dyn Storage>,
    api_key: Option<&str>,
    system: Option<&str>,
    history: Option<Vec<ConversationMessage>>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
    _call_context: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>>,
    _accumulators: Arc<RwLock<Accumulators>>,
) -> Result<Pin<Box<dyn Stream<Item = Result<(String, serde_json::Value)>> + Send>>> {
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
    let provider = model.split(':').next().unwrap_or(model);
    let fixed_output_schema = output_schema.map(|mut schema| {
        fix_output_schema_for_provider(&mut schema, provider);
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
    };

    // Get streaming response
    let stream = client.connect_and_listen(&messages, &config).await?;

    // Transform stream to (chunk_type, value) tuples
    let transformed = stream.map(move |chunk_result| match chunk_result {
        Ok(chunk) => {
            use crate::providers::StreamChunk;
            match chunk {
                StreamChunk::Content(text) => Ok(("content".to_string(), serde_json::json!(text))),
                StreamChunk::Reasoning(text) => {
                    Ok(("reasoning".to_string(), serde_json::json!(text)))
                }
                StreamChunk::Signature(sig) => {
                    Ok(("signature".to_string(), serde_json::json!(sig)))
                }
                StreamChunk::ToolCallComplete(tool) => Ok(("tool_call_complete".to_string(), tool)),
                StreamChunk::ToolCallPartial(tool) => Ok(("tool_call_partial".to_string(), tool)),
                StreamChunk::Usage {
                    input_tokens,
                    output_tokens,
                    cached_tokens,
                } => Ok((
                    "usage".to_string(),
                    serde_json::json!({
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "cached_tokens": cached_tokens
                    }),
                )),
            }
        }
        Err(e) => Err(e),
    });

    Ok(Box::pin(transformed))
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
    fn test_create_inference_client_openrouter() {
        std::env::set_var("OPENROUTER_API_KEY", "test-key");
        let result = create_inference_client("openrouter:openai/gpt-4o", None);
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
