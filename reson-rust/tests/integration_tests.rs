//! Comprehensive integration tests for all providers
//!
//! These tests require API keys to be set in environment variables:
//! - ANTHROPIC_API_KEY: For Anthropic Claude direct API
//! - OPENAI_API_KEY: For OpenAI models
//! - GOOGLE_GEMINI_API_KEY: For Google Gemini models
//! - OPENROUTER_API_KEY: For OpenRouter proxy
//!
//! Run with: cargo test --test integration_tests -- --ignored
//! Or specific test: cargo test --test integration_tests test_anthropic_simple -- --ignored

use reson_agentic::agentic;
use reson_agentic::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient, OAIClient,
    OpenRouterClient,
};
use reson_agentic::runtime::{Runtime, ToolFunction};
use reson_agentic::schema::{
    AnthropicSchemaGenerator, GoogleSchemaGenerator, OpenAISchemaGenerator, SchemaGenerator,
};
use reson_agentic::types::{ChatMessage, Provider, ToolCall, ToolResult};
use reson_agentic::utils::ConversationMessage;
use reson_agentic::Tool;
use serde::{Deserialize, Serialize};
use std::env;

// ============================================================================
// Helper Functions
// ============================================================================

fn get_anthropic_key() -> Option<String> {
    env::var("ANTHROPIC_API_KEY").ok()
}

fn get_openai_key() -> Option<String> {
    env::var("OPENAI_API_KEY").ok()
}

fn get_google_key() -> Option<String> {
    env::var("GOOGLE_GEMINI_API_KEY").ok()
}

fn get_openrouter_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY").ok()
}

/// Common parameters for add_numbers tool
fn add_numbers_params() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"}
        },
        "required": ["a", "b"]
    })
}

/// Generate add_numbers tool schema for Anthropic
fn anthropic_add_tool() -> serde_json::Value {
    AnthropicSchemaGenerator.generate_schema(
        "add_numbers",
        "Add two numbers together",
        add_numbers_params(),
    )
}

/// Generate add_numbers tool schema for OpenAI/OpenRouter
fn openai_add_tool() -> serde_json::Value {
    OpenAISchemaGenerator.generate_schema(
        "add_numbers",
        "Add two numbers together",
        add_numbers_params(),
    )
}

/// Generate add_numbers tool schema for Google
fn google_add_tool() -> serde_json::Value {
    GoogleSchemaGenerator.generate_schema(
        "add_numbers",
        "Add two numbers together",
        add_numbers_params(),
    )
}

/// Common parameters for multiply_numbers tool
fn multiply_numbers_params() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"}
        },
        "required": ["a", "b"]
    })
}

/// Generate multiply_numbers tool schema for OpenAI/OpenRouter
fn openai_multiply_tool() -> serde_json::Value {
    OpenAISchemaGenerator.generate_schema(
        "multiply_numbers",
        "Multiply two numbers together",
        multiply_numbers_params(),
    )
}

/// Add two numbers together
#[derive(Debug, Tool, Serialize, Deserialize)]
struct AddNumbers {
    /// First number
    a: i64,
    /// Second number
    b: i64,
}

/// Multiply two numbers together
#[derive(Debug, Tool, Serialize, Deserialize)]
struct MultiplyNumbers {
    /// First number
    a: i64,
    /// Second number
    b: i64,
}

async fn register_add_numbers_tool(runtime: &Runtime) -> reson_agentic::error::Result<()> {
    runtime
        .register_tool_with_schema(
            AddNumbers::tool_name(),
            AddNumbers::description(),
            AddNumbers::schema(),
            ToolFunction::Sync(Box::new(|args| {
                let a = args.get("a").and_then(parse_i64_value).unwrap_or(0);
                let b = args.get("b").and_then(parse_i64_value).unwrap_or(0);
                Ok((a + b).to_string())
            })),
        )
        .await
}

async fn register_multiply_numbers_tool(runtime: &Runtime) -> reson_agentic::error::Result<()> {
    runtime
        .register_tool_with_schema(
            MultiplyNumbers::tool_name(),
            MultiplyNumbers::description(),
            MultiplyNumbers::schema(),
            ToolFunction::Sync(Box::new(|args| {
                let a = args.get("a").and_then(parse_i64_value).unwrap_or(0);
                let b = args.get("b").and_then(parse_i64_value).unwrap_or(0);
                Ok((a * b).to_string())
            })),
        )
        .await
}

fn parse_i64_value(value: &serde_json::Value) -> Option<i64> {
    match value {
        serde_json::Value::Number(num) => num.as_i64(),
        serde_json::Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

fn parse_tool_call_args(value: &serde_json::Value) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Object(_) => Some(value.clone()),
        serde_json::Value::String(s) => serde_json::from_str::<serde_json::Value>(s)
            .ok()
            .filter(|v| v.is_object()),
        _ => None,
    }
}

fn extract_tool_call_args(
    tool_call: &serde_json::Value,
) -> reson_agentic::error::Result<serde_json::Value> {
    let direct_args = tool_call
        .get("arguments")
        .or_else(|| tool_call.get("input"))
        .or_else(|| tool_call.get("_tool_args"));
    if let Some(args) = direct_args.and_then(parse_tool_call_args) {
        return Ok(args);
    }

    if let Some(function) = tool_call.get("function") {
        let nested_args = function
            .get("arguments")
            .or_else(|| function.get("input"));
        if let Some(args) = nested_args.and_then(parse_tool_call_args) {
            return Ok(args);
        }
    }

    Err(reson_agentic::error::Error::NonRetryable(
        "Missing tool arguments".to_string(),
    ))
}

fn tool_call_name(tool_call: &serde_json::Value) -> Option<&str> {
    tool_call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .or_else(|| {
            tool_call
                .get("name")
                .or_else(|| tool_call.get("_tool_name"))
                .and_then(|n| n.as_str())
        })
}

fn require_i64_field(
    value: &serde_json::Value,
    field: &str,
) -> reson_agentic::error::Result<i64> {
    if let Some(raw) = value.get(field) {
        return parse_i64_value(raw).ok_or_else(|| {
            reson_agentic::error::Error::NonRetryable(format!(
                "Invalid '{}' value: {}",
                field, raw
            ))
        });
    }

    let args = extract_tool_call_args(value)?;
    let raw = args.get(field).ok_or_else(|| {
        reson_agentic::error::Error::NonRetryable(format!("Missing '{}' field", field))
    })?;
    parse_i64_value(raw).ok_or_else(|| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Invalid '{}' value: {}",
            field, raw
        ))
    })
}

fn require_tool_call_id(tool_call: &serde_json::Value) -> reson_agentic::error::Result<String> {
    let id = tool_call
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| {
            reson_agentic::error::Error::NonRetryable("Missing tool call id".to_string())
        })?;
    Ok(id.to_string())
}

#[agentic(model = "openai:resp:gpt-4o-mini")]
async fn openai_responses_simple_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    let response = runtime
        .run(
            Some(&prompt),
            Some("You are helpful. Be concise."),
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(100),
            None,
            None,
        )
        .await?;

    Ok(response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string()))
}

#[agentic(model = "openai:resp:gpt-4o-mini")]
async fn openai_responses_tool_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    register_add_numbers_tool(&runtime).await?;

    let tool_call = runtime
        .run(
            Some(&prompt),
            None,
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(
            "Expected tool call".to_string(),
        ));
    }

    let tool_name = runtime
        .get_tool_name(&tool_call)
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool name".to_string()))?;
    if tool_name != "add_numbers" {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool name: {}",
            tool_name
        )));
    }

    let tool_use_id = require_tool_call_id(&tool_call)?;
    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;
    if a != 25 || b != 35 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected add_numbers args: a={}, b={}",
            a, b
        )));
    }

    let tool_output = runtime.execute_tool(&tool_call).await?;
    let output_value = tool_output.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Tool output should be integer, got: {}",
            tool_output
        ))
    })?;
    if output_value != 60 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool output: {}",
            tool_output
        )));
    }

    let mut tc = ToolCall::new(&tool_name, serde_json::json!({"a": a, "b": b}));
    tc.tool_use_id = tool_use_id.clone();

    let history = vec![
        ConversationMessage::Chat(ChatMessage::user(prompt)),
        ConversationMessage::ToolCall(tc),
        ConversationMessage::ToolResult(
            ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
        ),
    ];

    let response = runtime
        .run(
            Some("What was the result? Just the number."),
            Some("Respond with only the number."),
            Some(history),
            None,
            None,
            Some(0.0),
            None,
            Some(128),
            None,
            None,
        )
        .await?;

    let response_str = response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string());
    let response_value = response_str.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Response should be integer, got: {}",
            response_str
        ))
    })?;
    if response_value != 60 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected response: {}",
            response_str
        )));
    }

    Ok(response_str)
}

#[agentic(model = "openai:resp:gpt-4o-mini")]
async fn openai_responses_tool_stream_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<bool> {
    use futures::StreamExt;

    register_add_numbers_tool(&runtime).await?;

    let mut stream = runtime
        .run_stream(
            Some(&prompt),
            None,
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    let mut saw_tool = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok((chunk_type, value)) => {
                if chunk_type == "tool_call_complete" {
                    let id = value
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if id.is_empty() {
                        return Err(reson_agentic::error::Error::NonRetryable(
                            "Missing tool call id in stream".to_string(),
                        ));
                    }

                    let name = tool_call_name(&value);
                    if name != Some("add_numbers") {
                        return Err(reson_agentic::error::Error::NonRetryable(format!(
                            "Unexpected tool name in stream: {:?}",
                            name
                        )));
                    }

                    let a = require_i64_field(&value, "a")?;
                    let b = require_i64_field(&value, "b")?;
                    if a != 25 || b != 35 {
                        return Err(reson_agentic::error::Error::NonRetryable(format!(
                            "Unexpected stream args: a={}, b={}",
                            a, b
                        )));
                    }

                    saw_tool = true;
                }
            }
            Err(e) => {
                return Err(reson_agentic::error::Error::NonRetryable(format!(
                    "Stream error: {}",
                    e
                )));
            }
        }
    }

    Ok(saw_tool)
}

#[agentic(model = "openai:resp:gpt-4o-mini")]
async fn openai_responses_multi_turn_tool_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    register_add_numbers_tool(&runtime).await?;
    register_multiply_numbers_tool(&runtime).await?;

    let tool_call = runtime
        .run(
            Some(&prompt),
            None,
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(
            "Expected add_numbers tool call".to_string(),
        ));
    }

    let tool_name = runtime
        .get_tool_name(&tool_call)
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool name".to_string()))?;
    if tool_name != "add_numbers" {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool name: {}",
            tool_name
        )));
    }
    let tool_use_id = require_tool_call_id(&tool_call)?;

    let tool_output = runtime.execute_tool(&tool_call).await?;

    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;
    if a != 10 || b != 20 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected add_numbers args: a={}, b={}",
            a, b
        )));
    }

    let add_output_value = tool_output.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Tool output should be integer, got: {}",
            tool_output
        ))
    })?;
    if add_output_value != 30 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected add_numbers output: {}",
            tool_output
        )));
    }

    let mut tc = ToolCall::new(&tool_name, serde_json::json!({ "a": a, "b": b }));
    tc.tool_use_id = tool_use_id.clone();

    let mut history = vec![
        ConversationMessage::Chat(ChatMessage::user(prompt)),
        ConversationMessage::ToolCall(tc),
        ConversationMessage::ToolResult(
            ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
        ),
    ];

    let followup_prompt = "Now multiply that result by 3 using multiply_numbers.";
    let tool_call = runtime
        .run(
            Some(followup_prompt),
            None,
            Some(history.clone()),
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(
            "Expected multiply_numbers tool call".to_string(),
        ));
    }

    let tool_name = runtime
        .get_tool_name(&tool_call)
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool name".to_string()))?;
    if tool_name != "multiply_numbers" {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool name: {}",
            tool_name
        )));
    }
    let tool_use_id = require_tool_call_id(&tool_call)?;

    let tool_output = runtime.execute_tool(&tool_call).await?;

    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;
    if a != add_output_value || b != 3 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected multiply_numbers args: a={}, b={}",
            a, b
        )));
    }

    let multiply_output_value = tool_output.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Tool output should be integer, got: {}",
            tool_output
        ))
    })?;
    if multiply_output_value != 90 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected multiply_numbers output: {}",
            tool_output
        )));
    }

    let mut tc = ToolCall::new(&tool_name, serde_json::json!({ "a": a, "b": b }));
    tc.tool_use_id = tool_use_id.clone();

    history.push(ConversationMessage::Chat(ChatMessage::user(
        followup_prompt,
    )));
    history.push(ConversationMessage::ToolCall(tc));
    history.push(ConversationMessage::ToolResult(
        ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
    ));

    let response = runtime
        .run(
            Some("What is the final result? Just the number."),
            Some("Respond with only the number."),
            Some(history),
            None,
            None,
            Some(0.0),
            None,
            Some(128),
            None,
            None,
        )
        .await?;

    let response_str = response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string());
    let response_value = response_str.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Response should be integer, got: {}",
            response_str
        ))
    })?;
    if response_value != 90 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected response: {}",
            response_str
        )));
    }

    Ok(response_str)
}

#[agentic(model = "openrouter:resp:openai/o4-mini")]
async fn openrouter_responses_simple_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    let response = runtime
        .run(
            Some(&prompt),
            None,
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(100),
            None,
            None,
        )
        .await?;

    Ok(response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string()))
}

#[agentic(model = "openrouter:resp:openai/o4-mini")]
async fn openrouter_responses_tool_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    register_add_numbers_tool(&runtime).await?;

    let tool_call = runtime
        .run(
            Some(&prompt),
            None,
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(
            "Expected tool call".to_string(),
        ));
    }

    let tool_name = runtime
        .get_tool_name(&tool_call)
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool name".to_string()))?;
    if tool_name != "add_numbers" {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool name: {}",
            tool_name
        )));
    }

    let tool_use_id = require_tool_call_id(&tool_call)?;
    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;
    if a != 25 || b != 35 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected add_numbers args: a={}, b={}",
            a, b
        )));
    }

    let tool_output = runtime.execute_tool(&tool_call).await?;
    let output_value = tool_output.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Tool output should be integer, got: {}",
            tool_output
        ))
    })?;
    if output_value != 60 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool output: {}",
            tool_output
        )));
    }

    let mut tc = ToolCall::new(&tool_name, serde_json::json!({"a": a, "b": b}));
    tc.tool_use_id = tool_use_id.clone();

    let history = vec![
        ConversationMessage::Chat(ChatMessage::user(prompt)),
        ConversationMessage::ToolCall(tc),
        ConversationMessage::ToolResult(
            ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
        ),
    ];

    let response = runtime
        .run(
            Some("What was the result? Just the number."),
            Some("Respond with only the number."),
            Some(history),
            None,
            None,
            Some(0.0),
            None,
            Some(128),
            None,
            None,
        )
        .await?;

    let response_str = response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string());
    let response_value = response_str.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Response should be integer, got: {}",
            response_str
        ))
    })?;
    if response_value != 60 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected response: {}",
            response_str
        )));
    }

    Ok(response_str)
}

#[agentic(model = "openrouter:resp:openai/o4-mini")]
async fn openrouter_responses_multi_turn_tool_agent(
    prompt: String,
    runtime: Runtime,
) -> reson_agentic::error::Result<String> {
    register_add_numbers_tool(&runtime).await?;
    register_multiply_numbers_tool(&runtime).await?;

    let tool_call = runtime
        .run(
            Some(&prompt),
            None,
            None,
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(
            "Expected add_numbers tool call".to_string(),
        ));
    }

    let tool_name = runtime
        .get_tool_name(&tool_call)
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool name".to_string()))?;
    if tool_name != "add_numbers" {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool name: {}",
            tool_name
        )));
    }
    let tool_use_id = require_tool_call_id(&tool_call)?;

    let tool_output = runtime.execute_tool(&tool_call).await?;

    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;
    if a != 10 || b != 20 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected add_numbers args: a={}, b={}",
            a, b
        )));
    }

    let add_output_value = tool_output.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Tool output should be integer, got: {}",
            tool_output
        ))
    })?;
    if add_output_value != 30 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected add_numbers output: {}",
            tool_output
        )));
    }

    let mut tc = ToolCall::new(&tool_name, serde_json::json!({ "a": a, "b": b }));
    tc.tool_use_id = tool_use_id.clone();

    let mut history = vec![
        ConversationMessage::Chat(ChatMessage::user(prompt)),
        ConversationMessage::ToolCall(tc),
        ConversationMessage::ToolResult(
            ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
        ),
    ];

    let followup_prompt = "Now multiply that result by 3 using multiply_numbers.";
    let tool_call = runtime
        .run(
            Some(followup_prompt),
            None,
            Some(history.clone()),
            None,
            None,
            Some(0.0),
            None,
            Some(512),
            None,
            None,
        )
        .await?;

    if !runtime.is_tool_call(&tool_call) {
        return Err(reson_agentic::error::Error::NonRetryable(
            "Expected multiply_numbers tool call".to_string(),
        ));
    }

    let tool_name = runtime
        .get_tool_name(&tool_call)
        .ok_or_else(|| reson_agentic::error::Error::NonRetryable("Missing tool name".to_string()))?;
    if tool_name != "multiply_numbers" {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected tool name: {}",
            tool_name
        )));
    }
    let tool_use_id = require_tool_call_id(&tool_call)?;

    let tool_output = runtime.execute_tool(&tool_call).await?;

    let a = require_i64_field(&tool_call, "a")?;
    let b = require_i64_field(&tool_call, "b")?;
    if a != add_output_value || b != 3 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected multiply_numbers args: a={}, b={}",
            a, b
        )));
    }

    let multiply_output_value = tool_output.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Tool output should be integer, got: {}",
            tool_output
        ))
    })?;
    if multiply_output_value != 90 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected multiply_numbers output: {}",
            tool_output
        )));
    }

    let mut tc = ToolCall::new(&tool_name, serde_json::json!({ "a": a, "b": b }));
    tc.tool_use_id = tool_use_id.clone();

    history.push(ConversationMessage::Chat(ChatMessage::user(
        followup_prompt,
    )));
    history.push(ConversationMessage::ToolCall(tc));
    history.push(ConversationMessage::ToolResult(
        ToolResult::success(&tool_use_id, &tool_output).with_tool_name(&tool_name),
    ));

    let response = runtime
        .run(
            Some("What is the final result? Just the number."),
            Some("Respond with only the number."),
            Some(history),
            None,
            None,
            Some(0.0),
            None,
            Some(128),
            None,
            None,
        )
        .await?;

    let response_str = response
        .as_str()
        .map(|s| s.to_string())
        .unwrap_or_else(|| response.to_string());
    let response_value = response_str.trim().parse::<i64>().map_err(|_| {
        reson_agentic::error::Error::NonRetryable(format!(
            "Response should be integer, got: {}",
            response_str
        ))
    })?;
    if response_value != 90 {
        return Err(reson_agentic::error::Error::NonRetryable(format!(
            "Unexpected response: {}",
            response_str
        )));
    }

    Ok(response_str)
}

// ============================================================================
// Anthropic Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_simple_generation() {
    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::system(
            "You are a helpful assistant. Respond concisely.",
        )),
        ConversationMessage::Chat(ChatMessage::user("What is 2+2? Just give the number.")),
    ];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(100)
        .with_temperature(0.0);

    let response = client.get_generation(&messages, &config).await;
    if let Err(err) = &response {
        eprintln!("Google simple generation error: {}", err);
    }
    let response = response.unwrap();

    println!("Response: {}", response.content);
    assert!(!response.content.is_empty());
    assert!(response.content.contains("4"));
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
}

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_with_tools() {
    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the add_numbers tool to add 15 and 27",
    ))];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(1024)
        .with_tools(vec![anthropic_add_tool()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await;
    if let Err(err) = &response {
        eprintln!("Google tools error: {}", err);
    }
    let response = response.unwrap();

    println!("Response: {:?}", response);

    // Should have tool calls
    assert!(!response.tool_calls.is_empty(), "Expected tool call");

    // Verify tool call structure
    let tool_call = &response.tool_calls[0];
    assert_eq!(
        tool_call.get("name").and_then(|v| v.as_str()),
        Some("add_numbers")
    );

    let input = tool_call.get("input").expect("Tool call should have input");
    assert_eq!(input.get("a").and_then(|v| v.as_i64()), Some(15));
    assert_eq!(input.get("b").and_then(|v| v.as_i64()), Some(27));
}

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_multi_turn_tool_conversation() {
    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    // Turn 1: User asks to add numbers
    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Please add 10 and 20 using the add_numbers tool",
    ))];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(1024)
        .with_tools(vec![anthropic_add_tool()])
        .with_native_tools(true);

    let response1 = client.get_generation(&messages, &config).await.unwrap();
    println!("Turn 1 response: {:?}", response1);
    assert!(!response1.tool_calls.is_empty());

    // Turn 2: Provide tool result and ask for final answer
    let tool_call_json = &response1.tool_calls[0];
    let tool_use_id = tool_call_json
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let tool_name = tool_call_json
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("add_numbers");
    let tool_input = tool_call_json
        .get("input")
        .cloned()
        .unwrap_or(serde_json::json!({}));

    let mut messages2 = messages.clone();
    // Add assistant's tool call as a ToolCall (NOT an empty assistant message)
    let mut tc = ToolCall::new(tool_name, tool_input);
    tc.tool_use_id = tool_use_id.to_string();
    messages2.push(ConversationMessage::ToolCall(tc));
    // Add tool result
    messages2.push(ConversationMessage::ToolResult(
        ToolResult::success(tool_use_id, "30").with_tool_name("add_numbers"),
    ));
    messages2.push(ConversationMessage::Chat(ChatMessage::user(
        "Great! What was the result?",
    )));

    let response2 = client.get_generation(&messages2, &config).await.unwrap();
    println!("Turn 2 response: {}", response2.content);

    // Should get a text response mentioning 30
    assert!(response2.content.contains("30"));
}

// ============================================================================
// OpenAI Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_openai_simple_generation() {
    let api_key = get_openai_key().expect("OPENAI_API_KEY not set");
    let client = OAIClient::new(api_key, "gpt-4o-mini");

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::system("You are helpful. Be concise.")),
        ConversationMessage::Chat(ChatMessage::user("What is 3+3? Just the number.")),
    ];

    let config = GenerationConfig::new("gpt-4o-mini")
        .with_max_tokens(100)
        .with_temperature(0.0);

    let response = client.get_generation(&messages, &config).await;
    if let Err(err) = &response {
        eprintln!("Google thinking error: {}", err);
    }
    let response = response.unwrap();

    println!("Response: {}", response.content);
    assert!(!response.content.is_empty());
    assert!(response.content.contains("6"));
}

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_openai_with_tools() {
    let api_key = get_openai_key().expect("OPENAI_API_KEY not set");
    let client = OAIClient::new(api_key, "gpt-4o-mini");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the add_numbers tool to calculate 100 + 200",
    ))];

    let config = GenerationConfig::new("gpt-4o-mini")
        .with_max_tokens(1024)
        .with_tools(vec![openai_add_tool()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);
    assert!(!response.tool_calls.is_empty());
}

// ============================================================================
// OpenAI Responses Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_openai_responses_simple_generation() {
    let _ = get_openai_key().expect("OPENAI_API_KEY not set");

    let response = openai_responses_simple_agent(
        "What is 3+3? Just the number.".to_string(),
    )
    .await
    .unwrap();

    println!("Response: {}", response);
    let value = response.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 6);
}

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_openai_responses_with_tools() {
    let prompt = "Use the add_numbers tool to calculate 25 + 35.";
    let _ = get_openai_key().expect("OPENAI_API_KEY not set");

    let response = openai_responses_tool_agent(prompt.to_string())
        .await
        .unwrap();

    println!("Response: {}", response);
    let value = response.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 60);
}

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_openai_responses_with_tools_streaming() {
    let _ = get_openai_key().expect("OPENAI_API_KEY not set");

    let saw_tool = openai_responses_tool_stream_agent(
        "You must use the add_numbers tool to calculate 25 + 35. Do not calculate it yourself - use the tool.".to_string(),
    )
    .await
    .unwrap();

    assert!(saw_tool, "Expected tool call via streaming");
}

#[tokio::test]
#[ignore = "Requires OPENAI_API_KEY"]
async fn test_openai_responses_multi_turn_with_tools() {
    let _ = get_openai_key().expect("OPENAI_API_KEY not set");

    let response = openai_responses_multi_turn_tool_agent(
        "Use the add_numbers tool to add 10 and 20.".to_string(),
    )
    .await
    .unwrap();

    println!("Response: {}", response);
    let value = response.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 90);
}

// ============================================================================
// Google GenAI Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_simple_generation() {
    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-flash-latest");

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::system("Be concise.")),
        ConversationMessage::Chat(ChatMessage::user("What is 5+5? Just the number.")),
    ];

    let config = GenerationConfig::new("gemini-flash-latest")
        .with_max_tokens(100)
        .with_temperature(0.0);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {}", response.content);
    assert!(!response.content.is_empty());
    assert!(response.content.contains("10"));
}

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_with_tools() {
    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-flash-latest");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the add_numbers tool to add 50 and 75",
    ))];

    let config = GenerationConfig::new("gemini-flash-latest")
        .with_max_tokens(1024)
        .with_tools(vec![google_add_tool()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);
    assert!(!response.tool_calls.is_empty());

    let tool_call = &response.tool_calls[0];
    let name = tool_call
        .get("name")
        .or_else(|| tool_call.get("_tool_name"))
        .and_then(|v| v.as_str());
    assert_eq!(name, Some("add_numbers"));
}

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_with_thinking() {
    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client =
        GoogleGenAIClient::new(api_key, "gemini-flash-latest").with_thinking_budget(1024);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What are the prime factors of 360? Think through this step by step.",
    ))];

    let config = GenerationConfig::new("gemini-flash-latest").with_max_tokens(2048);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Content: {}", response.content);
    if let Some(reasoning) = &response.reasoning {
        println!("Reasoning: {}", reasoning);
    }

    assert!(!response.content.is_empty());
    // Should mention prime factors
    assert!(
        response.content.contains("2")
            || response.content.contains("3")
            || response.content.contains("5"),
        "Response should mention prime factors"
    );
}

// ============================================================================
// OpenRouter Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_anthropic_model() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::system("Be very concise.")),
        ConversationMessage::Chat(ChatMessage::user("What is 7+7? Just number.")),
    ];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(100)
        .with_temperature(0.0);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {}", response.content);
    assert!(!response.content.is_empty());
    assert!(response.content.contains("14"));
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_with_tools() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "You must use the add_numbers tool to calculate 25 + 35. Do not calculate it yourself - use the tool.",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![openai_add_tool()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);
    assert!(!response.tool_calls.is_empty());
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_with_tools_streaming() {
    use futures::StreamExt;
    use tokio::time::{timeout, Duration};

    let mut last_error = None;

    for attempt in 1..=3 {
        let attempt_result = timeout(Duration::from_secs(20), async {
            let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
            let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

            let messages = vec![ConversationMessage::Chat(ChatMessage::user(
                "You must use the add_numbers tool to calculate 25 + 35. Do not calculate it yourself - use the tool.",
            ))];

            let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
                .with_max_tokens(1024)
                .with_tools(vec![openai_add_tool()])
                .with_native_tools(true);

            let mut stream = client
                .connect_and_listen(&messages, &config)
                .await
                .map_err(|e| format!("stream connect error: {}", e))?;

            let mut tool_calls: Vec<serde_json::Value> = Vec::new();
            let mut content = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => match chunk {
                        reson_agentic::providers::StreamChunk::Content(text) => {
                            print!("{}", text);
                            content.push_str(&text);
                        }
                        reson_agentic::providers::StreamChunk::ToolCallComplete(tc) => {
                            println!("Tool call complete: {:?}", tc);
                            tool_calls.push(tc);
                        }
                        reson_agentic::providers::StreamChunk::Usage {
                            input_tokens,
                            output_tokens,
                            ..
                        } => {
                            println!("Usage: {} in, {} out", input_tokens, output_tokens);
                        }
                        _ => {}
                    },
                    Err(e) => {
                        return Err(format!("stream error: {}", e));
                    }
                }
            }

            println!("\nTool calls: {:?}", tool_calls);
            Ok::<Vec<serde_json::Value>, String>(tool_calls)
        })
        .await;

        match attempt_result {
            Ok(Ok(tool_calls)) => {
                let saw_add = tool_calls
                    .iter()
                    .any(|tc| tool_call_name(tc) == Some("add_numbers"));
                if saw_add {
                    return;
                }
                last_error = Some(format!(
                    "attempt {}: missing add_numbers tool call",
                    attempt
                ));
            }
            Ok(Err(err)) => {
                last_error = Some(format!("attempt {}: {}", attempt, err));
            }
            Err(_) => {
                last_error = Some(format!("attempt {} timed out", attempt));
            }
        }
    }

    panic!(
        "OpenRouter streaming failed: {}",
        last_error.unwrap_or_else(|| "unknown error".to_string())
    );
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_openai_model() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "openai/gpt-4o-mini", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What is the capital of France? One word.",
    ))];

    let config = GenerationConfig::new("openai/gpt-4o-mini")
        .with_max_tokens(50)
        .with_temperature(0.0);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {}", response.content);
    assert!(response.content.to_lowercase().contains("paris"));
}

// ============================================================================
// OpenRouter Responses Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_responses_simple_generation() {
    let _ = get_openrouter_key().expect("OPENROUTER_API_KEY not set");

    let response = openrouter_responses_simple_agent(
        "What is the capital of France? One word.".to_string(),
    )
    .await
    .unwrap();

    println!("Response: {}", response);
    assert_eq!(response.trim().to_lowercase(), "paris");
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_responses_with_tools() {
    let _ = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let prompt = "Use the add_numbers tool to calculate 25 + 35.";

    let response = openrouter_responses_tool_agent(prompt.to_string())
        .await
        .unwrap();

    println!("Response: {}", response);
    let value = response.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 60);
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_responses_multi_turn_with_tools() {
    let _ = get_openrouter_key().expect("OPENROUTER_API_KEY not set");

    let response = openrouter_responses_multi_turn_tool_agent(
        "Use the add_numbers tool to add 10 and 20.".to_string(),
    )
    .await
    .unwrap();

    println!("Response: {}", response);
    let value = response.trim().parse::<i64>().expect("Response should be an integer");
    assert_eq!(value, 90);
}

// ============================================================================
// Multi-Turn Conversation Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_5_turn_tool_conversation() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![openai_add_tool(), openai_multiply_tool()])
        .with_native_tools(true);

    // Start conversation
    let mut history: Vec<ConversationMessage> = vec![ConversationMessage::Chat(ChatMessage::user(
        "I need you to: 1) Add 10 and 20, 2) Then multiply that result by 3. Use the tools.",
    ))];

    let mut turn = 0;
    let max_turns = 5;

    while turn < max_turns {
        turn += 1;
        println!("\n--- Turn {} ---", turn);

        let response = client.get_generation(&history, &config).await.unwrap();

        if response.tool_calls.is_empty() {
            println!("Final response: {}", response.content);
            // Should eventually get final answer (90)
            assert!(
                response.content.contains("90") || response.content.contains("ninety"),
                "Final answer should be 90"
            );
            break;
        }

        // Process tool call
        let tool_call = &response.tool_calls[0];
        let tool_name = tool_call
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                tool_call
                    .get("name")
                    .or_else(|| tool_call.get("_tool_name"))
                    .and_then(|v| v.as_str())
            })
            .unwrap_or("unknown");
        let tool_id = tool_call
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let input = extract_tool_call_args(tool_call).unwrap_or_else(|_| serde_json::json!({}));

        println!("Tool call: {} with input: {:?}", tool_name, input);

        // Execute tool
        let result = match tool_name {
            "add_numbers" => {
                let a = input.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                let b = input.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                (a + b).to_string()
            }
            "multiply_numbers" => {
                let a = input.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                let b = input.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                (a * b).to_string()
            }
            _ => "Unknown tool".to_string(),
        };

        println!("Tool result: {}", result);

        let mut tc = ToolCall::new(tool_name, input.clone());
        tc.tool_use_id = tool_id.to_string();
        history.push(ConversationMessage::ToolCall(tc));
        history.push(ConversationMessage::ToolResult(
            ToolResult::success(tool_id, &result).with_tool_name(tool_name),
        ));
    }

    assert!(turn <= max_turns, "Conversation took too many turns");
}

// ============================================================================
// Streaming Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_streaming() {
    use futures::StreamExt;

    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Count from 1 to 5, one number per line.",
    ))];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(100)
        .with_temperature(0.0);

    let stream = client.connect_and_listen(&messages, &config).await;
    if let Err(err) = &stream {
        eprintln!("Google streaming connect error: {}", err);
    }
    let mut stream = stream.unwrap();

    let mut full_content = String::new();
    let mut chunk_count = 0;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                chunk_count += 1;
                match chunk {
                    reson_agentic::providers::StreamChunk::Content(text) => {
                        print!("{}", text);
                        full_content.push_str(&text);
                    }
                    reson_agentic::providers::StreamChunk::Usage {
                        input_tokens,
                        output_tokens,
                        ..
                    } => {
                        println!("\nUsage: {} in, {} out", input_tokens, output_tokens);
                    }
                    _ => {}
                }
            }
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\nTotal chunks: {}", chunk_count);
    assert!(chunk_count > 0, "Should receive streaming chunks");
    assert!(full_content.contains("1") && full_content.contains("5"));
}

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_streaming() {
    use futures::StreamExt;

    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-flash-latest");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "List the days of the week, one per line.",
    ))];

    let config = GenerationConfig::new("gemini-flash-latest").with_max_tokens(200);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut full_content = String::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                reson_agentic::providers::StreamChunk::Content(text) => {
                    print!("{}", text);
                    full_content.push_str(&text);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!();
    assert!(full_content.to_lowercase().contains("monday"));
    assert!(full_content.to_lowercase().contains("sunday"));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_invalid_api_key_anthropic() {
    let client = AnthropicClient::new("invalid-key", "claude-haiku-4-5-20251001");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
    let config = GenerationConfig::new("claude-haiku-4-5-20251001");

    let result = client.get_generation(&messages, &config).await;
    assert!(result.is_err(), "Should fail with invalid API key");
}

#[tokio::test]
async fn test_invalid_api_key_google() {
    let client = GoogleGenAIClient::new("invalid-key", "gemini-flash-latest");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
    let config = GenerationConfig::new("gemini-flash-latest");

    let result = client.get_generation(&messages, &config).await;
    assert!(result.is_err(), "Should fail with invalid API key");
}

// ============================================================================
// Provider Detection Tests
// ============================================================================

#[test]
fn test_provider_from_model_string() {
    let (provider, model) = Provider::from_model_string("anthropic:claude-3-opus").unwrap();
    assert_eq!(provider, Provider::Anthropic);
    assert_eq!(model, "claude-3-opus");

    let (provider, model) = Provider::from_model_string("openai:gpt-4").unwrap();
    assert_eq!(provider, Provider::OpenAI);
    assert_eq!(model, "gpt-4");

    let (provider, model) = Provider::from_model_string("openai:resp:gpt-4").unwrap();
    assert_eq!(provider, Provider::OpenAIResponses);
    assert_eq!(model, "gpt-4");

    let (provider, model) = Provider::from_model_string("google-genai:gemini-pro").unwrap();
    assert_eq!(provider, Provider::GoogleGenAI);
    assert_eq!(model, "gemini-pro");

    let (provider, model) = Provider::from_model_string("openrouter:anthropic/claude-3").unwrap();
    assert_eq!(provider, Provider::OpenRouter);
    assert_eq!(model, "anthropic/claude-3");
}

#[test]
fn test_provider_supports_native_tools() {
    assert!(Provider::Anthropic.supports_native_tools());
    assert!(Provider::OpenAI.supports_native_tools());
    assert!(Provider::OpenAIResponses.supports_native_tools());
    assert!(Provider::GoogleGenAI.supports_native_tools());
    assert!(Provider::GoogleAnthropic.supports_native_tools());
    assert!(Provider::OpenRouter.supports_native_tools());
    assert!(Provider::OpenRouterResponses.supports_native_tools());
    assert!(Provider::Bedrock.supports_native_tools());
}

// ============================================================================
// Google Anthropic (Vertex AI with Claude) Tests
// ============================================================================

#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires GOOGLE_APPLICATION_CREDENTIALS"]
async fn test_google_anthropic_simple() {
    use reson_agentic::providers::GoogleAnthropicClient;

    let client = GoogleAnthropicClient::from_adc("claude-3-5-sonnet-v2@20241022", "us-east5");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What is 2 + 2? Reply with just the number.",
    ))];

    let config = GenerationConfig::new("claude-3-5-sonnet-v2@20241022").with_max_tokens(100);

    let response = client.get_generation(&messages, &config).await.unwrap();
    println!("Google Anthropic response: {}", response.content);
    assert!(response.content.contains("4"));
}

#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires GOOGLE_APPLICATION_CREDENTIALS"]
async fn test_google_anthropic_with_tools() {
    use reson_agentic::providers::GoogleAnthropicClient;

    let client = GoogleAnthropicClient::from_adc("claude-3-5-sonnet-v2@20241022", "us-east5");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the add_numbers tool to calculate 15 + 27.",
    ))];

    let config = GenerationConfig::new("claude-3-5-sonnet-v2@20241022")
        .with_max_tokens(1024)
        .with_tools(vec![anthropic_add_tool()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);
    assert!(!response.tool_calls.is_empty(), "Expected tool call");

    let tool_call = &response.tool_calls[0];
    assert_eq!(tool_call["name"], "add_numbers");
}

#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires GOOGLE_APPLICATION_CREDENTIALS"]
async fn test_google_anthropic_streaming() {
    use futures::StreamExt;
    use reson_agentic::providers::GoogleAnthropicClient;

    let client = GoogleAnthropicClient::from_adc("claude-3-5-sonnet-v2@20241022", "us-east5");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Count from 1 to 5, one number per line.",
    ))];

    let config = GenerationConfig::new("claude-3-5-sonnet-v2@20241022").with_max_tokens(200);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut full_content = String::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                reson_agentic::providers::StreamChunk::Content(text) => {
                    print!("{}", text);
                    full_content.push_str(&text);
                }
                reson_agentic::providers::StreamChunk::Usage {
                    input_tokens,
                    output_tokens,
                    ..
                } => {
                    println!("\nUsage: {} in, {} out", input_tokens, output_tokens);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!();
    assert!(full_content.contains("1"));
    assert!(full_content.contains("5"));
}

#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires GOOGLE_APPLICATION_CREDENTIALS"]
async fn test_google_anthropic_streaming_with_tools() {
    use futures::StreamExt;
    use reson_agentic::providers::GoogleAnthropicClient;

    let client = GoogleAnthropicClient::from_adc("claude-3-5-sonnet-v2@20241022", "us-east5");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the add_numbers tool to calculate 100 + 200.",
    ))];

    let config = GenerationConfig::new("claude-3-5-sonnet-v2@20241022")
        .with_max_tokens(1024)
        .with_tools(vec![anthropic_add_tool()])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut content = String::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                reson_agentic::providers::StreamChunk::Content(text) => {
                    print!("{}", text);
                    content.push_str(&text);
                }
                reson_agentic::providers::StreamChunk::ToolCallComplete(tc) => {
                    println!("Tool call complete: {:?}", tc);
                    tool_calls.push(tc);
                }
                reson_agentic::providers::StreamChunk::Usage {
                    input_tokens,
                    output_tokens,
                    ..
                } => {
                    println!("Usage: {} in, {} out", input_tokens, output_tokens);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\nTool calls: {:?}", tool_calls);
    assert!(!tool_calls.is_empty(), "Expected tool call via streaming");

    // Streaming tool calls have format: {"function": {"name": ..., "arguments": ...}, "id": ...}
    let tool_call = &tool_calls[0];
    let name = tool_call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .or_else(|| tool_call.get("name").and_then(|n| n.as_str()));
    assert_eq!(name, Some("add_numbers"));
}
