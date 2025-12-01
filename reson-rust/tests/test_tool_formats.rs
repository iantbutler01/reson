//! Tool Call Format Validation integration tests
//!
//! Tests for tool call format validation across all providers.
//! Mirrors Python tests from:
//! - integration_tests/test_tool_call_format_validation.py

use futures::StreamExt;
use reson::prelude::*;
use reson::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient, OpenRouterClient,
    StreamChunk,
};
use reson::utils::ConversationMessage;
use std::env;

// ============================================================================
// Helper Functions
// ============================================================================

fn get_anthropic_key() -> Option<String> {
    env::var("ANTHROPIC_API_KEY").ok()
}

fn get_google_key() -> Option<String> {
    env::var("GOOGLE_API_KEY").ok()
}

fn get_openrouter_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY").ok()
}

fn search_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "search_function",
        "description": "Search for something",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["text"]
        }
    })
}

fn calculate_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "calculate_function",
        "description": "Perform a calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
                "operation": {"type": "string", "enum": ["add", "multiply"], "default": "add"}
            },
            "required": ["a", "b"]
        }
    })
}

fn untyped_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "untyped_function",
        "description": "A function without tool_type registration",
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"}
            },
            "required": ["message"]
        }
    })
}

// ============================================================================
// Tool Call Format Tests (JSON parsing)
// ============================================================================

#[test]
fn test_tool_call_openai_format() {
    // OpenAI format: {"id": "call_abc", "function": {"name": "...", "arguments": "{...}"}}
    let openai_tool_call = serde_json::json!({
        "id": "call_abc123",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\", \"units\": \"celsius\"}"
        }
    });

    // Extract fields
    let id = openai_tool_call["id"].as_str().unwrap();
    let function_name = openai_tool_call["function"]["name"].as_str().unwrap();
    let arguments_str = openai_tool_call["function"]["arguments"].as_str().unwrap();

    assert_eq!(id, "call_abc123");
    assert_eq!(function_name, "get_weather");

    // Parse arguments JSON string
    let args: serde_json::Value = serde_json::from_str(arguments_str).unwrap();
    assert_eq!(args["location"], "San Francisco");
    assert_eq!(args["units"], "celsius");
}

#[test]
fn test_tool_call_anthropic_format() {
    // Anthropic format: {"id": "toolu_01", "name": "...", "input": {...}}
    let anthropic_tool_call = serde_json::json!({
        "id": "toolu_abc123",
        "name": "get_weather",
        "input": {
            "location": "San Francisco",
            "units": "celsius"
        }
    });

    let id = anthropic_tool_call["id"].as_str().unwrap();
    let name = anthropic_tool_call["name"].as_str().unwrap();
    let input = &anthropic_tool_call["input"];

    assert_eq!(id, "toolu_abc123");
    assert_eq!(name, "get_weather");
    assert_eq!(input["location"], "San Francisco");
    assert_eq!(input["units"], "celsius");
}

#[test]
fn test_tool_call_google_format() {
    // Google format: {"functionCall": {"name": "...", "args": {...}}}
    let google_tool_call = serde_json::json!({
        "functionCall": {
            "name": "get_weather",
            "args": {
                "location": "San Francisco",
                "units": "celsius"
            }
        }
    });

    let function_call = &google_tool_call["functionCall"];
    let name = function_call["name"].as_str().unwrap();
    let args = &function_call["args"];

    assert_eq!(name, "get_weather");
    assert_eq!(args["location"], "San Francisco");
    assert_eq!(args["units"], "celsius");
}

// ============================================================================
// ToolCall Type Tests
// ============================================================================

#[test]
fn test_toolcall_creation() {
    let tool_call = ToolCall::new(
        "get_weather",
        serde_json::json!({
            "location": "San Francisco",
            "units": "celsius"
        }),
    );

    // tool_use_id is auto-generated
    assert!(!tool_call.tool_use_id.is_empty());
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "San Francisco");
}

#[test]
fn test_toolcall_from_openai_format() {
    let openai_format = serde_json::json!({
        "id": "call_123",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"SF\"}"
        }
    });

    let tool_call = ToolCall::from_provider_format(openai_format, Provider::OpenAI).unwrap();

    assert_eq!(tool_call.tool_use_id, "call_123");
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "SF");
}

#[test]
fn test_toolcall_from_anthropic_format() {
    let anthropic_format = serde_json::json!({
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"location": "SF"}
    });

    let tool_call = ToolCall::from_provider_format(anthropic_format, Provider::Anthropic).unwrap();

    assert_eq!(tool_call.tool_use_id, "toolu_123");
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "SF");
}

#[test]
fn test_toolcall_from_google_format() {
    let google_format = serde_json::json!({
        "functionCall": {
            "name": "get_weather",
            "args": {"location": "SF"}
        }
    });

    let tool_call = ToolCall::from_provider_format(google_format, Provider::GoogleGenAI).unwrap();

    // Google doesn't provide ID, one is generated
    assert!(!tool_call.tool_use_id.is_empty());
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "SF");
}

#[test]
fn test_toolcall_to_provider_format_openai() {
    let tool_call = ToolCall::new(
        "get_weather",
        serde_json::json!({"location": "SF", "units": "celsius"}),
    );

    let format = tool_call.to_provider_assistant_message(Provider::OpenAI);

    assert_eq!(format["role"], "assistant");
    assert!(format["tool_calls"].is_array());

    let tool_calls = format["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);

    let tc = &tool_calls[0];
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "get_weather");
    // Arguments should be JSON string
    assert!(tc["function"]["arguments"].is_string());
}

#[test]
fn test_toolcall_to_provider_format_anthropic() {
    let anthropic_format = serde_json::json!({
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"location": "SF"}
    });

    let tool_call = ToolCall::from_provider_format(anthropic_format, Provider::Anthropic).unwrap();
    let format = tool_call.to_provider_assistant_message(Provider::Anthropic);

    assert_eq!(format["role"], "assistant");
    assert!(format["content"].is_array());

    let content = format["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);

    let tc = &content[0];
    assert_eq!(tc["type"], "tool_use");
    assert_eq!(tc["id"], "toolu_123");
    assert_eq!(tc["name"], "get_weather");
    assert_eq!(tc["input"]["location"], "SF");
}

#[test]
fn test_toolcall_to_provider_format_google() {
    let tool_call = ToolCall::new(
        "get_weather",
        serde_json::json!({"location": "SF"}),
    );

    let format = tool_call.to_provider_assistant_message(Provider::GoogleGenAI);

    assert_eq!(format["role"], "model");
    assert!(format["parts"].is_array());

    let parts = format["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 1);

    let fc = &parts[0]["functionCall"];
    assert_eq!(fc["name"], "get_weather");
    assert_eq!(fc["args"]["location"], "SF");
}

// ============================================================================
// Cross-Provider Format Consistency Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_cross_provider_tool_call_format() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 5 + 3 using calculate_function",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![calculate_tool_schema()])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::ToolCallComplete(tool_call) => {
                    println!("Tool call: {:?}", tool_call);

                    // Validate format consistency
                    // Should have either "name" (Anthropic) or "function.name" (OpenAI)
                    let name = tool_call
                        .get("name")
                        .or_else(|| tool_call.get("function").and_then(|f| f.get("name")))
                        .or_else(|| tool_call.get("_tool_name"))
                        .and_then(|v| v.as_str());

                    assert!(name.is_some(), "Tool call should have a name field");
                    println!("Tool name: {}", name.unwrap());

                    tool_calls.push(tool_call);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    assert!(!tool_calls.is_empty(), "Should receive tool calls");
}

#[tokio::test]
#[ignore = "Requires GOOGLE_API_KEY"]
async fn test_google_tool_call_format() {
    let api_key = get_google_key().expect("GOOGLE_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 10 + 20 using the calculate_function tool",
    ))];

    let config = GenerationConfig::new("gemini-1.5-flash")
        .with_max_tokens(1024)
        .with_tools(vec![calculate_tool_schema()])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::ToolCallComplete(tool_call) => {
                    println!("Google tool call: {:?}", tool_call);

                    // Google format should have functionCall or be normalized
                    let name = tool_call
                        .get("name")
                        .or_else(|| tool_call.get("_tool_name"))
                        .or_else(|| {
                            tool_call
                                .get("functionCall")
                                .and_then(|fc| fc.get("name"))
                        })
                        .and_then(|v| v.as_str());

                    assert!(name.is_some(), "Google tool call should have name");
                    tool_calls.push(tool_call);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    assert!(!tool_calls.is_empty(), "Should receive Google tool calls");
}

// ============================================================================
// Mixed Tool Registration Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_mixed_tool_registration_formats() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 8 + 12 using calculate_function, then process the result message \
         'calculation done' using untyped_function",
    ))];

    // Mix of tools
    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![calculate_tool_schema(), untyped_tool_schema()])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut tool_names: Vec<String> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::ToolCallComplete(tool_call) => {
                    let name = tool_call
                        .get("name")
                        .or_else(|| tool_call.get("_tool_name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    println!("Mixed tool: {}", name);
                    tool_names.push(name);
                    tool_calls.push(tool_call);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("Tools received: {:?}", tool_names);
    assert!(!tool_calls.is_empty(), "Should receive tool calls");
}

// ============================================================================
// Tool Call Edge Cases Tests
// ============================================================================

#[test]
fn test_tool_call_with_empty_input() {
    let tool_call = ToolCall::new(
        "get_time",
        serde_json::json!({}),
    );

    assert_eq!(tool_call.args, serde_json::json!({}));

    // Should still convert to provider formats correctly
    let openai_format = tool_call.to_provider_assistant_message(Provider::OpenAI);
    let anthropic_format = tool_call.to_provider_assistant_message(Provider::Anthropic);

    assert!(openai_format["tool_calls"][0]["function"]["arguments"].is_string());
    assert_eq!(anthropic_format["content"][0]["input"], serde_json::json!({}));
}

#[test]
fn test_tool_call_with_nested_input() {
    let tool_call = ToolCall::new(
        "complex_function",
        serde_json::json!({
            "config": {
                "nested": {
                    "deeply": {
                        "value": 42
                    }
                }
            },
            "items": [1, 2, 3]
        }),
    );

    assert_eq!(tool_call.args["config"]["nested"]["deeply"]["value"], 42);

    // Verify serialization preserves structure
    let format = tool_call.to_provider_assistant_message(Provider::Anthropic);
    assert_eq!(
        format["content"][0]["input"]["config"]["nested"]["deeply"]["value"],
        42
    );
}

#[test]
fn test_tool_call_with_special_characters() {
    let tool_call = ToolCall::new(
        "search",
        serde_json::json!({
            "query": "hello \"world\" with 'quotes' and \\ backslash",
            "path": "/path/to/file.txt"
        }),
    );

    // OpenAI format requires JSON string serialization
    let openai_format = tool_call.to_provider_assistant_message(Provider::OpenAI);
    let args_str = openai_format["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();

    // Parse back to verify
    let parsed: serde_json::Value = serde_json::from_str(args_str).unwrap();
    assert_eq!(
        parsed["query"],
        "hello \"world\" with 'quotes' and \\ backslash"
    );
}

// ============================================================================
// Multi-Turn ToolResult Conversation Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_multi_turn_toolresult_conversation() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");

    // Turn 1: Initial request
    let client1 = OpenRouterClient::new(api_key.clone(), "anthropic/claude-3-5-sonnet", None, None);

    let messages1 = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 25 + 17 using calculate_function",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![calculate_tool_schema()])
        .with_native_tools(true);

    let result1 = client1.get_generation(&messages1, &config).await.unwrap();

    // Process tool call
    let mut conversation: Vec<ConversationMessage> = messages1;

    if !result1.tool_calls.is_empty() {
        for tool_call_json in &result1.tool_calls {
            let tool_use_id = tool_call_json
                .get("id")
                .or_else(|| tool_call_json.get("tool_use_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let tool_name = tool_call_json
                .get("name")
                .or_else(|| tool_call_json.get("_tool_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            println!("Turn 1 tool call: {} ({})", tool_name, tool_use_id);

            // Add tool call to conversation
            let tool_call = ToolCall::new(tool_name.clone(), tool_call_json.clone());
            conversation.push(ConversationMessage::ToolCall(tool_call));

            // Execute and add result
            let result_content = "42"; // 25 + 17 = 42
            let tool_result = ToolResult::success_with_name(
                tool_use_id,
                tool_name,
                result_content.to_string(),
            );
            conversation.push(ConversationMessage::ToolResult(tool_result));
        }
    }

    // Turn 2: Follow-up
    conversation.push(ConversationMessage::Chat(ChatMessage::user(
        "Great! Now double that result.",
    )));

    let client2 = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);
    let result2 = client2.get_generation(&conversation, &config).await.unwrap();

    println!("Turn 2 response: {}", result2.content);

    // Should either get a tool call (to calculate 42 * 2) or a text response
    assert!(
        !result2.content.is_empty() || !result2.tool_calls.is_empty(),
        "Should get response in turn 2"
    );
}

// ============================================================================
// Streaming Partial Tool Call Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_streaming_tool_call_deltas() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Search for 'artificial intelligence' with max_results 8",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![search_tool_schema()])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut partial_count = 0;
    let mut complete_count = 0;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::ToolCallPartial(partial) => {
                    partial_count += 1;
                    println!("Partial #{}: {:?}", partial_count, partial);
                }
                StreamChunk::ToolCallComplete(tool_call) => {
                    complete_count += 1;
                    println!("Complete: {:?}", tool_call);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("Partials: {}, Complete: {}", partial_count, complete_count);

    // Should have at least one complete tool call
    assert!(
        complete_count > 0,
        "Should receive complete tool call from stream"
    );
}
