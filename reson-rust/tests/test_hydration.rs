//! ToolCall/ToolResult Hydration integration tests
//!
//! Tests for ToolCall and ToolResult hydration and provider conversion.
//! Mirrors Python tests from:
//! - integration_tests/test_toolcall_hydration.py

use reson_agentic::prelude::*;
use reson_agentic::utils::ConversationMessage;
use std::env;

// ============================================================================
// ToolCall Creation Tests
// ============================================================================

#[test]
fn test_toolcall_create_basic() {
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
    assert_eq!(tool_call.args["units"], "celsius");
}

#[test]
fn test_toolcall_from_openai_format() {
    // OpenAI format: {"id": "call_abc", "function": {"name": "...", "arguments": "{...}"}}
    let openai_format = serde_json::json!({
        "id": "call_abc123",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"San Francisco\", \"units\": \"celsius\"}"
        }
    });

    let tool_call = ToolCall::from_provider_format(openai_format, Provider::OpenAI).unwrap();

    assert_eq!(tool_call.tool_use_id, "call_abc123");
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "San Francisco");
    assert_eq!(tool_call.args["units"], "celsius");
    // raw_arguments should be preserved for OpenAI
    assert!(tool_call.raw_arguments.is_some());
}

#[test]
fn test_toolcall_from_openai_format_list() {
    // Multiple tool calls - would be handled individually in practice
    let openai_tool_calls = vec![
        serde_json::json!({
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": "{\"location\": \"SF\"}"
            }
        }),
        serde_json::json!({
            "id": "call_2",
            "function": {
                "name": "add_numbers",
                "arguments": "{\"a\": 5, \"b\": 3}"
            }
        }),
    ];

    let tool_calls: Vec<ToolCall> = openai_tool_calls
        .into_iter()
        .map(|tc| ToolCall::from_provider_format(tc, Provider::OpenAI).unwrap())
        .collect();

    assert_eq!(tool_calls.len(), 2);

    assert_eq!(tool_calls[0].tool_use_id, "call_1");
    assert_eq!(tool_calls[0].tool_name, "get_weather");
    assert_eq!(tool_calls[0].args["location"], "SF");

    assert_eq!(tool_calls[1].tool_use_id, "call_2");
    assert_eq!(tool_calls[1].tool_name, "add_numbers");
    assert_eq!(tool_calls[1].args["a"], 5);
    assert_eq!(tool_calls[1].args["b"], 3);
}

#[test]
fn test_toolcall_from_anthropic_format() {
    // Anthropic format: {"id": "toolu_01", "name": "...", "input": {...}}
    let anthropic_format = serde_json::json!({
        "id": "toolu_abc123",
        "name": "get_weather",
        "input": {
            "location": "San Francisco",
            "units": "celsius"
        }
    });

    let tool_call = ToolCall::from_provider_format(anthropic_format, Provider::Anthropic).unwrap();

    assert_eq!(tool_call.tool_use_id, "toolu_abc123");
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "San Francisco");
    assert_eq!(tool_call.args["units"], "celsius");
    // Anthropic doesn't use raw JSON strings
    assert!(tool_call.raw_arguments.is_none());
}

#[test]
fn test_toolcall_from_google_format() {
    // Google format: {"functionCall": {"name": "...", "args": {...}}}
    let google_format = serde_json::json!({
        "functionCall": {
            "name": "get_weather",
            "args": {
                "location": "San Francisco",
                "units": "celsius"
            }
        }
    });

    let tool_call = ToolCall::from_provider_format(google_format, Provider::GoogleGenAI).unwrap();

    // Google doesn't provide ID, one is generated
    assert!(!tool_call.tool_use_id.is_empty());
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(tool_call.args["location"], "San Francisco");
    assert_eq!(tool_call.args["units"], "celsius");
}

// ============================================================================
// ToolCall Provider Conversion Tests
// ============================================================================

#[test]
fn test_toolcall_to_openai_format() {
    let tool_call = ToolCall::new(
        "get_weather",
        serde_json::json!({"location": "SF", "units": "celsius"}),
    );

    let message = tool_call.to_provider_assistant_message(Provider::OpenAI);

    assert_eq!(message["role"], "assistant");
    assert!(message["tool_calls"].is_array());

    let tool_calls = message["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls.len(), 1);

    let tc = &tool_calls[0];
    assert_eq!(tc["type"], "function");
    assert_eq!(tc["function"]["name"], "get_weather");

    // Arguments should be JSON string
    let args_str = tc["function"]["arguments"].as_str().unwrap();
    let args: serde_json::Value = serde_json::from_str(args_str).unwrap();
    assert_eq!(args["location"], "SF");
    assert_eq!(args["units"], "celsius");
}

#[test]
fn test_toolcall_to_anthropic_format() {
    let anthropic_format = serde_json::json!({
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"location": "SF", "units": "celsius"}
    });

    let tool_call = ToolCall::from_provider_format(anthropic_format, Provider::Anthropic).unwrap();
    let message = tool_call.to_provider_assistant_message(Provider::Anthropic);

    assert_eq!(message["role"], "assistant");
    assert!(message["content"].is_array());

    let content = message["content"].as_array().unwrap();
    assert_eq!(content.len(), 1);

    let tc = &content[0];
    assert_eq!(tc["type"], "tool_use");
    assert_eq!(tc["id"], "toolu_123");
    assert_eq!(tc["name"], "get_weather");
    assert_eq!(tc["input"]["location"], "SF");
    assert_eq!(tc["input"]["units"], "celsius");
}

#[test]
fn test_toolcall_to_google_format() {
    let tool_call = ToolCall::new(
        "get_weather",
        serde_json::json!({"location": "SF", "units": "celsius"}),
    );

    let message = tool_call.to_provider_assistant_message(Provider::GoogleGenAI);

    assert_eq!(message["role"], "model");
    assert!(message["parts"].is_array());

    let parts = message["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 1);

    let fc = &parts[0]["functionCall"];
    assert_eq!(fc["name"], "get_weather");
    assert_eq!(fc["args"]["location"], "SF");
    assert_eq!(fc["args"]["units"], "celsius");
}

#[test]
fn test_toolcall_to_openrouter_format() {
    let tool_call = ToolCall::new("get_weather", serde_json::json!({"location": "SF"}));

    let message = tool_call.to_provider_assistant_message(Provider::OpenRouter);

    // OpenRouter uses OpenAI format
    assert_eq!(message["role"], "assistant");
    assert!(message["tool_calls"].is_array());
    assert_eq!(message["tool_calls"][0]["type"], "function");
}

// ============================================================================
// Roundtrip Tests
// ============================================================================

#[test]
fn test_openai_roundtrip() {
    let original = serde_json::json!({
        "id": "call_test123",
        "function": {
            "name": "calculate",
            "arguments": "{\"a\": 2, \"b\": 2}"
        }
    });

    // Parse from provider format
    let tool_call = ToolCall::from_provider_format(original.clone(), Provider::OpenAI).unwrap();

    // Convert back
    let converted = tool_call.to_provider_assistant_message(Provider::OpenAI);

    assert_eq!(converted["tool_calls"][0]["id"], "call_test123");
    assert_eq!(converted["tool_calls"][0]["function"]["name"], "calculate");

    // Parse arguments and verify
    let converted_args_str = converted["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    let converted_args: serde_json::Value = serde_json::from_str(converted_args_str).unwrap();
    assert_eq!(converted_args["a"], 2);
    assert_eq!(converted_args["b"], 2);
}

#[test]
fn test_anthropic_roundtrip() {
    let original = serde_json::json!({
        "id": "toolu_test123",
        "name": "calculate",
        "input": {"a": 2, "b": 2}
    });

    // Parse from provider format
    let tool_call = ToolCall::from_provider_format(original.clone(), Provider::Anthropic).unwrap();

    // Convert back
    let converted = tool_call.to_provider_assistant_message(Provider::Anthropic);

    assert_eq!(converted["content"][0]["id"], "toolu_test123");
    assert_eq!(converted["content"][0]["name"], "calculate");
    assert_eq!(converted["content"][0]["input"]["a"], 2);
    assert_eq!(converted["content"][0]["input"]["b"], 2);
}

// ============================================================================
// ToolResult Tests
// ============================================================================

#[test]
fn test_toolresult_success() {
    let result = ToolResult::success("call_123".to_string(), "The weather is sunny".to_string());

    assert_eq!(result.tool_use_id, "call_123");
    assert_eq!(result.content, "The weather is sunny");
    assert!(!result.is_error);
}

#[test]
fn test_toolresult_error() {
    let result = ToolResult::error("call_123".to_string(), "Tool execution failed".to_string());

    assert_eq!(result.tool_use_id, "call_123");
    assert_eq!(result.content, "Tool execution failed");
    assert!(result.is_error);
}

#[test]
fn test_toolresult_with_name() {
    let result = ToolResult::success_with_name(
        "call_123".to_string(),
        "get_weather".to_string(),
        "Sunny".to_string(),
    );

    assert_eq!(result.tool_use_id, "call_123");
    assert_eq!(result.tool_name, Some("get_weather".to_string()));
    assert_eq!(result.content, "Sunny");
}

#[test]
fn test_toolresult_to_openai_format() {
    let result = ToolResult::success("call_123".to_string(), "The weather is sunny".to_string());

    let format = result.to_provider_format(Provider::OpenAI);

    assert_eq!(format["role"], "tool");
    assert_eq!(format["tool_call_id"], "call_123");
    assert_eq!(format["content"], "The weather is sunny");
}

#[test]
fn test_toolresult_to_anthropic_format() {
    let result = ToolResult::success("toolu_123".to_string(), "The weather is sunny".to_string());

    let format = result.to_provider_format(Provider::Anthropic);

    // to_provider_format returns the tool_result content block, not a full message
    // The full message wrapping with role="user" is done at message coalescing time
    assert_eq!(format["type"], "tool_result");
    assert_eq!(format["tool_use_id"], "toolu_123");
    assert_eq!(format["content"], "The weather is sunny");
}

#[test]
fn test_toolresult_to_google_format() {
    let result = ToolResult::success_with_name(
        "google_123".to_string(),
        "get_weather".to_string(),
        "The weather is sunny".to_string(),
    );

    let format = result.to_provider_format(Provider::GoogleGenAI);

    // to_provider_format returns the functionResponse content block, not a full message
    // The full message wrapping with role="user" and "parts" is done at message coalescing time
    let fr = &format["functionResponse"];

    assert_eq!(fr["name"], "get_weather");
    // Response content can be in different formats
    let response = &fr["response"];
    assert!(
        response.get("content").is_some() || response.get("result").is_some(),
        "Google format should have response content"
    );
}

// ============================================================================
// Conversation Message Tests
// ============================================================================

#[test]
fn test_toolcall_in_conversation_message() {
    let tool_call = ToolCall::new("get_weather", serde_json::json!({"location": "SF"}));

    let message: ConversationMessage = tool_call.clone().into();

    match message {
        ConversationMessage::ToolCall(tc) => {
            assert_eq!(tc.tool_name, "get_weather");
            assert_eq!(tc.args["location"], "SF");
        }
        _ => panic!("Expected ToolCall variant"),
    }
}

#[test]
fn test_toolresult_in_conversation_message() {
    let result = ToolResult::success("call_123".to_string(), "Sunny".to_string());

    let message: ConversationMessage = result.clone().into();

    match message {
        ConversationMessage::ToolResult(tr) => {
            assert_eq!(tr.tool_use_id, "call_123");
            assert_eq!(tr.content, "Sunny");
        }
        _ => panic!("Expected ToolResult variant"),
    }
}

#[test]
fn test_conversation_with_tool_calls() {
    // Build a conversation with tool calls and results
    let tool_call = ToolCall::new(
        "get_weather",
        serde_json::json!({"location": "San Francisco"}),
    );

    let conversation: Vec<ConversationMessage> = vec![
        ConversationMessage::Chat(ChatMessage::user("What's the weather in SF?")),
        ConversationMessage::ToolCall(tool_call.clone()),
        ConversationMessage::ToolResult(ToolResult::success(
            tool_call.tool_use_id.clone(),
            "72°F sunny".to_string(),
        )),
        ConversationMessage::Chat(ChatMessage::user("Thanks!")),
    ];

    assert_eq!(conversation.len(), 4);

    // Verify types
    assert!(matches!(conversation[0], ConversationMessage::Chat(_)));
    assert!(matches!(conversation[1], ConversationMessage::ToolCall(_)));
    assert!(matches!(
        conversation[2],
        ConversationMessage::ToolResult(_)
    ));
    assert!(matches!(conversation[3], ConversationMessage::Chat(_)));
}

// ============================================================================
// Integration Tests (require API keys)
// ============================================================================

fn get_openrouter_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY").ok()
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_toolcall_in_conversation_history() {
    use reson_agentic::providers::{GenerationConfig, InferenceClient, OpenRouterClient};

    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");

    let weather_tool = serde_json::json!({
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    });

    // Simulate a previous conversation with tool call
    // Create tool call with specific ID for testing
    let previous_anthropic_format = serde_json::json!({
        "id": "toolu_previous_123",
        "name": "get_weather",
        "input": {"location": "New York"}
    });
    let previous_tool_call =
        ToolCall::from_provider_format(previous_anthropic_format, Provider::Anthropic).unwrap();

    let previous_tool_result = ToolResult::success_with_name(
        "toolu_previous_123".to_string(),
        "get_weather".to_string(),
        "Weather in New York: 18°C, rainy".to_string(),
    );

    // Build conversation history
    let conversation: Vec<ConversationMessage> = vec![
        ConversationMessage::Chat(ChatMessage::user("What's the weather in New York?")),
        ConversationMessage::ToolCall(previous_tool_call),
        ConversationMessage::ToolResult(previous_tool_result),
        ConversationMessage::Chat(ChatMessage::user("Thanks! Now what about San Francisco?")),
    ];

    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![weather_tool])
        .with_native_tools(true);

    let result = client.get_generation(&conversation, &config).await;

    match result {
        Ok(response) => {
            println!("Response with history: {}", response.content);
            println!("Tool calls: {:?}", response.tool_calls);

            // Should either get a tool call for SF or a response
            assert!(
                !response.content.is_empty() || !response.tool_calls.is_empty(),
                "Should get response when using tool call history"
            );
        }
        Err(e) => {
            panic!("Test failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_toolcall_hydration_workflow() {
    use reson_agentic::providers::{GenerationConfig, InferenceClient, OpenRouterClient};

    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");

    let add_tool = serde_json::json!({
        "name": "add_numbers",
        "description": "Add two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }
    });

    // Simulate hydrating a tool call from OpenAI format (e.g., from storage)
    let openai_tool_call_data = serde_json::json!({
        "id": "call_previous_calculation",
        "function": {
            "name": "add_numbers",
            "arguments": "{\"a\": 10, \"b\": 5}"
        }
    });

    // Create ToolCall from provider format (hydration)
    let hydrated_tool_call =
        ToolCall::from_provider_format(openai_tool_call_data, Provider::OpenAI).unwrap();

    // Create corresponding result
    let tool_result = ToolResult::success_with_name(
        "call_previous_calculation".to_string(),
        "add_numbers".to_string(),
        "15".to_string(),
    );

    // Use in conversation
    let conversation: Vec<ConversationMessage> = vec![
        ConversationMessage::Chat(ChatMessage::user("Please add 10 and 5")),
        ConversationMessage::ToolCall(hydrated_tool_call),
        ConversationMessage::ToolResult(tool_result),
        ConversationMessage::Chat(ChatMessage::user("Great! Now multiply that result by 2")),
    ];

    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![add_tool])
        .with_native_tools(true);

    let result = client.get_generation(&conversation, &config).await;

    match result {
        Ok(response) => {
            println!("Hydration workflow response: {}", response.content);

            // Should continue the conversation
            assert!(
                !response.content.is_empty() || !response.tool_calls.is_empty(),
                "Should continue conversation with hydrated tool call"
            );
        }
        Err(e) => {
            panic!("Hydration workflow test failed: {}", e);
        }
    }
}
