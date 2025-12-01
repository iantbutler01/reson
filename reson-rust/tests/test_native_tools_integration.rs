//! Native tool calling integration tests
//!
//! Tests for native tool calling functionality across all providers.
//! Mirrors Python tests from:
//! - integration_tests/test_native_tools.py
//! - integration_tests/test_native_tools_real_apis.py
//! - integration_tests/test_comprehensive_native_tools.py

use reson::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient, OAIClient,
    OpenRouterClient,
};
use reson::types::{ChatMessage, Provider, ToolCall, ToolResult};
use reson::utils::ConversationMessage;
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
    env::var("GOOGLE_API_KEY").ok()
}

fn get_openrouter_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY").ok()
}

/// Calculator tool schema (Anthropic format)
fn calculator_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "calculate",
        "description": "Perform a calculation operation",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The operation to perform"
                },
                "a": {"type": "number", "description": "First operand"},
                "b": {"type": "number", "description": "Second operand"}
            },
            "required": ["operation", "a", "b"]
        }
    })
}

/// Search tool schema
fn search_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "search_database",
        "description": "Search a database with query parameters",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Search query text"},
                "category": {"type": "string", "default": "general"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["text"]
        }
    })
}

/// Weather tool schema
fn weather_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["location"]
        }
    })
}

/// Add numbers tool schema
fn add_numbers_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "add_numbers",
        "description": "Add two numbers together",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    })
}

/// Multiply numbers tool schema
fn multiply_numbers_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "multiply_numbers",
        "description": "Multiply two numbers together",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    })
}

/// Execute a mock tool and return result
fn execute_mock_tool(tool_name: &str, input: &serde_json::Value) -> String {
    match tool_name {
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
        "calculate" => {
            let op = input.get("operation").and_then(|v| v.as_str()).unwrap_or("add");
            let a = input.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = input.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            match op {
                "add" => (a + b).to_string(),
                "subtract" => (a - b).to_string(),
                "multiply" => (a * b).to_string(),
                "divide" => {
                    if b != 0.0 {
                        (a / b).to_string()
                    } else {
                        "infinity".to_string()
                    }
                }
                _ => "unknown operation".to_string(),
            }
        }
        "search_database" => {
            let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let max = input.get("max_results").and_then(|v| v.as_i64()).unwrap_or(5);
            let category = input
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("general");
            format!(
                "Found {} results for '{}' in category '{}'",
                max, text, category
            )
        }
        "get_weather" => {
            let location = input.get("location").and_then(|v| v.as_str()).unwrap_or("");
            let units = input
                .get("units")
                .and_then(|v| v.as_str())
                .unwrap_or("celsius");
            format!(
                "Weather in {}: 22°{}, partly cloudy",
                location,
                units.chars().next().unwrap_or('C').to_uppercase()
            )
        }
        _ => format!("Unknown tool: {}", tool_name),
    }
}

/// Extract tool name from tool call response
fn get_tool_name(tool_call: &serde_json::Value) -> Option<String> {
    tool_call
        .get("name")
        .or_else(|| tool_call.get("_tool_name"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract tool ID from tool call response
fn get_tool_id(tool_call: &serde_json::Value) -> Option<String> {
    tool_call
        .get("id")
        .or_else(|| tool_call.get("_tool_use_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Extract tool input from tool call response
fn get_tool_input(tool_call: &serde_json::Value) -> serde_json::Value {
    tool_call
        .get("input")
        .or_else(|| tool_call.get("args"))
        .cloned()
        .unwrap_or(serde_json::json!({}))
}

// ============================================================================
// Provider Support Tests
// ============================================================================

#[test]
fn test_provider_supports_native_tools() {
    // Test that all expected providers support native tools
    assert!(Provider::Anthropic.supports_native_tools());
    assert!(Provider::OpenAI.supports_native_tools());
    assert!(Provider::GoogleGenAI.supports_native_tools());
    assert!(Provider::OpenRouter.supports_native_tools());
    assert!(Provider::Bedrock.supports_native_tools());
}

#[test]
fn test_provider_prefix_parsing() {
    // Test that provider prefixes are parsed correctly
    let (provider, model) = Provider::from_model_string("openai:gpt-4").unwrap();
    assert_eq!(provider, Provider::OpenAI);
    assert_eq!(model, "gpt-4");

    let (provider, model) = Provider::from_model_string("anthropic:claude-3").unwrap();
    assert_eq!(provider, Provider::Anthropic);
    assert_eq!(model, "claude-3");

    let (provider, model) = Provider::from_model_string("google-genai:gemini-pro").unwrap();
    assert_eq!(provider, Provider::GoogleGenAI);
    assert_eq!(model, "gemini-pro");
}

// ============================================================================
// OpenRouter Native Tools Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_native_tools_single_call() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the search_database tool to search for 'python tutorials' and find 3 results",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![search_tool_schema(), add_numbers_schema()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);

    // Should have tool calls
    assert!(
        !response.tool_calls.is_empty(),
        "Expected tool call in response"
    );

    let tool_call = &response.tool_calls[0];
    let tool_name = get_tool_name(tool_call);

    println!("Tool call detected: {:?}", tool_name);
    assert_eq!(tool_name, Some("search_database".to_string()));

    // Verify tool ID is preserved
    let tool_id = get_tool_id(tool_call);
    assert!(tool_id.is_some(), "Tool ID should be preserved");
    println!("Tool ID: {:?}", tool_id);

    // Execute mock tool
    let input = get_tool_input(tool_call);
    let result = execute_mock_tool("search_database", &input);
    println!("Tool result: {}", result);
    assert!(result.contains("python tutorials"));
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openrouter_multi_turn_tool_conversation() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![add_numbers_schema()])
        .with_native_tools(true);

    // Turn 1: Ask to add numbers
    let mut history: Vec<ConversationMessage> = vec![ConversationMessage::Chat(ChatMessage::user(
        "Please add 15 and 27 using the add_numbers tool",
    ))];

    let response1 = client.get_generation(&history, &config).await.unwrap();
    println!("Turn 1: {:?}", response1);

    assert!(!response1.tool_calls.is_empty());
    let tool_call = &response1.tool_calls[0];
    let tool_name = get_tool_name(tool_call).unwrap();
    let tool_id = get_tool_id(tool_call).unwrap();
    let input = get_tool_input(tool_call);

    // Execute tool
    let result = execute_mock_tool(&tool_name, &input);
    println!("Tool result: {}", result);

    // Turn 2: Provide tool result
    history.push(ConversationMessage::ToolResult(
        ToolResult::success(&tool_id, &result).with_tool_name(&tool_name),
    ));

    let response2 = client.get_generation(&history, &config).await.unwrap();
    println!("Turn 2: {}", response2.content);

    // Should mention 42 (15 + 27)
    assert!(
        response2.content.contains("42"),
        "Response should contain the result 42"
    );
}

// ============================================================================
// Google Native Tools Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires GOOGLE_API_KEY"]
async fn test_google_native_tools_single_call() {
    let api_key = get_google_key().expect("GOOGLE_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the get_weather tool to get weather for 'New York' in fahrenheit",
    ))];

    let config = GenerationConfig::new("gemini-1.5-flash")
        .with_max_tokens(1024)
        .with_tools(vec![weather_tool_schema(), add_numbers_schema()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);

    assert!(
        !response.tool_calls.is_empty(),
        "Expected tool call in response"
    );

    let tool_call = &response.tool_calls[0];
    let tool_name = get_tool_name(tool_call);

    println!("Tool call detected: {:?}", tool_name);
    assert_eq!(tool_name, Some("get_weather".to_string()));

    // Execute mock tool
    let input = get_tool_input(tool_call);
    let result = execute_mock_tool("get_weather", &input);
    println!("Tool result: {}", result);
    assert!(result.contains("New York"));
}

#[tokio::test]
#[ignore = "Requires GOOGLE_API_KEY"]
async fn test_google_multi_turn_tool_conversation() {
    let api_key = get_google_key().expect("GOOGLE_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let config = GenerationConfig::new("gemini-1.5-flash")
        .with_max_tokens(1024)
        .with_tools(vec![add_numbers_schema()])
        .with_native_tools(true);

    // Turn 1: Ask to add numbers
    let mut history: Vec<ConversationMessage> = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use add_numbers to calculate 25 + 17",
    ))];

    let response1 = client.get_generation(&history, &config).await.unwrap();
    println!("Turn 1: {:?}", response1);

    assert!(!response1.tool_calls.is_empty());
    let tool_call = &response1.tool_calls[0];
    let tool_name = get_tool_name(tool_call).unwrap();
    let tool_id = get_tool_id(tool_call).unwrap_or_else(|| "google_call_1".to_string());
    let input = get_tool_input(tool_call);

    // Execute tool
    let result = execute_mock_tool(&tool_name, &input);
    println!("Tool result: {}", result);

    // Turn 2: Provide tool result
    history.push(ConversationMessage::ToolResult(
        ToolResult::success(&tool_id, &result).with_tool_name(&tool_name),
    ));

    let response2 = client.get_generation(&history, &config).await.unwrap();
    println!("Turn 2: {}", response2.content);

    // Should mention 42 (25 + 17)
    assert!(
        response2.content.contains("42"),
        "Response should contain the result 42"
    );
}

// ============================================================================
// Anthropic Native Tools Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_native_tools_single_call() {
    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the calculate tool to multiply 12 by 8",
    ))];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(1024)
        .with_tools(vec![calculator_tool_schema()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);

    assert!(
        !response.tool_calls.is_empty(),
        "Expected tool call in response"
    );

    let tool_call = &response.tool_calls[0];
    let tool_name = get_tool_name(tool_call);

    println!("Tool call detected: {:?}", tool_name);
    assert_eq!(tool_name, Some("calculate".to_string()));

    // Verify input has correct operation
    let input = get_tool_input(tool_call);
    assert_eq!(
        input.get("operation").and_then(|v| v.as_str()),
        Some("multiply")
    );
}

// ============================================================================
// 5-Turn Conversation Test
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_native_5_turn_conversation() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![
            add_numbers_schema(),
            multiply_numbers_schema(),
            search_tool_schema(),
        ])
        .with_native_tools(true);

    let mut history: Vec<ConversationMessage> = vec![ConversationMessage::Chat(ChatMessage::user(
        "I need you to: 1) Add 8 and 7, 2) Then multiply that result by 4, \
         3) Search for 'python tutorials' with max 2 results. \
         Do each step one at a time.",
    ))];

    let mut turn_count = 0;
    let max_turns = 5;
    let mut tool_calls_made: Vec<String> = Vec::new();

    while turn_count < max_turns {
        turn_count += 1;
        println!("\n--- Turn {} ---", turn_count);

        let response = client.get_generation(&history, &config).await.unwrap();

        if response.tool_calls.is_empty() {
            println!("Final response: {}", response.content);
            break;
        }

        // Process tool call
        let tool_call = &response.tool_calls[0];
        let tool_name = get_tool_name(tool_call).unwrap();
        let tool_id = get_tool_id(tool_call).unwrap();
        let input = get_tool_input(tool_call);

        println!("Tool: {} with input: {:?}", tool_name, input);
        tool_calls_made.push(tool_name.clone());

        // Execute tool
        let result = execute_mock_tool(&tool_name, &input);
        println!("Result: {}", result);

        // Add tool result to history
        history.push(ConversationMessage::ToolResult(
            ToolResult::success(&tool_id, &result).with_tool_name(&tool_name),
        ));

        // Add continuation prompt
        history.push(ConversationMessage::Chat(ChatMessage::user(
            "Continue with the next step.",
        )));
    }

    println!("\nTool calls made: {:?}", tool_calls_made);
    assert!(
        tool_calls_made.len() >= 2,
        "Should have made at least 2 tool calls"
    );
}

// ============================================================================
// Backwards Compatibility Test
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_backwards_compatibility_single_tool() {
    // Test that single tool patterns still work correctly
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 10 + 5 using the add_numbers function",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![add_numbers_schema()])
        .with_native_tools(true);

    let response = client.get_generation(&messages, &config).await.unwrap();

    // Should get either a tool call or text response
    if !response.tool_calls.is_empty() {
        let tool_call = &response.tool_calls[0];
        let tool_name = get_tool_name(tool_call);
        println!("Got tool call: {:?}", tool_name);
        assert_eq!(tool_name, Some("add_numbers".to_string()));

        // Execute tool
        let input = get_tool_input(tool_call);
        let result = execute_mock_tool("add_numbers", &input);
        println!("Tool result: {}", result);
        assert_eq!(result, "15");
    } else {
        println!("Got text response: {}", response.content);
        // Text response is also valid
    }
}

// ============================================================================
// ToolCall Creation Tests
// ============================================================================

#[test]
fn test_toolcall_from_anthropic_format() {
    let anthropic_format = serde_json::json!({
        "type": "tool_use",
        "id": "toolu_123",
        "name": "get_weather",
        "input": {"location": "San Francisco"}
    });

    let tool_call = ToolCall::from_provider_format(anthropic_format.clone(), Provider::Anthropic).unwrap();

    assert_eq!(tool_call.tool_use_id, "toolu_123");
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(
        tool_call.args.get("location").and_then(|v| v.as_str()),
        Some("San Francisco")
    );
}

#[test]
fn test_toolcall_from_openai_format() {
    let openai_format = serde_json::json!({
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"New York\"}"
        }
    });

    let tool_call = ToolCall::from_provider_format(openai_format.clone(), Provider::OpenAI).unwrap();

    assert_eq!(tool_call.tool_use_id, "call_abc123");
    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(
        tool_call.args.get("location").and_then(|v| v.as_str()),
        Some("New York")
    );
}

#[test]
fn test_toolcall_from_google_format() {
    let google_format = serde_json::json!({
        "functionCall": {
            "name": "get_weather",
            "args": {"location": "Tokyo"}
        }
    });

    let tool_call = ToolCall::from_provider_format(google_format.clone(), Provider::GoogleGenAI).unwrap();

    assert_eq!(tool_call.tool_name, "get_weather");
    assert_eq!(
        tool_call.args.get("location").and_then(|v| v.as_str()),
        Some("Tokyo")
    );
}

// ============================================================================
// ToolCall Provider Conversion Tests
// ============================================================================

#[test]
fn test_toolcall_to_anthropic_format() {
    let tool_call = ToolCall {
        tool_use_id: "toolu_123".to_string(),
        tool_name: "get_weather".to_string(),
        args: serde_json::json!({"location": "Paris"}),
        raw_arguments: None,
        signature: None,
    };

    let format = tool_call.to_provider_assistant_message(Provider::Anthropic);

    assert_eq!(format["role"], "assistant");
    let content = format["content"].as_array().unwrap();
    assert_eq!(content[0]["type"], "tool_use");
    assert_eq!(content[0]["id"], "toolu_123");
    assert_eq!(content[0]["name"], "get_weather");
}

#[test]
fn test_toolcall_to_openai_format() {
    let tool_call = ToolCall {
        tool_use_id: "call_abc".to_string(),
        tool_name: "get_weather".to_string(),
        args: serde_json::json!({"location": "London"}),
        raw_arguments: None,
        signature: None,
    };

    let format = tool_call.to_provider_assistant_message(Provider::OpenAI);

    assert_eq!(format["role"], "assistant");
    let tool_calls = format["tool_calls"].as_array().unwrap();
    assert_eq!(tool_calls[0]["id"], "call_abc");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");
}

#[test]
fn test_toolcall_to_google_format() {
    let tool_call = ToolCall {
        tool_use_id: "google_123".to_string(),
        tool_name: "get_weather".to_string(),
        args: serde_json::json!({"location": "Berlin"}),
        raw_arguments: None,
        signature: None,
    };

    let format = tool_call.to_provider_assistant_message(Provider::GoogleGenAI);

    assert_eq!(format["role"], "model");
    let parts = format["parts"].as_array().unwrap();
    assert_eq!(parts[0]["functionCall"]["name"], "get_weather");
}

// ============================================================================
// ToolResult Provider Format Tests
// ============================================================================

#[test]
fn test_toolresult_to_anthropic_format() {
    let result = ToolResult::success("toolu_123", "Weather is sunny")
        .with_tool_name("get_weather");

    let format = result.to_provider_format(Provider::Anthropic);

    assert_eq!(format["type"], "tool_result");
    assert_eq!(format["tool_use_id"], "toolu_123");
    assert_eq!(format["content"], "Weather is sunny");
}

#[test]
fn test_toolresult_to_openai_format() {
    let result = ToolResult::success("call_abc", "Weather is cloudy")
        .with_tool_name("get_weather");

    let format = result.to_provider_format(Provider::OpenAI);

    assert_eq!(format["role"], "tool");
    assert_eq!(format["tool_call_id"], "call_abc");
    assert_eq!(format["content"], "Weather is cloudy");
}

#[test]
fn test_toolresult_to_google_format() {
    let result = ToolResult::success("google_123", "Weather is rainy")
        .with_tool_name("get_weather");

    let format = result.to_provider_format(Provider::GoogleGenAI);

    assert_eq!(format["functionResponse"]["name"], "get_weather");
    assert_eq!(format["functionResponse"]["response"]["result"], "Weather is rainy");
}

// ============================================================================
// Error ToolResult Tests
// ============================================================================

#[test]
fn test_toolresult_error() {
    let result = ToolResult::error("toolu_123", "Division by zero");

    let format = result.to_provider_format(Provider::Anthropic);

    assert_eq!(format["type"], "tool_result");
    assert_eq!(format["is_error"], true);
    assert_eq!(format["content"], "Division by zero");
}
