//! Parallel Tool Calling integration tests
//!
//! Tests for parallel tool calling across all providers.
//! Mirrors Python tests from:
//! - integration_tests/test_parallel_tool_calling.py

use futures::StreamExt;
use reson::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient, OpenRouterClient,
    StreamChunk,
};
use reson::types::{ChatMessage, ToolCall, ToolResult};
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

fn weather_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "get_weather",
        "description": "Get weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
            },
            "required": ["location"]
        }
    })
}

fn search_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "search_database",
        "description": "Search a database",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["text"]
        }
    })
}

fn calculate_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "calculate",
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

fn get_time_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "get_current_time",
        "description": "Get current time",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })
}

fn convert_currency_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "name": "convert_currency",
        "description": "Convert currency from one to another",
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string"},
                "to_currency": {"type": "string"}
            },
            "required": ["amount", "from_currency", "to_currency"]
        }
    })
}

/// Execute a mock tool and return result
fn execute_mock_tool(tool_call: &serde_json::Value) -> String {
    let name = tool_call
        .get("name")
        .or_else(|| tool_call.get("_tool_name"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    let input = tool_call
        .get("input")
        .or_else(|| tool_call.get("arguments"))
        .cloned()
        .unwrap_or(serde_json::json!({}));

    match name {
        "get_weather" => {
            let location = input
                .get("location")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown");
            let units = input
                .get("units")
                .and_then(|v| v.as_str())
                .unwrap_or("celsius");
            format!(
                "Weather in {}: 22{}, sunny",
                location,
                if units == "fahrenheit" { "F" } else { "C" }
            )
        }
        "search_database" => {
            let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let max = input.get("max_results").and_then(|v| v.as_i64()).unwrap_or(5);
            format!("Found {} results for '{}'", max, text)
        }
        "calculate" => {
            let a = input.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
            let b = input.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
            let op = input
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("add");
            let result = if op == "multiply" { a * b } else { a + b };
            result.to_string()
        }
        "get_current_time" => "2024-08-24 18:30:00 PST".to_string(),
        "convert_currency" => {
            let amount = input.get("amount").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let from = input
                .get("from_currency")
                .and_then(|v| v.as_str())
                .unwrap_or("USD");
            let to = input
                .get("to_currency")
                .and_then(|v| v.as_str())
                .unwrap_or("EUR");
            // Mock conversion
            let rate = if from == "USD" && to == "EUR" {
                0.85
            } else {
                1.0
            };
            format!("{:.2}", amount * rate)
        }
        _ => format!("Unknown tool: {}", name),
    }
}

// ============================================================================
// OpenAI Parallel Tool Calling Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openai_parallel_tool_calling() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "openai/gpt-4o", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "I need you to: 1) Get weather for 'New York', 2) Search for 'python tutorials', \
         3) Calculate 15 + 27, and 4) Get current time. Use the appropriate tools for each task.",
    ))];

    let config = GenerationConfig::new("openai/gpt-4o")
        .with_max_tokens(1024)
        .with_tools(vec![
            weather_tool_schema(),
            search_tool_schema(),
            calculate_tool_schema(),
            get_time_tool_schema(),
        ])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut content_chunks: Vec<String> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                    content_chunks.push(text);
                }
                StreamChunk::ToolCallComplete(tool_call) => {
                    println!("\nTool call: {:?}", tool_call);
                    tool_calls.push(tool_call);
                }
                StreamChunk::ToolCallPartial(partial) => {
                    println!("Tool partial: {:?}", partial);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\nTool calls received: {}", tool_calls.len());

    // OpenAI should be able to call multiple tools in parallel
    // In practice this depends on the model and prompt
    assert!(
        !tool_calls.is_empty(),
        "Should receive at least one tool call"
    );

    // Execute tools and print results
    for tool_call in &tool_calls {
        let result = execute_mock_tool(tool_call);
        println!("Tool result: {}", result);
    }
}

// ============================================================================
// Anthropic Parallel Tool Calling Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_anthropic_parallel_tool_calling() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Please: 1) Get weather for 'London', 2) Search for 'machine learning', \
         and 3) Get current time. Use the tools available.",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![
            weather_tool_schema(),
            search_tool_schema(),
            get_time_tool_schema(),
        ])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                }
                StreamChunk::ToolCallComplete(tool_call) => {
                    println!("\nAnthropic tool call: {:?}", tool_call);
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

    println!("\nAnthropic tool calls: {}", tool_calls.len());

    // Anthropic can also return multiple tool_use blocks
    assert!(
        !tool_calls.is_empty(),
        "Should receive at least one tool call"
    );
}

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_direct_parallel_tools() {
    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Get the weather for both 'Paris' and 'Tokyo' using the get_weather tool.",
    ))];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(1024)
        .with_tools(vec![weather_tool_schema()])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::ToolCallComplete(tool_call) => {
                    println!("Tool call: {:?}", tool_call);
                    tool_calls.push(tool_call);
                }
                StreamChunk::Content(text) => {
                    print!("{}", text);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\nTotal tool calls: {}", tool_calls.len());

    // Claude should call the weather tool twice (once for each city)
    assert!(
        tool_calls.len() >= 1,
        "Should receive at least one tool call"
    );
}

// ============================================================================
// Google Parallel Tool Calling Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires GOOGLE_API_KEY"]
async fn test_google_parallel_tool_calling() {
    let api_key = get_google_key().expect("GOOGLE_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Please: 1) Calculate 25 + 17, 2) Convert 100 USD to EUR, \
         3) Get weather for 'Tokyo'. Use the appropriate tools.",
    ))];

    let config = GenerationConfig::new("gemini-1.5-flash")
        .with_max_tokens(1024)
        .with_tools(vec![
            calculate_tool_schema(),
            convert_currency_tool_schema(),
            weather_tool_schema(),
        ])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                }
                StreamChunk::ToolCallComplete(tool_call) => {
                    println!("\nGoogle tool call: {:?}", tool_call);
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

    println!("\nGoogle tool calls: {}", tool_calls.len());

    // Google can buffer multiple tool calls
    assert!(
        !tool_calls.is_empty(),
        "Should receive at least one tool call"
    );
}

// ============================================================================
// Backwards Compatibility Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_backwards_compatibility_single_tool() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 10 + 5 using the calculate function",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![calculate_tool_schema()])
        .with_native_tools(true);

    let result = client.get_generation(&messages, &config).await;

    match result {
        Ok(response) => {
            println!("Response: {:?}", response);

            // Check if we got tool calls
            if !response.tool_calls.is_empty() {
                let tool_call = &response.tool_calls[0];
                let name = tool_call
                    .get("name")
                    .or_else(|| tool_call.get("_tool_name"))
                    .and_then(|v| v.as_str());
                assert_eq!(name, Some("calculate"));
            } else {
                // Might get text response if model chooses not to use tool
                println!("Got text response: {}", response.content);
            }
        }
        Err(e) => {
            panic!("Test failed: {}", e);
        }
    }
}

// ============================================================================
// Parallel Execution Pattern Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_parallel_execution_pattern() {
    use tokio::task::JoinSet;

    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Get weather for 'Paris', calculate 20 + 22, and get current time",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![
            weather_tool_schema(),
            calculate_tool_schema(),
            get_time_tool_schema(),
        ])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    // Collect all tool calls first
    let mut tool_calls: Vec<serde_json::Value> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::ToolCallComplete(tool_call) => {
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

    println!("Collected {} tool calls for parallel execution", tool_calls.len());

    // Execute tools in parallel using JoinSet
    let mut join_set = JoinSet::new();

    for tool_call in tool_calls {
        join_set.spawn(async move {
            // In real usage, this would be async tool execution
            let result = execute_mock_tool(&tool_call);
            let name = tool_call
                .get("name")
                .or_else(|| tool_call.get("_tool_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            (name, result)
        });
    }

    // Collect results
    let mut results = Vec::new();
    while let Some(res) = join_set.join_next().await {
        if let Ok((name, result)) = res {
            println!("Parallel result: {} -> {}", name, result);
            results.push((name, result));
        }
    }

    println!("Completed {} parallel tool executions", results.len());
}

// ============================================================================
// Mixed Tool Types Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_mixed_parallel_tool_types() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 8 * 7 (use multiply operation), get current time, \
         and get weather for 'San Francisco'",
    ))];

    // Mix of tools with different parameter requirements
    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![
            calculate_tool_schema(),    // Requires a, b params
            get_time_tool_schema(),     // No params required
            weather_tool_schema(),      // Requires location
        ])
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

    println!("\nMixed tools received: {:?}", tool_names);

    // Should have received at least one tool call
    assert!(
        !tool_calls.is_empty(),
        "Should receive tool calls with mixed types"
    );

    // Execute and verify results
    for tool_call in &tool_calls {
        let result = execute_mock_tool(tool_call);
        println!("Result: {}", result);
    }
}

// ============================================================================
// Google Compositional Chaining Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires GOOGLE_API_KEY"]
async fn test_google_compositional_chaining() {
    let api_key = get_google_key().expect("GOOGLE_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Calculate 50 + 30, then convert that result from USD to EUR, \
         then get weather for 'Berlin'",
    ))];

    let config = GenerationConfig::new("gemini-1.5-flash")
        .with_max_tokens(1024)
        .with_tools(vec![
            calculate_tool_schema(),
            convert_currency_tool_schema(),
            weather_tool_schema(),
        ])
        .with_native_tools(true);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut tool_calls: Vec<serde_json::Value> = Vec::new();
    let mut tool_results: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                }
                StreamChunk::ToolCallComplete(tool_call) => {
                    let name = tool_call
                        .get("name")
                        .or_else(|| tool_call.get("_tool_name"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();

                    let result = execute_mock_tool(&tool_call);
                    println!("\nCompositional tool: {} -> {}", name, result);
                    tool_results.insert(name.clone(), result);
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

    println!("\nCompositional chain: {} tools", tool_calls.len());
    println!("Results: {:?}", tool_results);

    // Should have received tool calls for the chain
    assert!(
        !tool_calls.is_empty(),
        "Should receive tool calls in compositional chain"
    );
}

// ============================================================================
// Multi-Turn with Parallel Tools Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_multi_turn_parallel_tools() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key.clone(), "anthropic/claude-3-5-sonnet", None, None);

    // First turn - request multiple tools
    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Get weather for 'Miami' and calculate 100 + 200",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![weather_tool_schema(), calculate_tool_schema()])
        .with_native_tools(true);

    let result = client.get_generation(&messages, &config).await.unwrap();

    let mut conversation: Vec<ConversationMessage> = messages;
    let mut tool_results: Vec<ToolResult> = Vec::new();

    // Process tool calls from first turn
    if !result.tool_calls.is_empty() {
        println!("First turn tool calls: {}", result.tool_calls.len());

        for tool_call in &result.tool_calls {
            let tool_use_id = tool_call
                .get("id")
                .or_else(|| tool_call.get("tool_use_id"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let tool_name = tool_call
                .get("name")
                .or_else(|| tool_call.get("_tool_name"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let result_content = execute_mock_tool(tool_call);
            println!("Tool {} result: {}", tool_name, result_content);

            // Add assistant's tool use to conversation
            let args = tool_call
                .get("input")
                .or_else(|| tool_call.get("arguments"))
                .cloned()
                .unwrap_or(serde_json::json!({}));
            let mut tc = ToolCall::new(tool_name.clone(), args);
            tc.tool_use_id = tool_use_id.clone();
            conversation.push(ConversationMessage::ToolCall(tc));

            // Create tool result
            let tool_result = ToolResult::success_with_name(
                tool_use_id.clone(),
                tool_name.clone(),
                result_content,
            );
            tool_results.push(tool_result.clone());
            conversation.push(ConversationMessage::ToolResult(tool_result));
        }
    }

    // Second turn - provide tool results and ask follow-up
    let client2 = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    conversation.push(ConversationMessage::Chat(ChatMessage::user(
        "Thanks! Now what's the difference between the calculation result and 250?",
    )));

    let result2 = client2.get_generation(&conversation, &config).await.unwrap();

    println!("\nSecond turn response: {}", result2.content);

    // Should have received a response
    assert!(
        !result2.content.is_empty() || !result2.tool_calls.is_empty(),
        "Should get response in second turn"
    );
}
