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

use reson_agentic::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient,
    OAIClient, OpenRouterClient,
};
use reson_agentic::schema::{AnthropicSchemaGenerator, GoogleSchemaGenerator, OpenAISchemaGenerator, SchemaGenerator};
use reson_agentic::types::{ChatMessage, Provider, ToolCall, ToolResult};
use reson_agentic::utils::ConversationMessage;
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

// ============================================================================
// Anthropic Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires ANTHROPIC_API_KEY"]
async fn test_anthropic_simple_generation() {
    let api_key = get_anthropic_key().expect("ANTHROPIC_API_KEY not set");
    let client = AnthropicClient::new(api_key, "claude-haiku-4-5-20251001");

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::system("You are a helpful assistant. Respond concisely.")),
        ConversationMessage::Chat(ChatMessage::user("What is 2+2? Just give the number.")),
    ];

    let config = GenerationConfig::new("claude-haiku-4-5-20251001")
        .with_max_tokens(100)
        .with_temperature(0.0);

    let response = client.get_generation(&messages, &config).await.unwrap();

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

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Response: {:?}", response);

    // Should have tool calls
    assert!(!response.tool_calls.is_empty(), "Expected tool call");

    // Verify tool call structure
    let tool_call = &response.tool_calls[0];
    assert_eq!(tool_call.get("name").and_then(|v| v.as_str()), Some("add_numbers"));

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
    let tool_use_id = tool_call_json.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let tool_name = tool_call_json.get("name").and_then(|v| v.as_str()).unwrap_or("add_numbers");
    let tool_input = tool_call_json.get("input").cloned().unwrap_or(serde_json::json!({}));

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

    let response = client.get_generation(&messages, &config).await.unwrap();

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
// Google GenAI Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_simple_generation() {
    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::system("Be concise.")),
        ConversationMessage::Chat(ChatMessage::user("What is 5+5? Just the number.")),
    ];

    let config = GenerationConfig::new("gemini-1.5-flash")
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
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Use the add_numbers tool to add 50 and 75",
    ))];

    let config = GenerationConfig::new("gemini-1.5-flash")
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
    let client = GoogleGenAIClient::new(api_key, "gemini-2.0-flash-thinking-exp")
        .with_thinking_budget(1024);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What are the prime factors of 360? Think through this step by step.",
    ))];

    let config = GenerationConfig::new("gemini-2.0-flash-thinking-exp")
        .with_max_tokens(2048);

    let response = client.get_generation(&messages, &config).await.unwrap();

    println!("Content: {}", response.content);
    if let Some(reasoning) = &response.reasoning {
        println!("Reasoning: {}", reasoning);
    }

    assert!(!response.content.is_empty());
    // Should mention prime factors
    assert!(
        response.content.contains("2") || response.content.contains("3") || response.content.contains("5"),
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

    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "You must use the add_numbers tool to calculate 25 + 35. Do not calculate it yourself - use the tool.",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(1024)
        .with_tools(vec![openai_add_tool()])
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
                reson_agentic::providers::StreamChunk::Usage { input_tokens, output_tokens, .. } => {
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

    let tool_call = &tool_calls[0];
    let name = tool_call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str());
    assert_eq!(name, Some("add_numbers"));
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
            .get("name")
            .or_else(|| tool_call.get("_tool_name"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let tool_id = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
        let input = tool_call.get("input").cloned().unwrap_or(serde_json::json!({}));

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

        // Add tool result to history
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

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

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
                    reson_agentic::providers::StreamChunk::Usage { input_tokens, output_tokens, .. } => {
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
    let client = GoogleGenAIClient::new(api_key, "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "List the days of the week, one per line.",
    ))];

    let config = GenerationConfig::new("gemini-1.5-flash")
        .with_max_tokens(200);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut full_content = String::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => {
                match chunk {
                    reson_agentic::providers::StreamChunk::Content(text) => {
                        print!("{}", text);
                        full_content.push_str(&text);
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
    let client = GoogleGenAIClient::new("invalid-key", "gemini-1.5-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
    let config = GenerationConfig::new("gemini-1.5-flash");

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
    assert!(Provider::GoogleGenAI.supports_native_tools());
    assert!(Provider::GoogleAnthropic.supports_native_tools());
    assert!(Provider::OpenRouter.supports_native_tools());
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
    assert!(
        !tool_calls.is_empty(),
        "Expected tool call via streaming"
    );

    // Streaming tool calls have format: {"function": {"name": ..., "arguments": ...}, "id": ...}
    let tool_call = &tool_calls[0];
    let name = tool_call
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .or_else(|| tool_call.get("name").and_then(|n| n.as_str()));
    assert_eq!(name, Some("add_numbers"));
}
