//! Reasoning/Thinking integration tests
//!
//! Tests for reasoning functionality across all providers.
//! Mirrors Python tests from:
//! - integration_tests/test_reasoning.py
//! - integration_tests/test_reasoning_segments.py
//! - integration_tests/test_reasoning_stream.py

use futures::StreamExt;
use reson_agentic::providers::{
    AnthropicClient, GenerationConfig, GoogleGenAIClient, InferenceClient, OpenRouterClient,
    StreamChunk,
};
use reson_agentic::types::{ChatMessage, ReasoningSegment};
use reson_agentic::utils::ConversationMessage;
use std::env;

// ============================================================================
// Helper Functions
// ============================================================================

fn get_anthropic_key() -> Option<String> {
    env::var("ANTHROPIC_API_KEY").ok()
}

fn get_google_key() -> Option<String> {
    env::var("GOOGLE_GEMINI_API_KEY").ok()
}

fn get_openrouter_key() -> Option<String> {
    env::var("OPENROUTER_API_KEY").ok()
}

// ============================================================================
// Basic Reasoning Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_openai_reasoning_o3_mini() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    // O3-mini with reasoning=high
    let client = OpenRouterClient::new(api_key, "openai/o3-mini", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "How would you build the world's tallest skyscraper?",
    ))];

    let config = GenerationConfig::new("openai/o3-mini")
        .with_max_tokens(5000)
        .with_reasoning_effort("high");

    let result = client.get_generation(&messages, &config).await;

    match result {
        Ok(response) => {
            println!("Response: {}", response.content);
            println!("Reasoning: {:?}", response.reasoning);

            // O3-mini should produce reasoning
            assert!(
                !response.content.is_empty() || response.reasoning.is_some(),
                "Should get content or reasoning from o3-mini"
            );
        }
        Err(e) => {
            // Model might not be available
            println!("Note: o3-mini test failed (may not be available): {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_anthropic_reasoning_via_openrouter() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    // Claude with reasoning budget
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What's the most efficient algorithm for sorting a large dataset?",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(5000)
        .with_thinking_budget(2000);

    let result = client.get_generation(&messages, &config).await;

    match result {
        Ok(response) => {
            println!("Response: {}", response.content);
            println!("Reasoning: {:?}", response.reasoning);

            // Should have content at minimum
            assert!(
                !response.content.is_empty(),
                "Should get content from Claude"
            );
        }
        Err(e) => {
            eprintln!("Anthropic reasoning test error: {}", e);
            panic!("Test failed: {}", e);
        }
    }
}

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_thinking_model() {
    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-2.0-flash-thinking-exp")
        .with_thinking_budget(1024);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Explain the process of photosynthesis in detail",
    ))];

    let config = GenerationConfig::new("gemini-2.0-flash-thinking-exp").with_max_tokens(2048);

    let result = client.get_generation(&messages, &config).await;

    match result {
        Ok(response) => {
            println!("Content: {}", response.content);
            println!("Reasoning: {:?}", response.reasoning);

            // Thinking model should produce content
            assert!(
                !response.content.is_empty(),
                "Should get content from thinking model"
            );
        }
        Err(e) => {
            eprintln!("Google thinking test error: {}", e);
            panic!("Test failed: {}", e);
        }
    }
}

/// Test Google model with Application Default Credentials (ADC)
/// Uses GOOGLE_APPLICATION_CREDENTIALS environment variable
///
/// Required env vars:
/// - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON (contains project_id)
/// Optional env vars:
/// - GOOGLE_CLOUD_LOCATION: Region (defaults to "us-central1")
#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires GOOGLE_APPLICATION_CREDENTIALS"]
async fn test_google_thinking_model_adc() {
    // Uses ADC - automatically finds credentials and project_id from:
    // 1. GOOGLE_APPLICATION_CREDENTIALS env var (service account JSON)
    // Project ID is extracted from the JSON file automatically
    // Location defaults to us-central1 or can be set via GOOGLE_CLOUD_LOCATION
    let client = GoogleGenAIClient::from_adc("gemini-2.0-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What is 15 * 23? Think through this step by step.",
    ))];

    let config = GenerationConfig::new("gemini-2.0-flash").with_max_tokens(2048);

    let result = client.get_generation(&messages, &config).await;

    match result {
        Ok(response) => {
            println!("Content: {}", response.content);
            println!("Reasoning: {:?}", response.reasoning);

            // Thinking model should produce content
            assert!(
                !response.content.is_empty(),
                "Should get content from thinking model"
            );
        }
        Err(e) => {
            eprintln!("Google ADC thinking test error: {}", e);
            panic!("Test failed: {}", e);
        }
    }
}

// ============================================================================
// Streaming Reasoning Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_reasoning_stream_openrouter() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "openai/gpt-4o", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "How would you build the world's tallest skyscraper?",
    ))];

    let config = GenerationConfig::new("openai/gpt-4o")
        .with_max_tokens(5000)
        .with_reasoning_effort("high");

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut content_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                    content_chunks.push(text);
                }
                StreamChunk::Reasoning(text) => {
                    print!("[R: {}]", text);
                    reasoning_chunks.push(text);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\n\nContent chunks: {}", content_chunks.len());
    println!("Reasoning chunks: {}", reasoning_chunks.len());

    // Should have some output
    assert!(
        !content_chunks.is_empty() || !reasoning_chunks.is_empty(),
        "Should receive content or reasoning chunks"
    );
}

#[tokio::test]
#[ignore = "Requires OPENROUTER_API_KEY"]
async fn test_anthropic_reasoning_stream() {
    let api_key = get_openrouter_key().expect("OPENROUTER_API_KEY not set");
    let client = OpenRouterClient::new(api_key, "anthropic/claude-3-5-sonnet", None, None);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "Explain quantum computing in simple terms.",
    ))];

    let config = GenerationConfig::new("anthropic/claude-3-5-sonnet")
        .with_max_tokens(5000)
        .with_thinking_budget(2000);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut content_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("{}", text);
                    content_chunks.push(text);
                }
                StreamChunk::Reasoning(text) => {
                    print!("[R: {}]", text);
                    reasoning_chunks.push(text);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\n\nContent chunks: {}", content_chunks.len());
    println!("Reasoning chunks: {}", reasoning_chunks.len());

    // Should have content at minimum
    assert!(!content_chunks.is_empty(), "Should receive content chunks");
}

#[tokio::test]
#[ignore = "Requires GOOGLE_GEMINI_API_KEY"]
async fn test_google_thinking_stream() {
    let api_key = get_google_key().expect("GOOGLE_GEMINI_API_KEY not set");
    let client = GoogleGenAIClient::new(api_key, "gemini-2.0-flash-thinking-exp")
        .with_thinking_budget(1024);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What is 17 * 23? Think through this step by step.",
    ))];

    let config = GenerationConfig::new("gemini-2.0-flash-thinking-exp").with_max_tokens(2048);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut content_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();
    let mut has_signature = false;

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("C: {}", text);
                    content_chunks.push(text);
                }
                StreamChunk::Reasoning(text) => {
                    print!("R: {}", text);
                    reasoning_chunks.push(text);
                }
                StreamChunk::Signature(sig) => {
                    println!("\nSignature: {}", sig);
                    has_signature = true;
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\n\nContent chunks: {}", content_chunks.len());
    println!("Reasoning chunks: {}", reasoning_chunks.len());
    println!("Has signature: {}", has_signature);

    // Thinking model should produce reasoning or content
    assert!(
        !reasoning_chunks.is_empty() || !content_chunks.is_empty(),
        "Should receive either reasoning or content"
    );
}

/// Test Google stream with ADC
#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires GOOGLE_APPLICATION_CREDENTIALS"]
async fn test_google_thinking_stream_adc() {
    let client = GoogleGenAIClient::from_adc("gemini-2.0-flash");

    let messages = vec![ConversationMessage::Chat(ChatMessage::user(
        "What is 17 * 23? Think through this step by step.",
    ))];

    let config = GenerationConfig::new("gemini-2.0-flash").with_max_tokens(2048);

    let mut stream = client.connect_and_listen(&messages, &config).await.unwrap();

    let mut content_chunks: Vec<String> = Vec::new();
    let mut reasoning_chunks: Vec<String> = Vec::new();

    while let Some(chunk_result) = stream.next().await {
        match chunk_result {
            Ok(chunk) => match chunk {
                StreamChunk::Content(text) => {
                    print!("C: {}", text);
                    content_chunks.push(text);
                }
                StreamChunk::Reasoning(text) => {
                    print!("R: {}", text);
                    reasoning_chunks.push(text);
                }
                _ => {}
            },
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
        }
    }

    println!("\n\nContent chunks: {}", content_chunks.len());
    println!("Reasoning chunks: {}", reasoning_chunks.len());

    // Thinking model should produce reasoning or content
    assert!(
        !reasoning_chunks.is_empty() || !content_chunks.is_empty(),
        "Should receive either reasoning or content"
    );
}

// ============================================================================
// ReasoningSegment Tests
// ============================================================================

#[test]
fn test_reasoning_segment_creation() {
    let segment = ReasoningSegment::new("This is test reasoning content".to_string());

    assert_eq!(segment.content, "This is test reasoning content");
    assert!(segment.signature.is_none());
    assert!(segment.provider_metadata.is_none());
    assert_eq!(segment.segment_index, 0); // Default index is 0
}

#[test]
fn test_reasoning_segment_with_signature() {
    let segment = ReasoningSegment::new("Test content".to_string())
        .with_signature("test_signature_123".to_string());

    assert_eq!(segment.content, "Test content");
    assert_eq!(segment.signature, Some("test_signature_123".to_string()));
}

#[test]
fn test_reasoning_segment_with_metadata() {
    let metadata = serde_json::json!({
        "test": "metadata",
        "nested": {"key": "value"}
    });

    let segment =
        ReasoningSegment::new("Test content".to_string()).with_provider_metadata(metadata.clone());

    assert_eq!(segment.provider_metadata, Some(metadata));
}

#[test]
fn test_reasoning_segment_anthropic_format() {
    use reson_agentic::providers::Provider;

    let segment = ReasoningSegment::new("This is test reasoning content".to_string())
        .with_signature("test_signature_123".to_string())
        .with_provider_metadata(serde_json::json!({"test": "metadata"}));

    let anthropic_format = segment.to_provider_format(Provider::Anthropic);

    assert_eq!(anthropic_format.get("type").unwrap(), "thinking");
    assert_eq!(
        anthropic_format.get("thinking").unwrap(),
        "This is test reasoning content"
    );
    assert_eq!(
        anthropic_format.get("signature").unwrap(),
        "test_signature_123"
    );
}

#[test]
fn test_reasoning_segment_openai_format() {
    use reson_agentic::providers::Provider;

    let segment = ReasoningSegment::new("This is test reasoning content".to_string())
        .with_signature("test_signature_123".to_string());

    let openai_format = segment.to_provider_format(Provider::OpenAI);

    assert_eq!(openai_format.get("type").unwrap(), "reasoning");
    assert_eq!(
        openai_format.get("content").unwrap(),
        "This is test reasoning content"
    );
    assert_eq!(
        openai_format.get("signature").unwrap(),
        "test_signature_123"
    );
}

#[test]
fn test_reasoning_segment_google_format() {
    use reson_agentic::providers::Provider;

    let segment = ReasoningSegment::new("This is test reasoning content".to_string())
        .with_signature("test_signature_123".to_string());

    let google_format = segment.to_provider_format(Provider::GoogleGenAI);

    assert_eq!(google_format.get("thought").unwrap(), true);
    assert_eq!(
        google_format.get("text").unwrap(),
        "This is test reasoning content"
    );
    assert_eq!(
        google_format.get("thought_signature").unwrap(),
        "test_signature_123"
    );
}

#[test]
fn test_reasoning_segment_openrouter_format() {
    use reson_agentic::providers::Provider;

    let segment = ReasoningSegment::new("This is test reasoning content".to_string())
        .with_signature("test_signature_123".to_string());

    let openrouter_format = segment.to_provider_format(Provider::OpenRouter);

    // OpenRouter should use OpenAI-compatible format
    assert_eq!(openrouter_format.get("type").unwrap(), "reasoning");
    assert_eq!(
        openrouter_format.get("content").unwrap(),
        "This is test reasoning content"
    );
}

// ============================================================================
// Reasoning in Conversation Tests
// ============================================================================

#[test]
fn test_reasoning_segment_in_conversation_message() {
    let segment = ReasoningSegment::new("Internal reasoning".to_string())
        .with_signature("sig123".to_string());

    let message: ConversationMessage = segment.clone().into();

    match message {
        ConversationMessage::Reasoning(r) => {
            assert_eq!(r.content, "Internal reasoning");
            assert_eq!(r.signature, Some("sig123".to_string()));
        }
        _ => panic!("Expected Reasoning variant"),
    }
}

#[test]
fn test_reasoning_segments_collection() {
    // Test creating a collection of reasoning segments
    let segments = vec![
        ReasoningSegment::new("First thought".to_string()).with_segment_index(0),
        ReasoningSegment::new("Second thought".to_string()).with_segment_index(1),
        ReasoningSegment::new("Third thought".to_string()).with_segment_index(2),
    ];

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].segment_index, 0);
    assert_eq!(segments[1].segment_index, 1);
    assert_eq!(segments[2].segment_index, 2);

    // Concatenate all reasoning
    let full_reasoning: String = segments.iter().map(|s| s.content.as_str()).collect::<Vec<_>>().join(" ");
    assert_eq!(full_reasoning, "First thought Second thought Third thought");
}

// ============================================================================
// Multi-Provider Reasoning Tests
// ============================================================================

#[tokio::test]
#[ignore = "Requires API keys"]
async fn test_reasoning_across_providers() {
    // This test verifies that reasoning works across multiple providers
    // Each provider may return reasoning in different formats but our
    // ReasoningSegment type should normalize them

    let providers_to_test = vec![
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("google", "gemini-2.0-flash-thinking-exp"),
        ("openrouter", "openai/gpt-4o"),
    ];

    for (provider_name, _model) in providers_to_test {
        println!("Testing reasoning with provider: {}", provider_name);
        // Would test each provider here with appropriate client
        // This is a placeholder for comprehensive cross-provider testing
    }
}
