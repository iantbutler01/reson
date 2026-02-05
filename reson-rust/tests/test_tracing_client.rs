//! Integration tests for TracingInferenceClient
//!
//! These tests require API keys set via environment variables:
//! - ANTHROPIC_API_KEY for Anthropic tests
//! - OPENAI_API_KEY for OpenAI tests
//!
//! Run with: cargo test --test test_tracing_client -- --ignored

use reson_agentic::providers::{
    AnthropicClient, CostInfo, GenerationConfig, InferenceClient, MemoryCostStore,
    OAIClient, TracingInferenceClient,
};
use reson_agentic::types::ChatMessage;
use reson_agentic::utils::ConversationMessage;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

fn get_anthropic_key() -> Option<String> {
    std::env::var("ANTHROPIC_API_KEY").ok()
}

fn get_openai_key() -> Option<String> {
    std::env::var("OPENAI_API_KEY").ok()
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY
async fn test_tracing_client_anthropic_cost_tracking() {
    let api_key = match get_anthropic_key() {
        Some(k) => k,
        None => {
            eprintln!("Skipping: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let client = AnthropicClient::new(api_key, "claude-3-haiku-20240307");
    let store = Arc::new(MemoryCostStore::new());
    let tracing_client =
        TracingInferenceClient::new(Box::new(client)).with_cost_store(store.clone());

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Say hello"))];
    let config = GenerationConfig::new("claude-3-haiku-20240307").with_max_tokens(100);

    let response = tracing_client.get_generation(&messages, &config).await;
    assert!(response.is_ok(), "Generation failed: {:?}", response.err());

    let response = response.unwrap();
    println!("Response: {}", response.content);
    println!(
        "Usage: {} input, {} output tokens",
        response.usage.input_tokens, response.usage.output_tokens
    );

    // Check cost was tracked
    let credits = store.credits();
    println!("Credits used: {} microdollars (${:.6})", credits, credits as f64 / 1_000_000.0);
    assert!(credits > 0, "No credits tracked");

    // Verify cost is reasonable for Haiku (very cheap)
    // Even a small request should cost at least a few microdollars
    assert!(credits < 1_000_000, "Cost seems too high: {} microdollars", credits);
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY
async fn test_tracing_client_callback_invoked() {
    let api_key = match get_anthropic_key() {
        Some(k) => k,
        None => {
            eprintln!("Skipping: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let client = AnthropicClient::new(api_key, "claude-3-haiku-20240307");
    let callback_count = Arc::new(AtomicU64::new(0));
    let callback_cost = Arc::new(AtomicU64::new(0));

    let count_clone = callback_count.clone();
    let cost_clone = callback_cost.clone();

    let callback: reson_agentic::providers::TraceCallback = Arc::new(
        move |_id, _msgs, _resp, cost: &CostInfo| {
            count_clone.fetch_add(1, Ordering::Relaxed);
            cost_clone.fetch_add(cost.total_microdollars(), Ordering::Relaxed);
            Box::pin(async {})
        },
    );

    let tracing_client =
        TracingInferenceClient::new(Box::new(client)).with_trace_callback(callback);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hi"))];
    let config = GenerationConfig::new("claude-3-haiku-20240307").with_max_tokens(50);

    let _ = tracing_client.get_generation(&messages, &config).await;

    assert_eq!(
        callback_count.load(Ordering::Relaxed),
        1,
        "Callback should be invoked once"
    );
    assert!(
        callback_cost.load(Ordering::Relaxed) > 0,
        "Callback should receive cost info"
    );
}

#[tokio::test]
#[ignore] // Requires OPENAI_API_KEY
async fn test_tracing_client_openai_cost_tracking() {
    let api_key = match get_openai_key() {
        Some(k) => k,
        None => {
            eprintln!("Skipping: OPENAI_API_KEY not set");
            return;
        }
    };

    let client = OAIClient::new(api_key, "gpt-4o-mini");
    let store = Arc::new(MemoryCostStore::new());
    let tracing_client =
        TracingInferenceClient::new(Box::new(client)).with_cost_store(store.clone());

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Say hello"))];
    let config = GenerationConfig::new("gpt-4o-mini").with_max_tokens(100);

    let response = tracing_client.get_generation(&messages, &config).await;
    assert!(response.is_ok(), "Generation failed: {:?}", response.err());

    let response = response.unwrap();
    println!("Response: {}", response.content);
    println!(
        "Usage: {} input, {} output tokens",
        response.usage.input_tokens, response.usage.output_tokens
    );

    let credits = store.credits();
    println!("Credits used: {} microdollars (${:.6})", credits, credits as f64 / 1_000_000.0);
    assert!(credits > 0, "No credits tracked");
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY
async fn test_tracing_client_multiple_requests_accumulate() {
    let api_key = match get_anthropic_key() {
        Some(k) => k,
        None => {
            eprintln!("Skipping: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let client = AnthropicClient::new(api_key, "claude-3-haiku-20240307");
    let store = Arc::new(MemoryCostStore::new());
    let tracing_client =
        TracingInferenceClient::new(Box::new(client)).with_cost_store(store.clone());

    let config = GenerationConfig::new("claude-3-haiku-20240307").with_max_tokens(50);

    // Make 3 requests
    for i in 0..3 {
        let messages = vec![ConversationMessage::Chat(ChatMessage::user(format!(
            "Say the number {}",
            i
        )))];
        let _ = tracing_client.get_generation(&messages, &config).await;
    }

    let credits = store.credits();
    println!(
        "Total credits after 3 requests: {} microdollars (${:.6})",
        credits,
        credits as f64 / 1_000_000.0
    );

    // Should have accumulated costs from all 3 requests
    assert!(credits > 0, "No credits tracked");
}

#[tokio::test]
#[ignore] // Requires ANTHROPIC_API_KEY and writes to filesystem
async fn test_tracing_client_trace_output() {
    let api_key = match get_anthropic_key() {
        Some(k) => k,
        None => {
            eprintln!("Skipping: ANTHROPIC_API_KEY not set");
            return;
        }
    };

    let trace_dir = "/tmp/reson_trace_test";
    // Clean up from previous runs
    let _ = std::fs::remove_dir_all(trace_dir);

    let client = AnthropicClient::new(api_key, "claude-3-haiku-20240307");
    let tracing_client = TracingInferenceClient::new(Box::new(client))
        .with_trace_output(trace_dir);

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
    let config = GenerationConfig::new("claude-3-haiku-20240307").with_max_tokens(50);

    let _ = tracing_client.get_generation(&messages, &config).await;

    // Check trace file was written
    let entries: Vec<_> = std::fs::read_dir(trace_dir)
        .expect("Trace dir should exist")
        .collect();

    assert!(!entries.is_empty(), "Should have written at least one trace file");

    // Read and validate trace content
    let entry = entries.into_iter().next().unwrap().unwrap();
    let content = std::fs::read_to_string(entry.path()).unwrap();
    let json: serde_json::Value = serde_json::from_str(&content).unwrap();

    assert!(json.get("request_id").is_some());
    assert!(json.get("timestamp").is_some());
    assert!(json.get("messages").is_some());
    assert!(json.get("response").is_some());
    assert!(json.get("usage").is_some());
    assert!(json.get("cost").is_some());

    println!("Trace file content:\n{}", serde_json::to_string_pretty(&json).unwrap());

    // Clean up
    let _ = std::fs::remove_dir_all(trace_dir);
}

#[tokio::test]
async fn test_cost_info_calculations() {
    use reson_agentic::types::TokenUsage;

    // Test Sonnet pricing
    let usage = TokenUsage::new(1000, 500, 0);
    let cost = CostInfo::from_usage(&usage, "claude-3-sonnet");
    // 1000 * 3 + 500 * 15 = 3000 + 7500 = 10500 microdollars
    assert!(cost.microdollar_cost >= 10500 && cost.microdollar_cost <= 10600);
    println!("Sonnet cost: {} microdollars", cost.microdollar_cost);

    // Test Haiku pricing
    let cost_haiku = CostInfo::from_usage(&usage, "claude-3-haiku");
    // 1000 * 0.8 + 500 * 4 = 800 + 2000 = 2800 microdollars
    assert!(cost_haiku.microdollar_cost >= 2800 && cost_haiku.microdollar_cost <= 2900);
    println!("Haiku cost: {} microdollars", cost_haiku.microdollar_cost);

    // Test Opus pricing
    let cost_opus = CostInfo::from_usage(&usage, "claude-3-opus");
    // 1000 * 15 + 500 * 75 = 15000 + 37500 = 52500 microdollars
    assert!(cost_opus.microdollar_cost >= 52500 && cost_opus.microdollar_cost <= 52600);
    println!("Opus cost: {} microdollars", cost_opus.microdollar_cost);

    // Test GPT-4o-mini pricing
    let cost_mini = CostInfo::from_usage(&usage, "gpt-4o-mini");
    // 1000 * 1.1 + 500 * 4.4 = 1100 + 2200 = 3300 microdollars
    assert!(cost_mini.microdollar_cost >= 3300 && cost_mini.microdollar_cost <= 3400);
    println!("GPT-4o-mini cost: {} microdollars", cost_mini.microdollar_cost);

    // Test with cache tokens
    let usage_with_cache = TokenUsage::new(1000, 500, 500);
    let cost_cached = CostInfo::from_usage(&usage_with_cache, "claude-3-sonnet");
    // 1000 * 3 + 500 * 15 + 500 * 0.3 = 3000 + 7500 + 150 = 10650 microdollars
    assert!(cost_cached.microdollar_cost >= 10600 && cost_cached.microdollar_cost <= 10700);
    println!("Sonnet with cache cost: {} microdollars", cost_cached.microdollar_cost);

    // Test dollar conversion
    println!("10500 microdollars = ${}", CostInfo::from_usage(&usage, "claude-3-sonnet").total_dollars());

    // Test provider-reported cost (e.g., from OpenRouter)
    let provider_cost_dollars = 0.0025; // $0.0025 = 2500 microdollars
    let cost_with_provider = CostInfo::from_usage_with_provider_cost(&usage, "unknown-model", provider_cost_dollars);

    // Provider cost should be stored
    assert!(cost_with_provider.has_provider_cost());
    assert_eq!(cost_with_provider.provider_cost_microdollars, Some(2500));

    // total_microdollars should prefer provider cost
    assert_eq!(cost_with_provider.total_microdollars(), 2500);

    // calculated_microdollars returns the calculated cost (0 for unknown model)
    assert_eq!(cost_with_provider.calculated_microdollars(), 0);

    println!("Provider cost: {} microdollars (${:.6})",
        cost_with_provider.total_microdollars(),
        cost_with_provider.total_dollars());

    // Test with_provider_cost builder
    let cost_builder = CostInfo::from_usage(&usage, "claude-3-sonnet")
        .with_provider_cost(0.005); // $0.005 = 5000 microdollars
    assert!(cost_builder.has_provider_cost());
    assert_eq!(cost_builder.total_microdollars(), 5000);
    // calculated cost still available
    assert!(cost_builder.calculated_microdollars() >= 10500);
}
