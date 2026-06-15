//! Gated live image-input tests.
//!
//! These tests are ignored by default and only do real provider calls when
//! `CHEVALIER_LIVE_IMAGE_TESTS=1` is present.

use chevalier_agentic::providers::{
    AnthropicClient, GenerationConfig, InferenceClient, OAIClient, OpenAIResponsesClient,
    OpenRouterClient, OpenRouterResponsesClient, StreamChunk,
};
use chevalier_agentic::types::{MediaPart, MediaSource, MultimodalMessage};
use chevalier_agentic::utils::ConversationMessage;
use futures::StreamExt;

const LETTER_L_PNG: &str = "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAAgKADAAQAAAABAAAAgAAAAABIjgR3AAAChElEQVR4Ae2dQW4bQBDD6iL//3KK/mDBcCCgZc4rKSA9Jx/8+f7+/tXfjsDv3XTLfwkkYPw5SEACxgTG811AAsYExvNdQALGBMbzXUACxgTG811AAsYExvNdQALGBMbzXUACxgTG811AAsYExvNdwFjA18X+5/PBtf/bd9RdAP6oOMEEOBxxSwIwOieYAIcjbkkARucEE+BwxC0JwOicYAIcjrglARidE0yAwxG3JACjc4IJcDjilgRgdE4wAQ5H3JIAjM4JJsDhiFsSgNE5wQQ4HHFLAjA6J5gAhyNuSQBG5wQT4HDELQnA6JxgAhyOuCUBGJ0TTIDDEbckAKNzgglwOOKWBGB0TjABDkfckgCMzgkmwOGIWxKA0TnBBDgccUsCMDonmACHI25JAEbnBBPgcMQtCcDonGACHI64JQEYnRNMgMMRtyQAo3OCCXA44pYEYHROMAEOR9ySAIzOCSbA4YhbEoDROcEEOBxxSwIwOieYAIcjbkkARucEE+BwxC0JwOicYAIcjrglARidE0yAwxG3JACjc4IJcDjilgRgdE4wAQ5H3JIAjM4Jnvx+wE/+tZ/89sBPdl+yF79t0AW8kD98k4BDuC/VCXihdPgmAYdwX6oT8ELp8E0CDuG+VCfghdLhmwQcwn2pTsALpcM3CTiE+1KdgBdKh28ScAj3pToBL5QO3yTgEO5LdQJeKB2+ScAh3JfqBLxQOnyTgEO4L9UJeKF0+OZz8T3n4f/7z1V3AWOlCUjAmMB4vgtIwJjAeL4LSMCYwHi+C0jAmMB4vgtIwJjAeL4LSMCYwHi+C0jAmMB4vgtIwJjAeL4LSMCYwHi+C0jAmMB4vgtIwJjAeP4PlA4M/SsN784AAAAASUVORK5CYII=";

fn live_image_tests_enabled() -> bool {
    std::env::var("CHEVALIER_LIVE_IMAGE_TESTS")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE"))
        .unwrap_or(false)
}

fn letter_l_message() -> Vec<ConversationMessage> {
    vec![ConversationMessage::Multimodal(MultimodalMessage::user(
        vec![
            MediaPart::text(
                "What uppercase letter is drawn in the image? Answer with the single letter only.",
            ),
            MediaPart::image(MediaSource::base64(LETTER_L_PNG, "image/png")),
        ],
    ))]
}

fn assert_reads_letter_l(provider: &str, model: &str, text: &str) {
    eprintln!("{provider} {model} image response: {text}");
    let normalized = text
        .trim()
        .trim_matches(|ch: char| matches!(ch, '.' | '`' | '"' | '\''));
    assert!(
        normalized.eq_ignore_ascii_case("l"),
        "{provider} {model} did not identify the letter L image; response was {text:?}"
    );
}

async fn stream_to_text<C>(client: &C, messages: &[ConversationMessage], model: &str) -> String
where
    C: InferenceClient + Sync,
{
    let config = GenerationConfig::new(model).with_max_tokens(64);
    let mut stream = client
        .connect_and_listen(messages, &config)
        .await
        .expect("stream should start");
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        if let StreamChunk::Content(part) = chunk.expect("stream chunk should succeed") {
            text.push_str(&part);
        }
    }
    text
}

#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and OPENAI_API_KEY"]
async fn live_openai_chat_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = std::env::var("CHEVALIER_LIVE_OPENAI_IMAGE_MODEL")
        .unwrap_or_else(|_| "gpt-4o-mini".to_string());
    let client = OAIClient::new(api_key, &model);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("OpenAI image generation should succeed");
    assert_reads_letter_l("openai-chat", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("openai-chat-stream", &model, &streamed);
}

#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and OPENAI_API_KEY"]
async fn live_openai_responses_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let model = std::env::var("CHEVALIER_LIVE_OPENAI_RESPONSES_IMAGE_MODEL")
        .unwrap_or_else(|_| "gpt-4.1-mini".to_string());
    let client = OpenAIResponsesClient::new(api_key, &model);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("OpenAI Responses image generation should succeed");
    assert_reads_letter_l("openai-responses", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("openai-responses-stream", &model, &streamed);
}

#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and ANTHROPIC_API_KEY"]
async fn live_anthropic_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let api_key = std::env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY not set");
    let model = std::env::var("CHEVALIER_LIVE_ANTHROPIC_IMAGE_MODEL")
        .unwrap_or_else(|_| "claude-haiku-4-5-20251001".to_string());
    let client = AnthropicClient::new(api_key, &model);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("Anthropic image generation should succeed");
    assert_reads_letter_l("anthropic", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("anthropic-stream", &model, &streamed);
}

#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and GOOGLE_GEMINI_API_KEY"]
async fn live_google_gemini_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let api_key = std::env::var("GOOGLE_GEMINI_API_KEY").expect("GOOGLE_GEMINI_API_KEY not set");
    let model = std::env::var("CHEVALIER_LIVE_GOOGLE_IMAGE_MODEL")
        .unwrap_or_else(|_| "gemini-2.5-flash".to_string());
    let client = chevalier_agentic::providers::GoogleGenAIClient::new(api_key, &model);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("Google Gemini image generation should succeed");
    assert_reads_letter_l("google-gemini", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("google-gemini-stream", &model, &streamed);
}

#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and OPENROUTER_API_KEY"]
async fn live_openrouter_chat_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
    let model = std::env::var("CHEVALIER_LIVE_OPENROUTER_IMAGE_MODEL")
        .unwrap_or_else(|_| "openai/gpt-4o-mini".to_string());
    let client = OpenRouterClient::new(api_key, &model, None, None);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("OpenRouter image generation should succeed");
    assert_reads_letter_l("openrouter-chat", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("openrouter-chat-stream", &model, &streamed);
}

#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and OPENROUTER_API_KEY"]
async fn live_openrouter_responses_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY not set");
    let model = std::env::var("CHEVALIER_LIVE_OPENROUTER_RESPONSES_IMAGE_MODEL")
        .unwrap_or_else(|_| "openai/gpt-5.2".to_string());
    let client = OpenRouterResponsesClient::new(api_key, &model, None, None);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("OpenRouter Responses image generation should succeed");
    assert_reads_letter_l("openrouter-responses", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("openrouter-responses-stream", &model, &streamed);
}

#[cfg(feature = "google-adc")]
#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and GOOGLE_APPLICATION_CREDENTIALS"]
async fn live_google_anthropic_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let model = std::env::var("CHEVALIER_LIVE_GOOGLE_ANTHROPIC_IMAGE_MODEL")
        .unwrap_or_else(|_| "claude-3-5-sonnet-v2@20241022".to_string());
    let region = std::env::var("CHEVALIER_LIVE_GOOGLE_ANTHROPIC_REGION")
        .unwrap_or_else(|_| "us-east5".to_string());
    let client = chevalier_agentic::providers::GoogleAnthropicClient::from_adc(&model, region);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("Google Anthropic image generation should succeed");
    assert_reads_letter_l("google-anthropic", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("google-anthropic-stream", &model, &streamed);
}

#[cfg(feature = "bedrock")]
#[tokio::test]
#[ignore = "Requires CHEVALIER_LIVE_IMAGE_TESTS=1 and AWS Bedrock credentials"]
async fn live_bedrock_claude_image_input_non_streaming_and_streaming() {
    if !live_image_tests_enabled() {
        eprintln!("Skipping: CHEVALIER_LIVE_IMAGE_TESTS not enabled");
        return;
    }

    let model = std::env::var("CHEVALIER_LIVE_BEDROCK_IMAGE_MODEL")
        .unwrap_or_else(|_| "anthropic.claude-3-5-haiku-20241022-v1:0".to_string());
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
        .ok();
    let client = chevalier_agentic::providers::BedrockClient::new(&model, region);
    let messages = letter_l_message();
    let config = GenerationConfig::new(&model).with_max_tokens(64);

    let response = client
        .get_generation(&messages, &config)
        .await
        .expect("Bedrock Claude image generation should succeed");
    assert_reads_letter_l("bedrock-claude", &model, &response.content);

    let streamed = stream_to_text(&client, &messages, &model).await;
    assert_reads_letter_l("bedrock-claude-stream", &model, &streamed);
}
