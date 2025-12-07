# reson-agentic

Agents are just functions - production-grade LLM agent framework for Rust.

## Features

- **Multi-provider support**: Anthropic, OpenAI, Google Gemini, OpenRouter, AWS Bedrock
- **Native tool calling** with structured outputs via `#[derive(Tool)]`
- **Agent macro** for ergonomic agent definitions with `#[agentic]`
- **Streaming responses** with reasoning/thinking support
- **Google File API** for video/large media uploads
- **Retry with exponential backoff**
- **Clone-friendly clients** for use in async contexts

## Installation

```toml
[dependencies]
reson-agentic = "0.1"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
```

## Quick Start

### Basic Chat

```rust
use reson_agentic::providers::{GoogleGenAIClient, GenerationConfig, InferenceClient};
use reson_agentic::types::ChatMessage;
use reson_agentic::utils::ConversationMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = GoogleGenAIClient::new("your-api-key", "gemini-2.0-flash");

    let messages = vec![
        ConversationMessage::Chat(ChatMessage::user("Hello!"))
    ];

    let config = GenerationConfig::new("gemini-2.0-flash");
    let response = client.get_generation(&messages, &config).await?;

    println!("{}", response.content);
    Ok(())
}
```

## Tool Definitions with `#[derive(Tool)]`

Define type-safe tools that automatically generate JSON schemas for LLM function calling:

```rust
use reson_agentic::Tool;
use serde::{Deserialize, Serialize};

/// Search the web for information
#[derive(Tool, Serialize, Deserialize, Debug)]
struct WebSearch {
    /// The search query to execute
    query: String,
    /// Maximum number of results to return
    max_results: Option<u32>,
}

/// Get weather for a location
#[derive(Tool, Serialize, Deserialize, Debug)]
struct GetWeather {
    /// City name or coordinates
    location: String,
    /// Temperature unit: "celsius" or "fahrenheit"
    unit: Option<String>,
}

// Access generated schema
let schema = WebSearch::schema();       // JSON Schema object
let name = WebSearch::tool_name();      // "web_search"
let desc = WebSearch::description();    // "Search the web for information"
```

The `#[derive(Tool)]` macro:
- Converts struct name to snake_case for the tool name
- Uses doc comments as descriptions (struct doc -> tool description, field docs -> parameter descriptions)
- Generates proper JSON Schema with types, required fields, and array items
- Supports `String`, `bool`, `i32`/`i64`/`u32`/`u64`, `f32`/`f64`, `Vec<T>`, and `Option<T>`

## Agent Functions with `#[agentic]`

The `#[agentic]` macro transforms an async function into an agent. It:
1. Creates a `Runtime` automatically and injects it into the function
2. Validates that `runtime.run()` or `runtime.run_stream()` is called
3. Configures the model from the macro attribute

```rust
use reson_agentic::agentic;
use reson_agentic::runtime::{Runtime, ToolFunction};
use reson_agentic::error::Result;

/// Analyze text and answer questions
#[agentic(model = "gemini:gemini-2.0-flash")]
async fn analyze_text(
    text: String,
    question: String,
    runtime: Runtime,  // Injected by macro - callers don't pass this
) -> Result<serde_json::Value> {
    // Register tools with the runtime
    runtime.register_tool_with_schema(
        WebSearch::tool_name(),
        WebSearch::description(),
        WebSearch::schema(),
        ToolFunction::Sync(Box::new(|args| {
            let query = args["query"].as_str().unwrap_or("");
            Ok(format!("Search results for: {}", query))
        })),
    ).await?;

    // Run the agent - runtime is mutable internally
    runtime.run(
        Some(&format!("Text: {}\n\nQuestion: {}", text, question)),
        Some("You are a helpful assistant. Use tools when needed."),
        None,  // history
        None,  // output_type
        None,  // temperature
        None,  // top_p
        None,  // max_tokens
        None,  // model override
        None,  // api_key override
    ).await
}

// Call the agent - runtime parameter is NOT passed by caller
let result = analyze_text(
    "The quick brown fox...".to_string(),
    "What animal is mentioned?".to_string(),
).await?;
```

## Video/Media Upload (Google Gemini)

Upload and analyze videos using Google's File API:

```rust
use reson_agentic::providers::{GoogleGenAIClient, FileState};
use reson_agentic::types::{ChatRole, MediaPart, MediaSource, MultimodalMessage};

let client = GoogleGenAIClient::new(api_key, "gemini-2.0-flash");

// Upload video
let video_bytes = std::fs::read("video.mp4")?;
let uploaded = client.upload_file(&video_bytes, "video/mp4", Some("my-video")).await?;

// Wait for processing (required for videos)
if uploaded.state == FileState::Processing {
    client.wait_for_file_processing(&uploaded.name, Some(120)).await?;
}

// Create multimodal message
let message = MultimodalMessage {
    role: ChatRole::User,
    parts: vec![
        MediaPart::Video {
            source: MediaSource::FileUri {
                uri: uploaded.uri.clone(),
                mime_type: Some("video/mp4".to_string()),
            },
            metadata: None,
        },
        MediaPart::Text { text: "Describe this video".to_string() },
    ],
    cache_marker: None,
};

// Clean up when done
client.delete_file(&uploaded.name).await?;
```

### Supported Media Types

| Type | Formats | Max Size |
|------|---------|----------|
| Video | MP4, MOV, AVI, WebM, MKV, FLV, 3GP | 2GB |
| Image | JPEG, PNG, GIF, WebP, HEIC | 20MB inline |
| Audio | MP3, WAV, FLAC, AAC, OGG, M4A | 2GB |
| Document | PDF, TXT, HTML, CSS, JS, etc. | Varies |

## Providers

| Provider | Client | Model Format |
|----------|--------|--------------|
| Google Gemini | `GoogleGenAIClient` | `gemini-2.0-flash` |
| Anthropic | `AnthropicClient` | `claude-sonnet-4-20250514` |
| OpenAI | `OAIClient` | `gpt-4o` |
| OpenRouter | `OpenRouterClient` | `anthropic/claude-sonnet-4` |
| AWS Bedrock | `BedrockClient` | `anthropic.claude-sonnet-4-20250514-v1:0` |
| Vertex AI (Claude)* | `GoogleAnthropicClient` | `claude-sonnet-4@20250514` |

*Requires `google-adc` feature: `reson-agentic = { version = "0.1", features = ["google-adc"] }`

All clients implement `Clone` for easy use in async contexts.

## Examples

See the [examples](./examples) directory:

- `video_upload.rs` - Video analysis with Google Gemini and `#[agentic]` macro
- `simple_tools.rs` - Basic tool registration and execution
- `tool_call_chain.rs` - Multi-turn tool calling
- `dynamic_tool_parsing.rs` - Type-safe tool parsing with `Deserializable`
- `templating_example.rs` - Prompt templates with minijinja
- `store_usage.rs` - Context storage patterns

Run examples with:
```bash
GOOGLE_GEMINI_API_KEY=your_key cargo run --example video_upload -- video.mp4
```

## Feature Flags

```toml
[dependencies]
reson-agentic = { version = "0.1", features = ["full"] }
```

| Feature | Description |
|---------|-------------|
| `full` | All features enabled |
| `storage` | Redis + SQLx storage backends |
| `bedrock` | AWS Bedrock support |
| `templating` | Minijinja prompt templates |
| `telemetry` | OpenTelemetry tracing |
| `google-adc` | Google Application Default Credentials (Vertex AI) |

## License

ApacheV2