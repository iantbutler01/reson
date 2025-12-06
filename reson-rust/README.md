# reson-agentic

Agents are just functions - production-grade LLM agent framework for Rust.

## Features

- Multi-provider support: Anthropic, OpenAI, Google Gemini, OpenRouter, AWS Bedrock
- Native tool calling with structured outputs
- Streaming responses with reasoning/thinking support
- Google File API for video/large media uploads
- Retry with exponential backoff
- Proc macros for ergonomic agent definitions

## Quick Start

```toml
[dependencies]
reson-agentic = "0.1"
tokio = { version = "1", features = ["full"] }
```

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

## Video Upload Example

```rust
use reson_agentic::providers::{GoogleGenAIClient, FileState};

let client = GoogleGenAIClient::new(api_key, "gemini-2.0-flash");

// Upload video
let video_bytes = std::fs::read("video.mp4")?;
let uploaded = client.upload_file(&video_bytes, "video/mp4", Some("my-video")).await?;

// Wait for processing
let ready = client.wait_for_file_processing(&uploaded.name, None).await?;

// Use ready.uri in your requests
```

## License

MIT
