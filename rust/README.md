# chevalier

Rust runtime for Chevalier's model/provider and tool-calling surface.

Use this crate when you want direct Rust access to:

- provider clients for Anthropic, OpenAI, Google Gemini, OpenRouter, Bedrock, and Vertex/Google Anthropic
- typed tool schemas with `#[derive(Tool)]`
- agent wrappers with `#[agentic]`
- streaming model events, reasoning, tool calls, tool results, and usage
- optional MCP, sandbox, and VFS integration
- request tracing and cost accounting

## Install

```toml
[dependencies]
chevalier = "0.7.0"
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Enable only what you need:

```toml
chevalier = { version = "0.7.0", features = ["mcp", "vfs"] }
```

## Basic Provider Call

```rust
use chevalier::providers::{GenerationConfig, GoogleGenAIClient, InferenceClient};
use chevalier::types::ChatMessage;
use chevalier::utils::ConversationMessage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = GoogleGenAIClient::new(
        std::env::var("GOOGLE_GEMINI_API_KEY")?,
        "gemini-2.0-flash",
    );

    let messages = vec![ConversationMessage::Chat(ChatMessage::user("Hello"))];
    let config = GenerationConfig::new("gemini-2.0-flash");
    let response = client.get_generation(&messages, &config).await?;

    println!("{}", response.content);
    Ok(())
}
```

## Tools

`#[derive(Tool)]` turns Rust structs into model-callable JSON schemas:

```rust
use chevalier::Tool;
use serde::{Deserialize, Serialize};

#[derive(Tool, Serialize, Deserialize, Debug)]
struct WebSearch {
    /// Query to search for.
    query: String,
    /// Maximum number of results.
    max_results: Option<u32>,
}

let name = WebSearch::tool_name();
let description = WebSearch::description();
let schema = WebSearch::schema();
```

## Optional Feature Flags

| Feature | Enables |
| --- | --- |
| `full` | Bedrock, templating, telemetry, Google ADC, MCP |
| `bedrock` | AWS Bedrock provider |
| `templating` | Minijinja prompt templates |
| `telemetry` | OpenTelemetry hooks |
| `google-adc` | Google Application Default Credentials |
| `mcp` | `chevalier-mcp` client/server support |
| `mcp-apps` | MCP Apps UI resources |
| `sandbox` | Remote sandbox facade client |
| `sandbox-local` | Sandbox client plus local host support |
| `sandbox-distributed-control` | etcd/NATS distributed sandbox routing |
| `sandbox-vfs-server` | Sandbox VFS server routes |
| `vfs` | `chevalier-vfs` integration |

## Development

```bash
cargo test --all-features
cargo clippy --all-features -- -D warnings
cargo fmt --all -- --check
```

Run provider integration tests only with the relevant live API keys set.

## License

Apache-2.0
