# Reson

**Agents are just functions.** Production-grade LLM agent framework built in Rust with Python bindings.

## What is Reson?

Reson helps you build AI agents that *actually work*. Instead of writing complex prompt chains or struggling with output parsing, you define your agent as a regular function - Reson handles the rest.

```python
from reson import agentic, Runtime
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

@agentic(model="anthropic:claude-3-5-sonnet")
async def extract_person(text: str, runtime: Runtime) -> Person:
    """Extract person information from text."""
    return await runtime.run(prompt=f"Extract from: {text}", output_type=Person)

person = await extract_person("Alice is 30 years old")
print(f"{person.name}, {person.age}")  # Alice, 30
```

No special state machines. No graph DSLs. Just functions.

## Monorepo Structure

Reson is a **Rust core** with **Python bindings**:

```
reson/
├── reson-rust/     # Rust library (reson-agentic crate)
├── reson-py/       # Python bindings via PyO3
└── docs/           # Provider documentation
```

| Component | Package | Install | Use Case |
|-----------|---------|---------|----------|
| **Rust** | `reson-agentic` | `cargo add reson-agentic` | Full feature set, maximum performance, production services |
| **Python** | `reson` | `pip install reson` | Rapid development, prototyping (subset of Rust features) |

The Python package is a compiled Rust extension via PyO3. Rust is the primary implementation; Python bindings expose core functionality for convenience.

## Installation

### Python

```bash
pip install reson
```

### Rust

```toml
[dependencies]
reson-agentic = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Quick Start

### Python

```python
import asyncio
from reson import agentic, Runtime

@agentic(model="openrouter:openai/gpt-4o")
async def summarize(text: str, runtime: Runtime) -> str:
    """Summarize text concisely."""
    return await runtime.run(prompt=f"Summarize: {text}")

result = asyncio.run(summarize("Long article text here..."))
print(result)
```

### Rust

```rust
use reson_agentic::prelude::*;

#[agentic(model = "openrouter:openai/gpt-4o")]
async fn summarize(text: String, runtime: Runtime) -> Result<serde_json::Value> {
    runtime.run(
        Some(&format!("Summarize: {}", text)),
        None, None, None, None, None, None, None, None
    ).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let result = summarize("Long article text here...".to_string()).await?;
    println!("{}", result);
    Ok(())
}
```

## Providers

Connect to any major LLM provider:

| Provider | Model Format | Example |
|----------|--------------|---------|
| Anthropic | `anthropic:model` | `anthropic:claude-3-5-sonnet-20241022` |
| OpenAI | `openai:model` | `openai:gpt-4o` |
| Google Gemini | `google-gemini:model` | `google-gemini:gemini-2.0-flash` |
| OpenRouter | `openrouter:provider/model` | `openrouter:anthropic/claude-sonnet-4` |
| AWS Bedrock | `bedrock:model-id` | `bedrock:anthropic.claude-3-sonnet-20240229-v1:0` |
| Vertex AI | `google-anthropic:model` | `google-anthropic:claude-3-opus@20240514` |

API keys are read from environment variables:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `GOOGLE_GEMINI_API_KEY`
- `OPENROUTER_API_KEY`
- AWS credentials for Bedrock
- Google ADC for Vertex AI

## Features

### Tool Calling

Let agents call functions:

```python
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Sunny, 22C in {city}"

@agentic(model="anthropic:claude-3-5-sonnet")
async def assistant(query: str, runtime: Runtime) -> str:
    runtime.tool(get_weather)
    return await runtime.run(prompt=query)

result = await assistant("What's the weather in Paris?")
# Agent calls get_weather("Paris") and responds with the result
```

### Structured Outputs

Return typed data with Pydantic:

```python
from typing import List
from pydantic import BaseModel

class Task(BaseModel):
    title: str
    priority: int
    done: bool = False

@agentic(model="anthropic:claude-3-5-sonnet")
async def extract_tasks(notes: str, runtime: Runtime) -> List[Task]:
    return await runtime.run(prompt=notes, output_type=List[Task])
```

### Streaming

Get responses as they arrive:

```python
@agentic(model="anthropic:claude-3-5-sonnet")
async def write_story(topic: str, runtime: Runtime) -> str:
    story = ""
    async for chunk_type, chunk in runtime.run_stream(prompt=f"Write about {topic}"):
        if chunk_type == "content":
            print(chunk, end="", flush=True)
            story += chunk
    return story
```

### Generator Functions

Yield intermediate results:

```python
from typing import AsyncGenerator
from reson import agentic_generator, Runtime

@agentic_generator(model="anthropic:claude-3-5-sonnet")
async def process_batch(items: list, runtime: Runtime) -> AsyncGenerator[dict, None]:
    for i, item in enumerate(items):
        result = await runtime.run(prompt=f"Process: {item}")
        yield {"index": i, "result": result}

async for result in process_batch(my_items):
    print(f"Processed: {result}")
```

### Video/Media Upload (Google Gemini)

Upload and analyze videos using Google's File API:

```rust
use reson_agentic::providers::{GoogleGenAIClient, FileState};

let client = GoogleGenAIClient::new(api_key, "gemini-2.0-flash");
let uploaded = client.upload_file(&video_bytes, "video/mp4", Some("my-video")).await?;

// Wait for processing, then use uploaded.uri in multimodal messages
```

See [reson-rust/examples/video_upload.rs](reson-rust/examples/video_upload.rs) for a full example.

## Feature Comparison

We work towards feature parity between Rust and Python, though some features may not yet be exposed in Python bindings.

| Feature | Python | Rust |
|---------|--------|------|
| Decorator/Macro API | `@agentic` | `#[agentic]` |
| Tool Registration | `runtime.tool(fn)` | `#[derive(Tool)]` + `register_tool_with_schema` |
| Streaming | `async for` | `futures::Stream` |
| Storage | Memory | Memory, Redis, PostgreSQL |
| Templates | Minijinja | Minijinja (feature-gated) |
| Media Upload | — | Google File API |
| Performance | 1x (Rust under hood) | 10-100x for parsing |

## Documentation

- **Python Guide**: [reson-py/README.md](reson-py/README.md)
- **Rust Guide**: [reson-rust/README.md](reson-rust/README.md)
- **Provider Docs**: [docs/](docs/)

## Examples

### Python

```bash
# Run integration tests as examples
cd reson
pytest integration_tests/test_simple_tools.py -v
```

### Rust

```bash
cd reson-rust
cargo run --example video_upload -- path/to/video.mp4
cargo run --example simple_tools
```

## Why Reson?

### vs LangChain
- No complex abstractions or chains
- Standard Python control flow
- Type-safe outputs by default

### vs Raw SDKs
- Unified API across providers
- Automatic tool calling
- Built-in retries and error handling

### vs Other Agent Frameworks
- Functions, not graphs
- Compile-time validation (Rust)
- Production-tested at [bismuth.sh](https://bismuth.sh)

## Architecture

```
Your Code
    ↓
@agentic / #[agentic] (decorator/macro)
    ↓
Runtime (tool registration, context, streaming)
    ↓
Provider Client (Anthropic, OpenAI, Google, etc.)
    ↓
LLM API
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache-2.0
