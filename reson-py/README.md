# reson

Python bindings for the reson LLM agent framework. Built on a Rust core via PyO3 for maximum performance with Python ergonomics.

## Installation

```bash
pip install reson
```

## Quick Start

```python
import asyncio
from reson import agentic, Runtime

@agentic(model="anthropic:claude-3-5-sonnet")
async def summarize(text: str, runtime: Runtime) -> str:
    """Summarize text concisely."""
    return await runtime.run(prompt=f"Summarize: {text}")

result = asyncio.run(summarize("Long article text here..."))
print(result)
```

## The `@agentic` Decorator

The `@agentic` decorator transforms an async function into an agent. It:
1. Creates a `Runtime` automatically and injects it as the last parameter
2. Configures the model from the decorator argument
3. Handles tool calling loops automatically

```python
from reson import agentic, Runtime

@agentic(model="openrouter:openai/gpt-4o")
async def my_agent(query: str, runtime: Runtime) -> str:
    """Your agent's docstring becomes its system context."""
    return await runtime.run(prompt=query)

# Call without passing runtime - it's injected automatically
result = await my_agent("Hello!")
```

## Tool Registration

Register Python functions as tools the agent can call:

```python
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@agentic(model="anthropic:claude-3-5-sonnet")
async def assistant(question: str, runtime: Runtime) -> str:
    runtime.tool(get_weather)
    runtime.tool(search_web)
    return await runtime.run(prompt=question)

# Agent will call tools as needed
result = await assistant("What's the weather in Tokyo?")
```

Tool schemas are generated automatically from:
- Function name → tool name
- Docstring → tool description
- Type hints → parameter types
- Default values → optional parameters

## Structured Outputs

Return typed data using Pydantic models:

```python
from typing import List
from pydantic import BaseModel

class Task(BaseModel):
    title: str
    priority: int
    done: bool = False

class TaskList(BaseModel):
    tasks: List[Task]

@agentic(model="anthropic:claude-3-5-sonnet")
async def extract_tasks(notes: str, runtime: Runtime) -> TaskList:
    """Extract tasks from meeting notes."""
    return await runtime.run(prompt=notes, output_type=TaskList)

result = await extract_tasks("Need to fix the bug by Friday...")
for task in result.tasks:
    print(f"[{'x' if task.done else ' '}] {task.title} (P{task.priority})")
```

## Streaming

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

Chunk types include:
- `"content"` - Text content
- `"reasoning"` - Model's thinking/reasoning (when available)
- `"tool_call"` - Tool invocation
- `"tool_result"` - Tool execution result

## Generator Functions

Yield intermediate results with `@agentic_generator`:

```python
from typing import AsyncGenerator
from reson import agentic_generator, Runtime

@agentic_generator(model="anthropic:claude-3-5-sonnet")
async def process_batch(items: list, runtime: Runtime) -> AsyncGenerator[dict, None]:
    for i, item in enumerate(items):
        result = await runtime.run(prompt=f"Process: {item}")
        yield {"index": i, "result": result}

async for result in process_batch(["item1", "item2", "item3"]):
    print(f"Processed: {result}")
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

## Runtime API

The `Runtime` object provides:

```python
# Tool registration
runtime.tool(func)                    # Register a Python function as a tool

# Execution
await runtime.run(                    # Run the agent
    prompt="...",                     # User prompt
    system="...",                     # System instructions (optional)
    output_type=MyModel,              # Pydantic model for structured output (optional)
    temperature=0.7,                  # Sampling temperature (optional)
    max_tokens=1000,                  # Max response tokens (optional)
)

# Streaming
async for chunk_type, chunk in runtime.run_stream(prompt="..."):
    ...
```

## Types

Import commonly used types:

```python
from reson.types import (
    ChatMessage,      # A message in a conversation
    ChatRole,         # user, assistant, system
    ToolCall,         # A tool invocation by the model
    ToolResult,       # Result of executing a tool
    ReasoningSegment, # Model's reasoning/thinking
)
```

## Direct Client Usage

For lower-level access, use inference clients directly:

```python
from reson.services.inference_clients import InferenceClient, InferenceProvider
from reson.types import ChatMessage, ChatRole

client = InferenceClient(
    provider=InferenceProvider.Anthropic,
    model="claude-3-5-sonnet-20241022",
    api_key="..."  # or use environment variable
)

messages = [ChatMessage(role=ChatRole.User, content="Hello!")]
response = await client.generate(messages)
print(response.content)
```

## Development

This package is built from Rust source using maturin:

```bash
cd reson-py
pip install maturin
maturin develop  # Build and install locally
```

Run tests:
```bash
pytest integration_tests/ -v
```

## License

Apache-2.0
