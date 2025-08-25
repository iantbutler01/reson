# Native Tool Calling

Native tool calling enables you to use provider-native APIs for improved performance and reduced token overhead.

## Basic Usage

Add `native_tools=True` to your decorator:

```python
from reson import agentic, Runtime
from reson.types import Deserializable

class WeatherQuery(Deserializable):
    location: str
    units: str = "celsius"

def get_weather(query: WeatherQuery) -> str:
    return f"Weather in {query.location}: 22°{query.units[0].upper()}, partly cloudy"

@agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
async def weather_assistant(query: str, runtime: Runtime) -> str:
    runtime.tool(get_weather)
    return await runtime.run(prompt=query)

# Usage
result = await weather_assistant("What's the weather in Paris?")
```

## Multi-turn Conversations

Handle tool calls and continue conversations:

```python
@agentic(model="openrouter:anthropic/claude-sonnet-4", native_tools=True)
async def calculation_agent(query: str, runtime: Runtime) -> str:
    runtime.tool(add_numbers)
    runtime.tool(multiply_numbers)
    
    history = []
    result = await runtime.run(prompt=query)
    
    # Handle multiple tool calls
    while runtime.is_tool_call(result):
        # Execute the tool
        tool_result = await runtime.execute_tool(result)
        
        # Add result to conversation history
        tool_msg = runtime.create_tool_result_message(result, str(tool_result))
        history.append(tool_msg)
        
        # Continue conversation
        result = await runtime.run(
            prompt="Continue with the next step",
            history=history
        )
    
    return result  # Final text response
```

## Supported Providers

- `openrouter:provider/model` - OpenRouter with any provider
- `vertex-gemini:model` - Google Vertex AI Gemini models  
- `anthropic:model` - Anthropic Claude models
- `openai:model` - OpenAI GPT models

## Tool Functions

Tools can use either Deserializable objects or primitive types:

```python
# Option 1: Deserializable parameters (for complex data)
class SearchQuery(Deserializable):
    text: str
    category: str = "general"
    max_results: int = 5

def search_database(query: SearchQuery) -> str:
    """Search a database with the given parameters."""
    return f"Found {query.max_results} results for '{query.text}' in {query.category}"

# Option 2: Primitive parameters (simpler tools)
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def get_weather(location: str, units: str = "celsius") -> str:
    """Get weather for a location."""
    return f"Weather in {location}: 22°{units[0].upper()}, partly cloudy"
```

Both patterns work - use Deserializable for complex data structures, primitives for simple tools.

## Key Methods

- `runtime.tool(func)` - Register a tool function
- `runtime.run(prompt="...")` - Execute LLM call  
- `runtime.is_tool_call(result)` - Check if result is a tool call
- `runtime.execute_tool(result)` - Execute a tool call
- `runtime.create_tool_result_message(result, output)` - Create conversation message

## When to Use

Use native tool calling when you want:
- Better performance with provider-native APIs
- Reduced token usage compared to XML approach
- Access to provider-specific optimizations

The traditional XML approach continues to work unchanged if you prefer it.
