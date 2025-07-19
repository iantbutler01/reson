# Reson

## What is Reson?

Reson helps you build AI agents that *actually work*. Instead of writing complex prompt chains or struggling with output parsing, you just define Python types for your outputs and reson handles the rest.

Most AI frameworks get you 80% of the way there with nice demos, but fall apart when it's time to deploy to production. Reson solves that last crucial 20% because it's built on battle-tested implementation that's been running in production for over a year at [bismuth.sh](https://www.bismuth.sh)

## Getting Started

```bash
pip install reson
```

That's it! No complex dependencies or setup required.

## The Basics

Here's how simple it is to use reson:

```python
from typing import List
from pydantic import BaseModel
from reson.reson import agentic, Runtime

class Person(BaseModel):
    name: str
    age: int
    skills: List[str] = []

@agentic(model="openrouter:openai/gpt-4o")
async def extract_people(text: str, runtime: Runtime) -> List[Person]:
    """
    Extract people information from the given text.
    Return a list of Person objects with names, ages, and skills.
    """
    # API keys for providers like OpenRouter are typically configured
    # via environment variables (e.g., OPENROUTER_KEY).
    return await runtime.run(prompt=f"Extract people from: {text}", output_type=List[Person])

# Now you can just use it
people = await extract_people("John is 30 and knows Python. Sarah is 28 and knows JavaScript.")
for person in people:
    print(f"{person.name}, {person.age}: {', '.join(person.skills)}")
```

No need to worry about parsing JSON, handling errors, or crafting the perfect prompt. Reson uses the type hints and docstring to guide the model in generating properly structured outputs.

## Cool Features

### Type-Driven Responses

Just define your output types using standard Python classes or scalars, Pydantic models or `Deserializable` classes (powered by the `gasp` library) for complex streaming control, and Reson will make sure the LLM generates outputs that match that structure.

```python
from pydantic import BaseModel  # Option 1: Pydantic models
from reson.types import Deserializable  # Option 2: Custom deserializable classes (from gasp)

class PersonPydantic(BaseModel):
    name: str
    age: int

class PersonDeserializable(Deserializable):
    name: str
    age: int

# Both approaches work with reson's type system
```

### Multiple LLM Providers

Connect to various LLM providers with a simple string. API keys are generally expected to be in environment variables (e.g., `ANTHROPIC_KEY`, `OPENAI_API_KEY`, `GOOGLE_GEMINI_API_KEY`, `OPENROUTER_KEY`, AWS credentials for Bedrock/Vertex).

```python
# Examples of model strings:
# @agentic(model="openrouter:openai/gpt-4o")       # OpenRouter
# @agentic(model="anthropic:claude-3-opus-20240229") # Anthropic (Direct)
# @agentic(model="bedrock:anthropic.claude-3-sonnet-20240229-v1:0") # AWS Bedrock
# @agentic(model="google-gemini:gemini-1.5-pro-latest") # Google Gemini API
# @agentic(model="openai:gpt-4o")                  # OpenAI API
# @agentic(model="vertexai:gemini-1.5-flash-001")    # Google Vertex AI (Direct)
# @agentic(model="google-anthropic:claude-3-opus-20240229") # Anthropic on Vertex
```

### Tool Registration

Let your agents call functions to get more information or perform actions:

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Code to fetch weather...
    return f"Sunny and 72°F in {city}"

@agentic(model="openrouter:openai/gpt-4o")
async def plan_trip(destination: str, weather_tool: Callable, runtime: Runtime) -> str:
    """Plan a trip given a destination and weather information."""
    # The weather_tool (which is `get_weather` passed in) is automatically registered.
    return await runtime.run(prompt=f"Plan a trip to {destination}")
```

### Streaming Support

Get results as they come in:

```python
@agentic(model="anthropic:claude-3-opus-20240229")
async def write_story(topic: str, runtime: Runtime) -> str:
    """Write a story about the given topic."""
    result = ""
    async for chunk in runtime.run_stream(prompt=f"Write a story about {topic}"):
        result += chunk
        print(chunk, end="", flush=True)  # Show the story as it's being written
    return result
```

### Generator Yield Semantics

Create agentic functions that yield results incrementally:

```python
from typing import AsyncGenerator, Dict, List

@agentic(model="openrouter:openai/gpt-4o")
async def process_items(items: List[str], runtime: Runtime) -> AsyncGenerator[Dict, None]:
    """Process items one by one and yield results as they're ready."""
    for i, item in enumerate(items):
        result = await runtime.run(prompt=f"Process this item: {item}")
        yield {"index": i, "item": item, "result": result}

# Usage
# async for result in process_items(my_items):
#     print(f"Item {result['index']}: {result['result']}")
```

This is great for:
- Processing large datasets piece by piece
- Showing progress during long-running tasks
- Streaming results to users in real-time
- Mixing status updates with actual results

## Extensible Architecture

Reson is built with a modular design:

*   **Inference Clients**: Each LLM provider (OpenAI, Anthropic, Bedrock, etc.) has its own `InferenceClient` implementation. The `TracingInferenceClient` wraps these to add features like caching, cost tracking, request/response tracing, and fallback logic.
*   **Output Parsers**: The `OutputParser` interface defines how LLM responses are converted into Python types.
    *   `TypeParser`: The default parser, uses `gasp` for robust parsing based on Python type hints, Pydantic models, and `Deserializable` classes.
*   **Stores**: For persisting context or trace data, with backends like memory, Redis, and PostgreSQL.

This architecture allows for easier extension and maintenance.

## Gradual complexity when you need it

### Jinja2 Templating in Docstrings

Your docstrings can use Jinja2 template syntax for dynamic prompt generation:

```python
@agentic(model="openrouter:openai/gpt-4o")
async def extract_people_from_list(people_list: List[str], runtime: Runtime) -> List[Person]:
    """
    Extract people information from the following descriptions:
    
    {% for person_desc in people_list %}
    Person {{ loop.index }}: {{ person_desc }}
    {% endfor %}
    
    Return a list of Person objects with names, ages, and skills.
    
    {{return_type}}
    """
    return await runtime.run()
```

### Detailed Type Guidance with {{return_type}}

The special `{{return_type}}` placeholder in your docstring or prompt automatically inserts type information to guide the LLM:

```python
class Company(BaseModel):
    name: str
    industry: str
    employee_count: int

@agentic(model="openrouter:openai/gpt-4o")
async def extract_company(text: str, runtime: Runtime) -> Company:
    """
    Extract company information from this text.
    
    {{return_type}}
    """
    return await runtime.run(prompt=f"Extract from: {text}") # Prompt can be explicit too
```

You can also use it in custom prompts passed to `runtime.run()`:

```python
# data = "Some data string"
# prompt = f"""
# Analyze this data: {data}

# {{{{return_type}}}}  # Double braces for f-strings if {{return_type}} is not a var
# """
# result = await runtime.run(prompt=prompt, output_type=Company)
```

### Storage Backends

Configure different storage backends for context and tracing:

```python
from reson.stores import MemoryStoreConfig, RedisStoreConfig, PostgresStoreConfig

# In-memory storage (default)
# @agentic(model="openrouter:openai/gpt-4o")
# async def with_memory_store(runtime: Runtime): ...

# Redis for persistent storage
# @agentic(
#     model="openrouter:openai/gpt-4o", 
#     store_cfg=RedisStoreConfig(
#         host="redis.example.com",
#         port=6379,
#         password="secret"
#     )
# )
# async def with_redis_store(runtime: Runtime): ...

# PostgreSQL with JSONB document storage
# @agentic(
#     model="openrouter:openai/gpt-4o",
#     store_cfg=PostgresStoreConfig(
#         dsn="postgresql://user:pass@db.example.com/db",
#         table="agent_data",   # Table will be created if it doesn't exist
#         column="data"         # JSONB column that stores all context data
#     )
# )
# async def with_postgres_store(runtime: Runtime): ...
```

The PostgreSQL store uses a JSONB column for efficient document storage, allowing you to leverage PostgreSQL's reliability and query capabilities while storing context data as JSON documents.

### Tool Binding Control

Control exactly which tools are accessible to the LLM:

```python
# By default, all callable parameters become tools
# @agentic(model="openrouter:openai/gpt-4o")
# async def default_behavior(search_database, update_records, runtime: Runtime):
#     # Both functions are automatically available to the LLM
#     return await runtime.run()

# Disable autobind for precise control
# @agentic(model="openrouter:openai/gpt-4o", autobind=False)
# async def controlled_access(search_database, update_records, runtime: Runtime):
#     # Only explicitly registered tools are available
#     runtime.tool(search_database, name="search") # Manually register 'search_database' as 'search'
#     return await runtime.run()
```

### Context Management API

Store and retrieve state between function calls using the configured store:

```python
@agentic(model="openrouter:openai/gpt-4o")
async def conversational_agent(message: str, runtime: Runtime) -> str:
    # Get conversation history (defaults to empty list if not found)
    history = await runtime.context.get("conversation", [])
    history.append(f"User: {message}")
    
    # Use the history in your prompt
    prompt_with_history = "\n".join(history) + "\nAssistant:"
    response = await runtime.run(prompt=prompt_with_history)
    
    # Update the history for next time
    history.append(f"Assistant: {response}")
    await runtime.context.set("conversation", history)
    
    return response
```

## Examples

Check out the `example.py`, `example_generator.py`, `example_tools.py`, and `example_postgres.py` files for complete working examples showing how to use the various features of reson in realistic scenarios.

### Enterprise-Grade Features Included:

- **OpenTelemetry Integration**: Built-in OTEL tracing gives you visibility into every aspect of your AI system's performance.
  
- **Robust Error Handling**: The `TracingInferenceClient` implements automatic retries and fallback mechanisms (if a fallback client is configured) to ensure your AI components are resilient against API outages and rate limits.
  
- **Comprehensive Tracing**: Every LLM request can be traced. Set the `RESON_TRACE` environment variable to a local directory path or an S3 URI (e.g., `s3:my-trace-bucket`) to log inputs, outputs, and costs.
  
- **Cost Monitoring**: The `TracingInferenceClient` automatically tracks token usage and estimates costs for supported models, helping you stay on budget.
  
- **Multi-Provider Reliability**: Automatic fallback between primary and secondary LLM providers (when configured via `TracingInferenceClient`) enhances reliability.

### Why This Matters

When your AI system is customer-facing or mission-critical, these features aren't optional extras - they're essential. Reson gives you production readiness out of the box, so you can focus on building great AI applications instead of reinventing infrastructure wheels.

## Leveraging Python's Native Patterns

The AI framework world is full of custom abstractions, DSLs, and execution graphs that force you to think differently. Reson takes the opposite approach - it should feel like you're just writing normal Python code.

### Native Control Flow, No Special Rules

With reson, you use standard Python control flow exactly as you'd expect:

```python
# class Analysis(BaseModel): ... # Define Analysis model
# @agentic(model="openrouter:openai/gpt-4o")
# async def analyze_data(data: List[Dict], runtime: Runtime) -> Analysis:
#     # Normal conditionals
#     if len(data) == 0:
#         return Analysis(status="empty", results=[])
        
#     # Regular Python loops
#     results_data = []
#     for item in data:
#         try:
#             # Standard exception handling
#             result = await runtime.run(prompt=f"Analyze this: {item}")
#             results_data.append(result)
#         except Exception as e:
#             # Handle errors just like regular Python
#             print(f"Error processing {item}: {e}")
            
#     # Regular function calls mix with agentic functions
#     # summary = await summarize_results(results_data, runtime) # Assuming summarize_results is another agentic fn
    
#     # Return normal Python objects
#     return Analysis(status="complete", results=results_data, summary="summary_placeholder")
```

No special dispatch functions, no state machines to manage, no graph nodes to connect - just write normal Python.

### Regular Function Composition

Since agentic functions are just async functions, you compose them like any other:

```python
# class Category(BaseModel): ...
# class Entity(BaseModel): ...
# class DocumentAnalysis(BaseModel): ...
# class Document(BaseModel): text: str

# # Split complex tasks into smaller, reusable functions
# @agentic(model="openrouter:openai/gpt-4o") 
# async def classify_text(text: str, runtime: Runtime) -> Category:
#     return await runtime.run(prompt=f"Classify this text: {text}")

# @agentic(model="openrouter:openai/gpt-4o")
# async def extract_entities_from_text(text: str, runtime: Runtime) -> List[Entity]:
#     return await runtime.run(prompt=f"Extract entities from: {text}")

# # Compose them naturally
# async def process_document(doc: Document) -> DocumentAnalysis:
#     category = await classify_text(doc.text) # Assuming classify_text is defined
#     entities = await extract_entities_from_text(doc.text) # Assuming extract_entities_from_text is defined
#     return DocumentAnalysis(doc=doc, category=category, entities=entities)
```

### Generator semantics for pause/resume

The new generator support means you can use Python's native `async for` to handle streaming or paused execution:

```python
# class Result(BaseModel): item: str; processed: Any
# @agentic(model="openrouter:openai/gpt-4o")
# async def process_with_pause(data: List[str], runtime: Runtime) -> AsyncGenerator[Result, None]:
#     for item in data:
#         # Process
#         processed_result = await runtime.run(prompt=f"Process: {item}")
#         # Yield (pause execution, return control to caller)
#         yield Result(item=item, processed=processed_result)
#         # Execution resumes here when the caller requests the next item
    
# # Use standard Python iteration semantics
# # async for result_item in process_with_pause(data):
# #     # Do something with each result
# #     # Maybe save to database, or wait for user confirmation
# #     await save_result(result_item) # Assuming save_result is defined
```

No need for complex state management or custom resumption tokens - Python's generators handle the pause/resume pattern natively.

### Comparison with Other Frameworks

| Feature | Reson | Other Frameworks |
|---------|-------|------------------|
| Control flow | Standard Python if/else, loops, try/except | Custom graph-based flows, special state machines, or DSLs |
| Function composition | Regular function calls and async/await | Special operators, nodes, chains, or pipelines |
| Error handling | Standard try/except blocks | Custom error handlers, fallback nodes, or retry policies |
| State management | Python variables and objects | Special state containers, contexts, or memory objects |
| Pause/resume | Native generator semantics | Custom callback patterns or state serialization |

### Simplifying AI Development

Just as we moved from assembly to high-level languages, AI development should move from prompt engineering to semantic programming. Reson pushes toward a future where AI capabilities are just regular function calls in your codebase, not special constructs that need special handling.

## Inspired by 12-Factor Agents

Reson is built on principles aligned with [12-Factor Agents](https://github.com/humanlayer/12-factor-agents), an excellent set of guidelines for building production-ready AI applications developed by @dexhorthy and the @humanlayer team.

We strongly agree with the 12-Factor approach and have designed reson to embody these principles through Python-native patterns:

- Natural language inputs are converted to structured function calls
- Developers have full control over prompts and context windows
- Tools are represented as typed outputs with clear schemas
- Execution state uses standard Python data structures
- Launch, pause and resume operations use familiar generator patterns
- Control flow follows standard Python semantics
- Error handling is comprehensive and integrated
- Components are small, focused, and composable
- Everything is designed to be stateless and predictable

Reson's goal is to provide one take on these principles with minimal abstractions, letting you build production-grade AI applications using familiar Python patterns. 

We're grateful to @dexhorthy and the @humanlayer team for articulating these principles so clearly, and we encourage everyone to check out their excellent framework!

## Tips & Tricks

1. **Use the docstring**: The docstring of your agentic function is used as the default prompt for the LLM. Make it clear and descriptive. It can also include Jinja2 templating.

2. **Type annotations matter**: The more specific your type annotations (Pydantic, `Deserializable`, standard types), the better the LLM will structure its outputs. Use `{{return_type}}` in your prompt/docstring to automatically include schema information.

3. **Generator semantics for progress reporting**: Use `agentic_generator` and `async for ... yield` when you need to show progress on long-running tasks or stream partial results:

   ```python
    @agentic_generator(model="openrouter:openai/gpt-4o") # Use agentic_generator for async generators
    async def analyze_documents_stream(docs: List[str], runtime: Runtime) -> AsyncGenerator[Dict, None]:
        total = len(docs)
        for i, doc in enumerate(docs):
           # Yield progress update first
            yield {"type": "progress", "done": i, "total": total}
           
            # Process document
            result = await runtime.run(prompt=f"Analyze this document: {doc}")
           
            # Yield actual result
            yield {"type": "result", "document": doc, "analysis": result}
   ```

## Why the Name?

Because good AI interactions should "resonate" with users and be in harmony with the developer's intent. Plus, it's short and hopefully easy to remember!
(Not to mention the domain was available.)
