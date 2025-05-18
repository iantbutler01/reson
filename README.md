# Reson

## What is Reson?

Reson helps you build AI agents that *actually work*. Instead of writing complex prompt chains or struggling with output parsing, you just define Python types for your outputs and reson handles the rest.

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
    return await runtime.run(prompt=f"Extract people from: {text}")

# Now you can just use it
people = await extract_people("John is 30 and knows Python. Sarah is 28 and knows JavaScript.")
for person in people:
    print(f"{person.name}, {person.age}: {', '.join(person.skills)}")
```

No need to worry about parsing JSON, handling errors, or crafting the perfect prompt. Reson uses the type hints and docstring to guide the model in generating properly structured outputs.

## Cool Features

### Type-Driven Responses

Just define your output types using Pydantic models, Deserializable classes, or basic Python type hints, and reson will make sure the LLM generates outputs that match that structure.

```python
from pydantic import BaseModel  # Option 1: Pydantic models
from reson.types import Deserializable  # Option 2: Custom deserializable classes

class PersonPydantic(BaseModel):
    name: str
    age: int

class PersonDeserializable(Deserializable):
    name: str
    age: int

# Both approaches work with reson's type system
```

### Multiple LLM Providers

Connect to various LLM providers with a simple string:

```python
@agentic(model="openrouter:openai/gpt-4o")  # OpenRouter
@agentic(model="anthropic:claude-3-7-sonnet-latest")  # Anthropic
@agentic(model="bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0")  # AWS Bedrock
@agentic(model="google-gemini:gemini-2.5-pro-preview-05-06")  # Google Gemini
```

### Tool Registration

Let your agents call functions to get more information or perform actions:

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Code to fetch weather...
    return f"Sunny and 72°F in {city}"

@agentic(model="openrouter:openai/gpt-4o")
async def plan_trip(destination: str, weather_tool, runtime: Runtime) -> str:
    """Plan a trip given a destination and weather information."""
    # The weather_tool is automatically registered as a tool
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

Now you can create agentic functions that yield results incrementally:

```python
from typing import AsyncGenerator, Dict, List

@agentic(model="openrouter:openai/gpt-4o")
async def process_items(items: List[str], runtime: Runtime) -> AsyncGenerator[Dict, None]:
    """Process items one by one and yield results as they're ready."""
    for i, item in enumerate(items):
        result = await runtime.run(prompt=f"Process this item: {item}")
        yield {"index": i, "item": item, "result": result}

# Usage
async for result in process_items(my_items):
    print(f"Item {result['index']}: {result['result']}")
```

This is great for:
- Processing large datasets piece by piece
- Showing progress during long-running tasks
- Streaming results to users in real-time
- Mixing status updates with actual results

### BAML Integration

Seamlessly integrate with BAML for enhanced prompt engineering:

```python
@agentic(model="anthropic:claude-3-opus-20240229")
async def extract_with_baml(runtime: Runtime):
    """Use BAML integration to extract information."""
    import baml_client as b
    request = b.ExtractPerson("John Doe is 30 years old...")
    return await runtime.run_with_baml(baml_request=request)
```

## Real-World Examples

Check out the `example.py` and `example_generator.py` files for complete working examples showing how to use the various features of reson in realistic scenarios.

## Production Ready, Not Just a Prototype

Most AI frameworks get you 80% of the way there with nice demos, but fall apart when it's time to deploy to production. Reson solves that last crucial 20% because it's built on battle-tested infrastructure that's already running in production environments.

### Enterprise-Grade Features Included:

- **OpenTelemetry Integration**: Built-in OTEL tracing gives you visibility into every aspect of your AI system's performance.
  
- **Robust Error Handling**: Automatic retries and fallback mechanisms ensure your AI components are resilient against API outages and rate limits.
  
- **Comprehensive Tracing**: Every LLM request is traceable, with inputs and outputs logged for debugging, auditing, and continuous improvement.
  
- **Cost Monitoring**: Automatic token counting and cost tracking across different models help you stay on budget.
  
- **Multi-Provider Reliability**: Automatic fallback between different LLM providers when one is unresponsive, with smart timing to return to primary providers.

### Why This Matters

When your AI system is customer-facing or mission-critical, these features aren't optional extras - they're essential. Reson gives you production readiness out of the box, so you can focus on building great AI applications instead of reinventing infrastructure wheels.

## Leveraging Python's Native Patterns

The AI framework world is full of custom abstractions, DSLs, and execution graphs that force you to think differently. Reson takes the opposite approach - it should feel like you're just writing normal Python code.

### Native Control Flow, No Special Rules

With reson, you use standard Python control flow exactly as you'd expect:

```python
@agentic(model="openrouter:openai/gpt-4o")
async def analyze_data(data: List[Dict], runtime: Runtime) -> Analysis:
    # Normal conditionals
    if len(data) == 0:
        return Analysis(status="empty", results=[])
        
    # Regular Python loops
    results = []
    for item in data:
        try:
            # Standard exception handling
            result = await runtime.run(prompt=f"Analyze this: {item}")
            results.append(result)
        except Exception as e:
            # Handle errors just like regular Python
            print(f"Error processing {item}: {e}")
            
    # Regular function calls mix with agentic functions
    summary = await summarize_results(results, runtime)
    
    # Return normal Python objects
    return Analysis(status="complete", results=results, summary=summary)
```

No special dispatch functions, no state machines to manage, no graph nodes to connect - just write normal Python.

### Regular Function Composition

Since agentic functions are just async functions, you compose them like any other:

```python
# Split complex tasks into smaller, reusable functions
@agentic(model="openrouter:openai/gpt-4o") 
async def classify_text(text: str, runtime: Runtime) -> Category:
    return await runtime.run(prompt=f"Classify this text: {text}")

@agentic(model="openrouter:openai/gpt-4o")
async def extract_entities(text: str, runtime: Runtime) -> List[Entity]:
    return await runtime.run(prompt=f"Extract entities from: {text}")

# Compose them naturally
async def process_document(doc: Document) -> DocumentAnalysis:
    category = await classify_text(doc.text)
    entities = await extract_entities(doc.text)
    return DocumentAnalysis(doc=doc, category=category, entities=entities)
```

### Generator semantics for pause/resume

The new generator support means you can use Python's native `async for` to handle streaming or paused execution:

```python
@agentic(model="openrouter:openai/gpt-4o")
async def process_with_pause(data: List[str], runtime: Runtime) -> AsyncGenerator[Result, None]:
    for item in data:
        # Process
        result = await runtime.run(prompt=f"Process: {item}")
        # Yield (pause execution, return control to caller)
        yield Result(item=item, processed=result)
        # Execution resumes here when the caller requests the next item
    
# Use standard Python iteration semantics
async for result in process_with_pause(data):
    # Do something with each result
    # Maybe save to database, or wait for user confirmation
    await save_result(result)
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

1. **Use the docstring**: The docstring of your agentic function is used as the default prompt for the LLM. Make it clear and descriptive.

2. **Type annotations matter**: The more specific your type annotations, the better the LLM will structure its outputs.

3. **Generator semantics for progress reporting**: Use generator semantics when you need to show progress on long-running tasks:

   ```python
   @agentic(model="openrouter:openai/gpt-4o")
   async def analyze_documents(docs: List[str], runtime: Runtime) -> AsyncGenerator[Dict, None]:
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
