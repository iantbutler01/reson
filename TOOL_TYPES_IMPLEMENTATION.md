# Tool Types Implementation Summary

## Overview

I've implemented Pattern 2 for tool types in the reson library. This allows tools (callables) to be automatically converted to typed classes based on the output type, enabling the LLM to choose between calling tools or returning the final result.

## Key Components Added

### 1. Type Detection Functions (in reson.py)

```python
def _is_pydantic_type(type_annotation) -> bool:
    """Check if a type is Pydantic-based."""
    
def _is_deserializable_type(type_annotation) -> bool:
    """Check if a type is Deserializable-based."""
```

### 2. Tool Class Generation

```python
def _create_pydantic_tool_model(func: Callable, name: str) -> Type:
    """Create a Pydantic model from a callable's signature using type()."""
    
def _create_deserializable_tool_class(func: Callable, name: str) -> Type:
    """Create a Deserializable class from a callable's signature using type()."""
```

These functions:
- Extract parameter annotations from the callable
- Handle default values
- Create dynamic classes with `_tool_func` and `_tool_name` attributes
- Use Python's `type()` function to create classes at runtime

### 3. Parameter Validation

```python
def _validate_callable_params(func: Callable, name: str) -> None:
    """Ensure all parameters are properly typed."""
```

This ensures that all tool parameters have type annotations, which is required for proper tool class generation.

### 4. Runtime Helper Methods

Added to the `Runtime` class:

```python
def is_tool_call(self, result: Any) -> bool:
    """Check if a result is a tool call."""
    
def get_tool_name(self, result: Any) -> Optional[str]:
    """Get the tool name from a tool call result."""
    
async def execute_tool(self, tool_result: Any) -> Any:
    """Execute a tool call result."""
```

These methods allow users to:
- Check if the LLM returned a tool call
- Get the tool name for logging/debugging
- Execute the tool with proper argument unpacking

### 5. Modified LLM Functions

Both `_call_llm` and `_call_llm_stream` were updated to:
- Detect when tools are provided with a typed output
- Generate tool classes based on the output type system (Pydantic or Deserializable)
- Create a Union type: `Union[Tool1, Tool2, ..., OutputType]`
- Pass this to the parser for proper type handling

## Usage Pattern

```python
@agentic(model="openrouter:openai/gpt-4o")
async def agent_with_tools(
    input: str,
    tool1: Callable,
    tool2: Callable,
    runtime: Runtime
) -> OutputType:
    """Agent that can use tools."""
    
    result = await runtime.run()
    
    # Handle tool calls
    while runtime.is_tool_call(result):
        tool_name = runtime.get_tool_name(result)
        tool_output = await runtime.execute_tool(result)
        
        # Continue with tool output
        result = await runtime.run(
            prompt=f"Tool {tool_name} returned: {tool_output}"
        )
    
    return result
```

## Design Decisions

1. **Tool Types via type()**: Using Python's `type()` function to create classes dynamically, avoiding metaclass complexity.

2. **Automatic Type Detection**: The system automatically detects whether to use Pydantic or Deserializable based on the output type.

3. **Parameter Validation**: All tool parameters must be typed to ensure proper tool class generation.

4. **Union Types**: Tools and the final output are combined into a Union type, allowing the LLM to choose.

5. **Explicit Tool Execution**: Users control the tool execution loop, providing maximum flexibility.

## Example Files

- `example_tools.py`: Complex example with research agent using multiple tools
- `example_simple_tools.py`: Simple example demonstrating basic tool usage

## Notes

- The parser methods (`enhance_prompt`, `create_stream_parser`) expect single types, not Unions, so we use the original output type for these operations while using the Union type for parsing.
- Tool classes include `_tool_func` and `_tool_name` attributes for execution and identification.
- The system falls back to the existing `tool_chain` method for non-typed outputs.
