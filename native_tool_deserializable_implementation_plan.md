# Native Tool Deserializable Implementation Plan

## Overview
Enhance native tool calling to support Deserializable tool types with partial parsing for consistency with XML patterns.

## Core Changes

### 1. Enhanced Tool Registration
```python
# Current
runtime.tool(calculate_func)

# Enhanced  
runtime.tool(calculate_func, tool_type=CalculationQuery)
```

**Implementation:**
- Add `tool_type` parameter to `Runtime.tool()` method
- Store tool type mapping: `_tool_types: Dict[str, Type[Deserializable]]`
- Maintain backward compatibility for tools without tool_type

### 2. NativeToolParser Class
```python
class NativeToolParser(TypeParser):
    """Parser for native tool calls that builds Deserializable objects from JSON deltas."""
    
    def __init__(self, tools_registry: Dict[str, Type[Deserializable]]):
        super().__init__()
        self.tools_registry = tools_registry
        
    def parse_tool_delta(self, tool_name: str, delta_json: str) -> ParserResult:
        """Parse partial tool call JSON into partial Deserializable."""
        
    def parse_tool_complete(self, tool_name: str, complete_json: str) -> ParserResult:
        """Parse complete tool call JSON into complete Deserializable."""
```

### 3. Streaming Integration
**In `_call_llm_stream` when handling tool_call_delta:**
```python
if chunk_type == "tool_call_delta":
    tool_name = extract_tool_name(chunk_content)
    if tool_name in runtime._tool_types:
        # Tool has registered Deserializable type - parse to typed object
        tool_parser = NativeToolParser(runtime._tool_types)
        partial_result = tool_parser.parse_tool_delta(tool_name, delta_json)
        yield partial_result.value, chunk_content, "tool_call_delta"
    else:
        # Tool has NO registered type - yield raw delta for current behavior
        yield chunk_content, chunk_content, "tool_call_delta"

if chunk_type == "tool_call_complete":
    tool_name = extract_tool_name(chunk_content)
    if tool_name in runtime._tool_types:
        # Tool has registered Deserializable type - parse to typed object
        tool_parser = NativeToolParser(runtime._tool_types)
        complete_result = tool_parser.parse_tool_complete(tool_name, complete_json)
        yield complete_result.value, None, "tool_call_complete"
    else:
        # Tool has NO registered type - use existing _create_tool_instance fallback
        tool_calls = _parse_native_tool_calls(chunk_content, tools, provider)
        for tool_call in tool_calls:
            yield tool_call, None, "tool_call_complete"
```

### 4. Backward Compatibility Strategy
**Tools without tool_type specified:**
- **Delta events**: Yield raw JSON delta objects (current behavior)
- **Complete events**: Use existing `_create_tool_instance` wrapper (current behavior)
- **Execution**: Use existing `Runtime.execute_tool()` (current behavior)
- **No breaking changes**: All existing tools continue to work unchanged

**Tools with tool_type specified:**
- **Delta events**: Yield partial Deserializable objects with `__gasp_from_partial__`
- **Complete events**: Yield complete Deserializable objects 
- **Execution**: Enhanced execution with proper typed objects
- **UI benefits**: Progressive tool building with type safety

### 4. Parser Logic
**Delta Parsing:**
```python
def parse_tool_delta(self, tool_name: str, delta_json: str) -> ParserResult:
    try:
        partial_data = json.loads(delta_json)  # {"a": 5}
        tool_type = self.tools_registry[tool_name]  # CalculationQuery
        partial_tool = tool_type.__gasp_from_partial__(partial_data)
        return ParserResult(value=partial_tool, is_partial=True)
    except Exception as e:
        return ParserResult(error=e, is_partial=True)
```

**Complete Parsing:**
```python
def parse_tool_complete(self, tool_name: str, complete_json: str) -> ParserResult:
    try:
        complete_data = json.loads(complete_json)  # {"a": 5, "b": 12, "operation": "add"}
        tool_type = self.tools_registry[tool_name]
        complete_tool = tool_type.__gasp_from_partial__(complete_data)
        return ParserResult(value=complete_tool, is_partial=False)
    except Exception as e:
        return ParserResult(error=e, is_partial=False)
```

## Key Benefits
- **Delta consistency**: Partial Deserializable objects just like XML
- **Performance**: No XML parsing overhead, immediate tool detection
- **UI streaming**: Progressive tool building with typed objects
- **Backward compatibility**: Tools without tool_type work unchanged
- **Pattern consistency**: Uses existing OutputParser interface

## Implementation Files
- `reson/reson.py` - Enhanced `Runtime.tool()` method, tool type storage
- `reson/utils/parsers/native_tool_parser.py` - New NativeToolParser class
- `reson/utils/parsers/__init__.py` - Export new parser
- `integration_tests/test_native_tool_deserializable.py` - Test the feature

## Test Validation
- Tool deltas yield partial Deserializable objects
- Tool complete yields complete Deserializable objects  
- Backward compatibility with existing tool registration
- Performance improvement over XML parsing
- UI can display progressive tool building
