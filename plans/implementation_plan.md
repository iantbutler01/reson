# Implementation Plan

## [Overview]
Add a ToolCall class that allows users to explicitly represent assistant tool calls in conversation history, enabling proper hydration when only ToolResult objects are available.

The current system has a gap where users can hydrate ToolResult objects from storage but cannot reconstruct the corresponding assistant tool call messages that providers require to precede tool results. This implementation adds a ToolCall class parallel to ToolResult, with factory methods to convert from various provider formats and integration into the existing message conversion system. The approach maintains the user-driven philosophy where users explicitly construct ToolCall objects rather than automatic inference.

## [Types]
Add a new ToolCall class to represent assistant-side tool calls in conversation history.

```python
class ToolCall(ResonBase):
    """Represents an assistant tool call for conversation history."""
    
    tool_use_id: str = Field(description="Unique identifier for the tool call")
    tool_name: str = Field(description="Name of the tool being called")
    args: Optional[Dict[str, Any]] = Field(default=None, description="Tool arguments as dict")
    raw_arguments: Optional[str] = Field(default=None, description="Tool arguments as JSON string")
    signature: Optional[str] = Field(default=None, description="Provider signature for preservation")
    tool_obj: Optional[Any] = Field(default=None, description="Original typed tool call object")
    
    def to_provider_assistant_message(self, provider: InferenceProvider) -> Dict[str, Any]:
        """Convert to provider-specific assistant message format."""
        # Implementation varies by provider:
        # - Anthropic/Bedrock/Google-Anthropic: {"role": "assistant", "content": [{"type": "tool_use", "id": tool_use_id, "name": tool_name, "input": args}]}
        # - OpenAI/OpenRouter/CUSTOM_OPENAI: {"role": "assistant", "tool_calls": [{"id": tool_use_id, "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}]}
        # - Google GenAI: {"role": "model", "content": [{"functionCall": {"name": tool_name, "args": args}}]}
        
    @classmethod
    def create(
        cls,
        tool_call_obj_or_list: Union[Any, List[Any]],
        signature: Optional[str] = None,
    ) -> Union["ToolCall", List["ToolCall"]]:
        """Factory method to create ToolCall(s) from provider-format tool call objects."""
        
    @classmethod
    def _create_single(cls, tool_call_obj: Any, signature: Optional[str] = None) -> "ToolCall":
        """Create single ToolCall from various provider formats:
        - Anthropic: {"id": "toolu_01", "name": "get_weather", "input": {"location": "SF"}}
        - OpenAI/OpenRouter: {"id": "call_abc", "function": {"name": "get_weather", "arguments": '{"location":"SF"}'}}
        - Google: {"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}
        - Deserializable: Object with _tool_name, _tool_use_id, and field attributes
        """
```

## [Files]
Modify existing files to add ToolCall support and export the new type.

**Modified files:**
- `reson/services/inference_clients.py` - Add ToolCall class with factory methods and provider conversion
- `reson/types.py` - Export ToolCall in __all__ list and import statement
- `reson/reson.py` - Update type hints to accept ToolCall in history parameters
- `reson/services/inference_clients.py` - Update `_convert_messages_to_provider_format` method to handle ToolCall objects
- `reson/services/inference_clients.py` - Update `GoogleGenAIInferenceClient._build_parts` to support functionCall parts

**New test file:**
- `integration_tests/test_toolcall_hydration.py` - Test ToolCall creation and provider conversion

## [Functions]
Add factory methods and provider conversion logic for ToolCall objects.

**New functions:**
- `ToolCall.create(tool_call_obj_or_list, signature=None)` in `reson/services/inference_clients.py` - Factory method for creating ToolCall from provider formats
- `ToolCall._create_single(tool_call_obj, signature=None)` in `reson/services/inference_clients.py` - Internal method to create single ToolCall from various formats
- `ToolCall.to_provider_assistant_message(provider)` in `reson/services/inference_clients.py` - Convert ToolCall to provider-specific assistant message format

**Modified functions:**
- `InferenceClient._convert_messages_to_provider_format()` in `reson/services/inference_clients.py` - Add ToolCall handling to message conversion logic
- `GoogleGenAIInferenceClient._build_parts()` in `reson/services/inference_clients.py` - Add support for functionCall content in parts building

## [Classes]
Add ToolCall class as a new conversation history element type.

**New classes:**
- `ToolCall(ResonBase)` in `reson/services/inference_clients.py` - Main class for representing assistant tool calls with provider conversion methods

**Modified classes:**
- Update type hints in `Runtime` class methods (`run`, `run_stream`) in `reson/reson.py` to accept `ToolCall` in history parameters
- Update `InferenceClient` abstract base class method signatures to accept `ToolCall` in message lists

## [Dependencies]
No new dependencies required.

All functionality uses existing dependencies including `json`, `typing`, and `pydantic` which are already available in the codebase.

## [Testing]
Add focused integration tests for ToolCall creation and provider conversion.

**New test functions in `integration_tests/test_toolcall_hydration.py`:**
- `test_toolcall_create_from_openai_format()` - Test creating ToolCall from OpenAI tool_calls format
- `test_toolcall_create_from_anthropic_format()` - Test creating ToolCall from Anthropic tool_use format  
- `test_toolcall_create_from_google_format()` - Test creating ToolCall from Google functionCall format
- `test_toolcall_create_from_deserializable()` - Test creating ToolCall from NativeToolParser Deserializable objects
- `test_toolcall_provider_conversion_openai()` - Test ToolCall.to_provider_assistant_message() for OpenAI
- `test_toolcall_provider_conversion_anthropic()` - Test ToolCall.to_provider_assistant_message() for Anthropic
- `test_toolcall_provider_conversion_google()` - Test ToolCall.to_provider_assistant_message() for Google GenAI
- `test_history_with_toolcall_then_toolresult()` - Test complete workflow with ToolCall followed by ToolResult in history

## [Implementation Order]
Implement in dependency order to minimize integration conflicts.

1. **Add ToolCall class definition** - Create the base ToolCall class with fields and basic structure in `reson/services/inference_clients.py`

2. **Implement factory methods** - Add `ToolCall.create()` and `ToolCall._create_single()` with provider format detection and conversion logic

3. **Add provider conversion method** - Implement `ToolCall.to_provider_assistant_message()` with provider-specific formatting for OpenAI, Anthropic, and Google

4. **Update message converter** - Modify `InferenceClient._convert_messages_to_provider_format()` to detect and handle ToolCall objects in message sequences

5. **Extend Google builder** - Update `GoogleGenAIInferenceClient._build_parts()` to support functionCall content parts

6. **Export in types module** - Add ToolCall import and export in `reson/types.py`

7. **Update type hints** - Modify Runtime method signatures in `reson/reson.py` to accept ToolCall in history parameters

8. **Add integration tests** - Create comprehensive tests in `integration_tests/test_toolcall_hydration.py` to validate all functionality

9. **Validate end-to-end** - Test complete workflow with real provider calls to ensure proper message sequencing
