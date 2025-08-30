# Implementation Plan

## Overview
Create a new ToolResult class and add signature support to ChatMessage for handling signature and thought_signature fields from Anthropic and Google providers.

This implementation introduces ToolResult as a new message type in the conversation history, similar to ChatMessage, that encapsulates tool result content. ToolResult will be transformed into provider-specific JSON formats during API calls. Additionally, ChatMessage will gain an optional signature field to handle signature and thought_signature fields from Anthropic and Google providers, ensuring proper verification of thinking/reasoning content in multi-turn conversations.

## Types
Create new ToolResult class as a conversation history element and enhance existing ChatMessage with signature field.

### New ToolResult Class
```python
class ToolResult(ResonBase):
    """Encapsulates tool result content for conversation history."""
    
    tool_use_id: str
    content: str
    is_error: bool = False
    signature: Optional[str] = Field(default=None)  # For signature/thought_signature handling
    tool_obj: Optional[Any] = Field(default=None)  # Original tool call object (dict or marshalled Deserializable)
    
    def to_provider_format(self, provider: str) -> Dict[str, Any]:
        """Convert to provider-specific format for API calls."""
        pass
    
    def to_chat_message(self) -> ChatMessage:
        """Convert to ChatMessage for backward compatibility."""
        pass
    
    @classmethod
    def create(
        cls,
        tool_call_obj_or_list: Union[Any, Tuple[Any, str], List[Tuple[Any, str]]],
        result: Optional[str] = None,
        signature: Optional[str] = None
    ) -> Union['ToolResult', List['ToolResult']]:
        """Factory method to create ToolResult(s) from tool call objects."""
        pass
```

### Enhanced ChatMessage Type
```python
class ChatMessage(ResonBase):
    role: ChatRole
    content: str
    cache_marker: bool = False
    model_families: List[str] = Field(default_factory=list)
    signature: Optional[str] = Field(default=None)  # NEW: Support for signature/thought_signature
```

## Files
Create ToolResult class and enhance existing message handling with signature support.

### Modified Files:
- **reson/services/inference_clients.py**: 
  - Add `signature` field to `ChatMessage` class
  - Add new `ToolResult` class
  - Update conversation history handling to support ToolResult
- **reson/reson.py**: 
  - Update `Runtime.create_tool_result_message` to return `ToolResult` instead of `ChatMessage`
  - Add ToolResult to conversation history handling
  - Update provider-specific API call formatting to handle ToolResult
- **reson/types.py**: Export `ToolResult` class

### Files requiring API call format updates:
- **reson/services/inference_clients.py**: Update all inference client classes to handle ToolResult in message conversion
- **reson/utils/schema_generators/**: Update schema generators if they handle message formatting

### Test Files to Update:
- **integration_tests/test_comprehensive_native_tools.py**: Update to handle ToolResult return type
- **integration_tests/test_native_tools_real_apis.py**: Update tool result creation calls
- All test files using `create_tool_result_message`: Update to work with ToolResult

## Functions
Move create_tool_result_message to ToolResult class and update message handling functions.

### New Functions in ToolResult Class:
- **`create`**: Class method factory to create ToolResult(s) from tool call objects and results
- **`to_provider_format`**: Convert ToolResult to provider-specific API format (OpenAI, Anthropic, Google)
- **`to_chat_message`**: Convert ToolResult to ChatMessage for backward compatibility
- **`_extract_signature`**: Extract signature/thought_signature from tool call objects
- **`_format_anthropic_content`**: Format content for Anthropic API calls
- **`_format_openai_content`**: Format content for OpenAI/OpenRouter API calls  
- **`_format_google_content`**: Format content for Google API calls

### Modified Functions in Runtime Class:
- **`create_tool_result_message`**: Return ToolResult instead of ChatMessage
- **Message history handling**: Support ToolResult in conversation history
- **API call preparation**: Convert ToolResult to appropriate provider format

### Updated Functions in InferenceClient Classes:
- **Message conversion methods**: Handle ToolResult when preparing API requests
- **Provider-specific formatters**: Convert ToolResult to correct API format for each provider

## Classes
Create ToolResult class as conversation history element and enhance ChatMessage.

### New ToolResult Class:
- **Location**: `reson/services/inference_clients.py` (alongside ChatMessage)
- **Purpose**: Encapsulate tool result content for conversation history
- **Key Features**: 
  - Stores tool_use_id, content, error status, signature, and original tool_obj
  - Converts to provider-specific formats for API calls
  - Integrates with conversation history like ChatMessage
  - Preserves access to tool call metadata through tool_obj (dict or marshalled Deserializable)
- **Inheritance**: Extends `ResonBase` like ChatMessage

### Enhanced ChatMessage Class:
- **Location**: `reson/services/inference_clients.py`
- **Modification**: Add optional `signature: Optional[str]` field
- **Purpose**: Store signature/thought_signature data for reasoning verification
- **Backward Compatibility**: Optional field ensures existing code continues working

## Dependencies
No new external dependencies required.

### Internal Dependencies:
- `ResonBase` for ToolResult base class
- `ChatMessage` and `ChatRole` from services module
- Provider-specific API formatting logic
- Conversation history handling in Runtime class

### Integration Points:
- ToolResult must integrate with existing conversation history mechanisms
- Provider-specific API clients must handle ToolResult → API format conversion
- Backward compatibility with existing `create_tool_result_message` usage patterns

## Testing
Comprehensive testing approach for new ToolResult class and signature handling.

### Test Strategy:
- **ToolResult Creation Tests**: Verify ToolResult creation from tool call objects
- **Provider Format Tests**: Test ToolResult conversion to OpenAI, Anthropic, Google formats
- **Signature Handling Tests**: Test signature/thought_signature extraction and preservation
- **Backward Compatibility Tests**: Ensure existing code using ChatMessage still works
- **Integration Tests**: Test ToolResult in full conversation flows

### New Test Files Needed:
- **test_tool_result.py**: Test ToolResult class functionality
- **test_tool_result_signatures.py**: Test signature handling
- **test_tool_result_providers.py**: Test provider-specific formatting

### Existing Tests to Verify:
- All tests using `create_tool_result_message` must work with ToolResult
- Native and XML tool calling tests must preserve functionality
- Multi-turn conversation tests must handle ToolResult properly

## Implementation Order
Sequential implementation steps to minimize conflicts and ensure successful integration.

### Step 1: Create ToolResult class structure
- Add ToolResult class to `reson/services/inference_clients.py`
- Implement basic fields: tool_use_id, content, is_error, signature, tool_obj
- Add to exports in `reson/types.py`

### Step 2: Implement ToolResult factory method
- Add `ToolResult.create()` class method
- Handle single tool results and parallel tool results
- Extract signatures from tool call objects (Anthropic signature, Google thought_signature)

### Step 3: Add provider-specific formatting to ToolResult
- Implement `to_provider_format()` method
- Add Anthropic format (tool_result blocks with tool_use_id)
- Add OpenAI format (function_call_output with call_id)
- Add Google format (functionResponse with thought signature support)

### Step 4: Enhance ChatMessage with signature field
- Add optional `signature` field to ChatMessage
- Ensure backward compatibility
- Update type exports

### Step 5: Update Runtime.create_tool_result_message
- Modify to return ToolResult instead of ChatMessage
- Maintain backward compatibility by using ToolResult.create()
- Update conversation history to support ToolResult

### Step 6: Update inference clients for ToolResult support
- Modify message conversion in all InferenceClient classes
- Handle ToolResult → provider format conversion in API calls
- Ensure proper signature handling for each provider

### Step 7: Add ToolResult to conversation history handling
- Update Runtime to store ToolResult in history
- Ensure proper serialization/deserialization
- Handle mixed ChatMessage/ToolResult history

### Step 8: Comprehensive testing and validation
- Test ToolResult creation and formatting
- Verify signature extraction and preservation
- Test all provider formats (OpenAI, Anthropic, Google)
- Run existing tests to ensure no regressions
- Test multi-turn conversations with ToolResult

### Step 9: Documentation and cleanup
- Update docstrings for new ToolResult functionality
- Document signature field usage
- Update examples showing ToolResult usage
