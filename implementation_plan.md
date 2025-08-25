# Implementation Plan

[Overview]
Implement comprehensive native tool streaming support across all inference providers including parallel and compositional tool calling capabilities.

This implementation addresses critical gaps in the current reson codebase where native tool calling is only supported in non-streaming mode. The current `_call_llm_stream` function completely lacks native tool support, and Google's streaming implementation doesn't detect tool calls. Additionally, Google's thought signature preservation is broken for multi-turn conversations, and Anthropic streaming lacks any tool call support. This implementation will bring feature parity across all providers (OpenAI, OpenRouter, Anthropic, Google GenAI, Google Anthropic, Vertex Gemini) for both streaming and non-streaming tool calling, including support for parallel tool calling (multiple simultaneous tools) and compositional tool calling (sequential tool chains where output of one feeds into another).

**CRITICAL REQUIREMENT**: The existing XML tool calling functionality must remain fully functional and backward compatible. All existing XML-based tool calling tests must continue to pass without modification. This implementation only adds native tool calling support alongside the existing XML approach - it does not replace or modify the XML functionality.

[Types]
Add new types and enums to support streaming tool call events and provider-specific tool call patterns.

```python
# New streaming event types in reson/types.py
from enum import Enum
from typing import Any, Dict, List, Union

class StreamingEventType(Enum):
    CONTENT = "content"
    REASONING = "reasoning" 
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_COMPLETE = "tool_call_complete"
    TOOL_CALLS_COMPLETE = "tool_calls_complete"

class StreamingToolCallEvent:
    event_type: StreamingEventType
    tool_call_id: str
    tool_name: str
    arguments: Union[str, Dict[str, Any]]  # partial for delta, complete for complete
    provider: str

class CompositionalToolState:
    active_calls: List[Any]
    completed_calls: List[Any] 
    thought_signatures: Dict[str, bytes]
    conversation_context: List[Any]
```

[Files]
Modify existing inference clients and core streaming functions to support native tool calling in streaming mode.

**Existing files to be modified:**
- `reson/reson.py` - Add native tool support to `_call_llm_stream` function (currently completely missing)
- `reson/services/inference_clients.py` - Update all inference clients to support tool streaming:
  - `GoogleGenAIInferenceClient.connect_and_listen` - Add buffered tool call detection
  - `AnthropicInferenceClient.connect_and_listen` - Add complete tool block streaming  
  - `OAIInferenceClient.connect_and_listen` - Add incremental tool call delta handling
  - `OpenRouterInferenceClient.connect_and_listen` - Inherit OpenAI delta approach
  - `GoogleAnthropicInferenceClient.connect_and_listen` - Use Anthropic patterns via Google
- `reson/reson.py` - Fix `create_tool_result_message` for Google thought signature preservation
- `integration_tests/test_native_tools_real_apis.py` - Add comprehensive tests for all providers
- `integration_tests/test_comprehensive_native_tools.py` - Add parallel and compositional tests
- `integration_tests/test_native_streaming_tools.py` - Complete streaming tool call tests

**New files to be created:**
- None - all functionality will be added to existing files

[Functions]
Add and modify functions to support native tool streaming across all providers with parallel and compositional capabilities.

**New functions to be added:**
- `_detect_google_tool_calls_in_stream(chunk_data, provider)` in `reson/reson.py` - Detect buffered tool calls in Google streaming responses
- `_handle_anthropic_tool_blocks(chunk_data)` in `reson/reson.py` - Process complete Anthropic tool blocks in streaming
- `_aggregate_openai_tool_deltas(chunk_data, accumulated_calls)` in `reson/reson.py` - Build complete tool calls from OpenAI deltas
- `_preserve_thought_signatures(original_response, tool_results)` in `reson/reson.py` - Maintain Google thought signature context
- `_handle_parallel_tool_calls(tool_calls_list)` in `reson/reson.py` - Process multiple simultaneous tool calls
- `_manage_compositional_state(tool_state, new_call)` in `reson/reson.py` - Handle sequential tool chains

**Modified functions:**
- `_call_llm_stream()` in `reson/reson.py` - Add complete native tool support (currently missing entirely)
- `GoogleGenAIInferenceClient.connect_and_listen()` - Add tool call detection in streaming chunks
- `AnthropicInferenceClient.connect_and_listen()` - Add tool block streaming support  
- `OAIInferenceClient.connect_and_listen()` - Add tool call delta and completion handling
- `create_tool_result_message()` in `reson/reson.py` - Fix Google thought signature preservation
- `_parse_native_tool_calls()` in `reson/reson.py` - Add support for streaming contexts

[Classes]
Modify existing inference client classes to support streaming tool calls with provider-specific patterns.

**Modified classes:**
- `GoogleGenAIInferenceClient` in `reson/services/inference_clients.py` - Add buffered tool call detection and thought signature handling in `connect_and_listen` method
- `AnthropicInferenceClient` in `reson/services/inference_clients.py` - Add complete tool block streaming support in `connect_and_listen` method
- `OAIInferenceClient` in `reson/services/inference_clients.py` - Add incremental tool call delta aggregation in `connect_and_listen` method
- `OpenRouterInferenceClient` in `reson/services/inference_clients.py` - Inherit OpenAI streaming patterns for tool calls
- `GoogleAnthropicInferenceClient` in `reson/services/inference_clients.py` - Apply Anthropic tool streaming patterns through Google infrastructure
- `Runtime` in `reson/reson.py` - Update `create_tool_result_message` to preserve Google thought signatures properly

**New classes:**
- None - functionality will be added to existing classes

[Dependencies]
No new dependencies required as all functionality will use existing imports and libraries.

Current dependencies support all required functionality:
- `google.genai` for Google streaming API interaction
- `httpx` for Anthropic streaming
- `openai` patterns already supported for OpenAI/OpenRouter
- All schema generators already exist for each provider

[Testing]
Add comprehensive test coverage for all providers in both streaming and non-streaming modes with parallel and compositional scenarios.

**Test modifications in `integration_tests/test_native_tools_real_apis.py`:**
- `test_direct_anthropic_native_tools()` - Test direct Anthropic client (if credits available, otherwise skip)
- `test_google_genai_native_tools()` - Test Google GenAI non-streaming + streaming + thinking
- `test_google_anthropic_native_tools()` - Test Claude via Google Vertex non-streaming + streaming
- `test_vertex_gemini_native_tools()` - Test Vertex Gemini non-streaming + streaming + thinking
- `test_openai_native_tools()` - Test OpenAI streaming tool calls
- `test_openrouter_native_tools()` - Enhance existing tests with streaming

**Test modifications in `integration_tests/test_comprehensive_native_tools.py`:**
- `test_google_parallel_tool_calling()` - Multiple simultaneous tool calls
- `test_google_compositional_tool_calling()` - Sequential tool chains  
- `test_google_thinking_tool_preservation()` - Thought signature context across tool execution
- `test_provider_tool_streaming_comparison()` - Compare streaming behaviors across providers

**CRITICAL**: All existing XML tool calling regression tests must continue to pass:
- `test_xml_regression_single_turn()` - XML tool calling single turn functionality
- `test_xml_regression_multi_turn()` - XML tool calling multi-turn functionality  
- `test_xml_vs_native_comparison()` - Comparison between XML and native approaches
- Any failure of XML tests indicates broken backward compatibility

**Test modifications in `integration_tests/test_native_streaming_tools.py`:**
- `test_google_streaming_tool_detection()` - Buffered tool call detection in streams
- `test_anthropic_streaming_tool_blocks()` - Complete tool block streaming
- `test_openai_streaming_tool_deltas()` - Incremental tool call assembly
- `test_parallel_streaming_tools()` - Multiple tools in single streaming response

[Implementation Order]
Sequential implementation to minimize conflicts and ensure successful integration across all providers.

1. **Add native tool support to `_call_llm_stream`** - Implement the missing core functionality that enables streaming tool calls
2. **Implement Google tool call detection in streaming** - Add buffered tool call detection to `GoogleGenAIInferenceClient.connect_and_listen`
3. **Fix Google thought signature preservation** - Update `create_tool_result_message` to preserve original response context
4. **Add Anthropic streaming tool support** - Implement complete tool block streaming in `AnthropicInferenceClient.connect_and_listen`
5. **Add OpenAI/OpenRouter tool delta handling** - Implement incremental tool call aggregation in streaming responses
6. **Implement parallel tool calling support** - Add handling for multiple simultaneous tool calls across all providers
7. **Implement compositional tool calling support** - Add sequential tool chain management for Google providers
8. **Add comprehensive test coverage** - Create tests for all providers, modes, and scenarios
9. **Test Google GenAI streaming + thinking + parallel + compositional** - Validate complete Google functionality
10. **Test all other providers streaming and non-streaming** - Ensure feature parity across providers
11. **Validate thought signature preservation in multi-turn scenarios** - Confirm Google thinking context is maintained
12. **Performance testing and optimization** - Ensure streaming performance is maintained with tool call detection
