# Implementation Plan

## Overview
Add ReasoningSegment class for thinking signatures support across all providers.

This enables preservation of reasoning context and signatures across multi-turn conversations, following the same pattern as ToolResult. Users can capture reasoning segments with signatures and manually add them to conversation history for continued reasoning context.

## Types
Add ReasoningSegment class with signature preservation and provider format conversion.

```python
class ReasoningSegment(ResonBase):
    content: str
    signature: Optional[str] = Field(default=None)
    provider_metadata: Dict[str, Any] = Field(default_factory=dict)
    segment_index: int = Field(default=0)
```

Provider format conversion methods and signature extraction utilities.

## Files
Modify inference clients and runtime for reasoning segment support.

- `reson/services/inference_clients.py` - Add ReasoningSegment class
- `reson/reson.py` - Add reasoning_segments registry to Runtime class  
- `reson/types.py` - Export ReasoningSegment

Update streaming logic to detect segment boundaries and extract signatures from provider responses.

## Functions
Add reasoning segment management and signature extraction methods.

**ReasoningSegment methods:**
- `to_provider_format(provider: str) -> Dict[str, Any]`
- `_format_anthropic_content()`, `_format_openai_content()`, `_format_google_content()`

**Runtime class updates:**
- Add `reasoning_segments` property
- Add `clear_reasoning_segments()` method
- Modify streaming to detect segment transitions from "reasoning" chunk type
- Extract signatures from provider responses

**Specific inference client updates:**
- `AnthropicInferenceClient` - Extract signatures from signature_delta events in streaming, signature field in responses
- `BedrockInferenceClient` - Same signature handling as Anthropic (uses Anthropic format)  
- `OAIInferenceClient` - Extract reasoning items from responses, handle reasoning field in choices
- `OpenRouterInferenceClient` - Inherits from OAI, same reasoning item handling
- `GoogleGenAIInferenceClient` - Extract thought signatures from response parts
- `GoogleAnthropicInferenceClient` - Inherits from Anthropic, same signature handling

Each client needs:
- `get_generation()` - Extract signatures from provider response format
- `connect_and_listen()` - Detect reasoning segments during streaming, capture signatures
- Segment boundary detection when chunk types transition from "reasoning"

## Classes
Update Runtime class with reasoning segments registry and ReasoningSegment class implementation.

**ReasoningSegment:**
- Provider format conversion like ToolResult
- Signature preservation from Anthropic, OpenAI, Google responses

**Runtime modifications:**
- `_reasoning_segments: List[ReasoningSegment]` registry
- Auto-segmentation on chunk type transitions
- Maintain backward compatibility with existing `reasoning` property

## Dependencies
No new dependencies required.

Uses existing provider response structures and API patterns.

## Testing
Add comprehensive tests for reasoning segments and signature preservation.

- `test_reasoning_segments.py` - Core functionality
- `test_multi_turn_reasoning.py` - Multi-turn conversation patterns
- Update existing reasoning tests for backward compatibility

Test signature extraction from all providers and multi-turn conversation flows with reasoning context preservation.

## Implementation Order
Build ReasoningSegment foundation then add provider-specific signature support.

1. Create ReasoningSegment class with provider format methods
2. Add reasoning_segments registry to Runtime class
3. Implement Anthropic signature extraction in AnthropicInferenceClient, BedrockInferenceClient, GoogleAnthropicInferenceClient
4. Add OpenAI reasoning item support in OAIInferenceClient, OpenRouterInferenceClient
5. Add Google thought signature handling in GoogleGenAIInferenceClient
6. Update streaming logic for segment boundary detection across all clients
7. Add comprehensive test suite covering all providers
8. Validate backward compatibility with existing reasoning accumulator
