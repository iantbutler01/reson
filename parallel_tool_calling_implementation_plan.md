# Parallel Tool Calling Implementation Plan

## [Overview]
Enable parallel tool calling by removing early return logic that stops after first tool, allowing natural sequential emission of multiple `tool_call_complete` events for user-driven parallel execution.

Current streaming infrastructure already works correctly with sequential tool generation pattern: `0_delta 0_delta 0_complete 1_delta 1_complete 2_complete`. LLMs generate tools sequentially, parallel execution happens on the user side with async tasks. Only need to remove artificial limitations that stop after first tool.

## [Types]
No new types required - leverage existing `tool_call_complete` events and current `ToolCallInstance` objects.

Sequential tool emission pattern means existing single-tool data structures work perfectly for multiple tools. Each tool completes independently and flows through existing streaming infrastructure.

## [Files]
Minimal changes to existing files for maximum backwards compatibility.

**Files to modify:**
- `reson/reson.py` - Remove early return logic in `_call_llm()` and `_call_llm_stream()`
- `integration_tests/test_parallel_tool_calling.py` - New comprehensive parallel test suite

**No changes needed:**
- `reson/services/inference_clients.py` - Current streaming accumulation works correctly
- `reson/utils/parsers/native_tool_parser.py` - Current parsing works correctly
- All existing test files - Maintain 100% backwards compatibility

## [Functions]
Minimal function modifications to enable parallel flow.

**Modified functions:**
- `_call_llm()` in `reson/reson.py` - Remove `return tool_calls[0], None, None` limitation
- `_call_llm_stream()` in `reson/reson.py` - Ensure all tools flow through streaming

**No new functions needed:**
- Existing `execute_tool()` works for each tool
- Users handle parallel execution with `asyncio.create_task()` or similar
- No need for specialized parallel execution methods

## [Classes]
No class modifications required - existing classes handle parallel flow naturally.

Current `Runtime`, inference clients, and tool parsing classes work correctly for sequential tool emission. Parallel execution is user responsibility using existing async patterns.

## [Dependencies]
No new dependencies required - existing asyncio for user-driven parallel execution.

Current tool calling infrastructure supports multiple tools through sequential emission without additional dependencies.

## [Testing]
Add comprehensive parallel tool calling tests as failing tests to drive implementation.

**New test file:** `integration_tests/test_parallel_tool_calling.py`

**Test scenarios:**
1. **OpenAI parallel tools** - Multiple tools in single response, sequential completion
2. **Anthropic parallel tools** - Multiple `tool_use` blocks, sequential emission  
3. **Google compositional tools** - Tool chaining and multiple function calls
4. **Backwards compatibility** - Existing single tool patterns still work
5. **Mixed tool types** - Parallel tools with/without `tool_type` registration
6. **User parallel execution** - Handling multiple `tool_call_complete` events concurrently

**Verification approach:**
- Tests initially fail due to early return limitation
- Implementation removes early returns
- Tests pass, proving parallel support works
- All existing tests continue passing

## [Implementation Order]
Test-driven development approach starting with failing tests.

1. **Add failing parallel tool tests** - Comprehensive test coverage for multiple tool scenarios
2. **Remove early return in `_call_llm()`** - Let all tools flow through instead of stopping at first
3. **Verify streaming continues** - Ensure `_call_llm_stream()` emits all sequential tool completions
4. **Fix Google multi-function handling** - Process all function calls in buffered responses
5. **Fix Anthropic multi-block handling** - Process all tool_use blocks in response
6. **Add Google compositional tests** - Tool chaining scenarios
7. **Validate backwards compatibility** - Ensure all existing tests still pass
8. **Add parallel execution examples** - Documentation and example patterns
9. **Performance testing** - Verify no regressions in single tool scenarios
10. **Documentation updates** - Parallel tool calling usage patterns

**Critical principle:** Preserve 100% backwards compatibility - existing single tool code works unchanged, parallel support is purely additive through natural streaming flow.
