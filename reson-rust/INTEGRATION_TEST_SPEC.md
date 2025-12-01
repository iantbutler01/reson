# Reson Rust Integration Test Specification

This document specifies all integration tests to be ported from the Python implementation to Rust.
Tests are organized by category and mapped to their Python source files.

## Environment Variables Required

```bash
ANTHROPIC_API_KEY     # Direct Anthropic API
OPENAI_API_KEY        # Direct OpenAI API
GOOGLE_API_KEY        # Google Gemini API
OPENROUTER_API_KEY    # OpenRouter proxy (supports all models)
# GOOGLE_APPLICATION_CREDENTIALS  # For Vertex AI (optional)
```

---

## 1. Basic Provider Tests

### Source: `test.py`, `test_google_genai.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_anthropic_simple_generation` | Basic text generation with Claude | ✅ Done |
| `test_openai_simple_generation` | Basic text generation with GPT | ✅ Done |
| `test_google_simple_generation` | Basic text generation with Gemini | ✅ Done |
| `test_openrouter_anthropic_model` | OpenRouter with Claude | ✅ Done |
| `test_openrouter_openai_model` | OpenRouter with GPT | ✅ Done |
| `test_extract_scalar` | Return scalar string type | ⏳ Pending |
| `test_extract_people` | Return List[Person] structured output | ⏳ Pending |
| `test_extract_company_streaming` | Streaming structured output | ⏳ Pending |

---

## 2. Native Tool Calling Tests

### Source: `test_native_tools.py`, `test_native_tools_real_apis.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_runtime_native_tools_validation` | Runtime validates native tools support | ⏳ Pending |
| `test_tool_registration_works_with_native_tools` | Tool registration with native_tools=true | ⏳ Pending |
| `test_schema_generation_integration` | Schema generation for registered tools | ⏳ Pending |
| `test_provider_support` | All providers support native tools | ✅ Done |
| `test_openrouter_native_tools` | OpenRouter + Claude with tools | ✅ Done |
| `test_vertex_gemini_native_tools` | Vertex Gemini with tools | ⏳ Pending |
| `test_google_anthropic_native_tools` | Google Anthropic with tools | ⏳ Pending |
| `test_multi_turn_conversation` | Multi-turn with tool results | ✅ Done |

---

## 3. Comprehensive Native Tools Tests

### Source: `test_comprehensive_native_tools.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_xml_regression_single_turn` | XML tool calling still works (no native) | ⏳ Pending |
| `test_xml_regression_multi_turn` | XML multi-turn tool calling | ⏳ Pending |
| `test_native_5_turn_conversation` | 5-turn conversation with native tools | ✅ Done |
| `test_output_type_termination` | Output type terminates conversation | ⏳ Pending |
| `test_native_vs_xml_comparison` | Compare native vs XML approaches | ⏳ Pending |

---

## 4. Streaming Tests

### Source: `test_native_streaming_tools.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_streaming_native_tools` | Stream chunks with tool calls | ⏳ Pending |
| `test_google_streaming_tool_detection` | Google streaming tool detection | ⏳ Pending |
| `test_google_anthropic_streaming_tools` | Google Anthropic streaming tools | ⏳ Pending |
| `test_streaming_tool_execution` | Execute tools from stream | ⏳ Pending |
| `test_anthropic_streaming` | Basic Anthropic streaming | ✅ Done |
| `test_google_streaming` | Basic Google streaming | ✅ Done |

---

## 5. Reasoning/Thinking Tests

### Source: `test_reasoning.py`, `test_reasoning_segments.py`, `test_reasoning_stream.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_reasoning` | O3-mini reasoning tokens | ⏳ Pending |
| `test_anthropic_reasoning` | Claude reasoning with budget | ⏳ Pending |
| `test_reasoning_segments_across_providers` | ReasoningSegment across providers | ⏳ Pending |
| `test_reasoning_segment_provider_formats` | Provider format conversion | ⏳ Pending |
| `test_reasoning_stream` | Streaming with reasoning chunks | ⏳ Pending |
| `test_anthropic_reasoning_stream` | Anthropic streaming reasoning | ⏳ Pending |
| `test_google_with_thinking` | Gemini thinking mode | ✅ Done |

---

## 6. Parallel Tool Calling Tests

### Source: `test_parallel_tool_calling.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_openai_parallel_tool_calling` | OpenAI parallel tools (multiple in response) | ⏳ Pending |
| `test_anthropic_parallel_tool_calling` | Anthropic parallel tool_use blocks | ⏳ Pending |
| `test_google_parallel_tool_calling` | Google compositional tool calling | ⏳ Pending |
| `test_backwards_compatibility` | Single tool patterns still work | ⏳ Pending |
| `test_parallel_execution_pattern` | User pattern for async parallel execution | ⏳ Pending |
| `test_google_compositional_chaining` | Tool results feed into next tool | ⏳ Pending |
| `test_mixed_parallel_tool_types` | Mixed tool registration types | ⏳ Pending |

---

## 7. Tool Call Format Validation Tests

### Source: `test_tool_call_format_validation.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_unregistered_tool_returns_dict` | Unregistered tools return raw dict | ⏳ Pending |
| `test_partial_parsing_returns_none` | Incomplete JSON returns None | ⏳ Pending |
| `test_registered_tool_returns_type_instance` | Registered tools return typed instance | ⏳ Pending |
| `test_mixed_tool_registration` | Some registered, some not | ⏳ Pending |
| `test_cross_provider_format_consistency` | Consistent format across providers | ⏳ Pending |
| `test_streaming_partial_marshalling_progression` | Progressive streaming parsing | ⏳ Pending |
| `test_tool_call_format_edge_cases` | Edge cases in tool call formats | ⏳ Pending |
| `test_multi_turn_toolresult_conversation` | Multi-turn with ToolResult messages | ⏳ Pending |

---

## 8. ToolCall/ToolResult Hydration Tests

### Source: `test_toolcall_hydration.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `TestToolCallCreation` | ToolCall::create() from various formats | ⏳ Pending |
| `TestToolCallProviderConversion` | to_provider_assistant_message() | ⏳ Pending |
| `test_toolcall_in_conversation_history` | ToolCall in message history | ⏳ Pending |
| `test_toolcall_hydration_workflow` | Full hydration workflow | ⏳ Pending |
| `test_toolcall_message_conversion_integration` | Message conversion integration | ⏳ Pending |

---

## 9. Native Tool Deserializable Tests

### Source: `test_native_tool_deserializable.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_tool_type_registration` | Register tools with tool_type | ⏳ Pending |
| `test_deserializable_streaming` | Stream Deserializable types | ⏳ Pending |
| `test_mixed_tool_types` | Mix of typed and untyped tools | ⏳ Pending |
| `test_typed_output_with_tools` | Typed output alongside tools | ⏳ Pending |
| `test_pydantic_marshalling` | Pydantic model marshalling | N/A (use serde) |
| `test_dataclass_marshalling` | Dataclass marshalling | N/A (use structs) |
| `test_collection_types` | Vec, HashMap types | ⏳ Pending |
| `test_primitive_types` | String, i32, f64, bool | ⏳ Pending |

---

## 10. Simple Tools Tests

### Source: `test_simple_tools.py`, `test_tools.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_simple_agent_math` | Simple agent with calculate tool | ⏳ Pending |
| `test_simple_agent_facts` | Simple agent with get_fact tool | ⏳ Pending |
| `test_simple_agent_no_tools` | Direct answer without tools | ⏳ Pending |
| `test_research_agent_loop` | Multi-tool research loop | ⏳ Pending |

---

## 11. Generator Tests

### Source: `test_generator.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_process_data_stream` | Async generator yielding items | ⏳ Pending |
| `test_process_data_with_status` | Generator with status updates | ⏳ Pending |

---

## 12. Schema Format Tests

### Source: `test_google_schema_format.py`, `test_tool_type_schema_precedence.py`

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_schema_with_real_client` | Google schema format validation | ⏳ Pending |
| `test_schema_prefers_tool_type_and_warns_on_union` | tool_type takes precedence | ⏳ Pending |
| `test_google_additional_properties` | Google additionalProperties handling | ⏳ Pending |

---

## 13. Error Handling Tests

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_invalid_api_key_anthropic` | Invalid key returns error | ✅ Done |
| `test_invalid_api_key_google` | Invalid key returns error | ✅ Done |
| `test_invalid_api_key_openai` | Invalid key returns error | ⏳ Pending |
| `test_rate_limit_retry` | Retry on rate limit | ⏳ Pending |
| `test_timeout_handling` | Request timeout handling | ⏳ Pending |

---

## 14. Provider Detection Tests

| Test | Description | Rust Status |
|------|-------------|-------------|
| `test_provider_from_model_string` | Parse "provider:model" format | ✅ Done |
| `test_provider_supports_native_tools` | Check native tools support | ✅ Done |

---

## Tests NOT Ported (Postgres/Storage specific)

These tests are for the PostgreSQL storage backend which is a separate concern:

- `test_postgres.py` - Basic Postgres
- `test_async_postgres.py` - Async Postgres operations
- `test_sync_postgres.py` - Sync Postgres operations
- `test_serialization.py` - Model serialization
- `test_nested_preload.py` - Nested relationship preloading
- `test_many_to_many_nested_preload.py` - M2M preloading
- `test_instance_preload.py` - Instance-level preloading
- `test_idempotent_preload.py` - Idempotent preload behavior

---

## Implementation Priority

### Phase 1: Core Provider Tests (HIGH)
1. Basic generation for all providers
2. Tool calling for all providers
3. Multi-turn conversations
4. Error handling

### Phase 2: Streaming & Reasoning (MEDIUM)
1. Streaming for all providers
2. Reasoning/thinking mode
3. Streaming with tools

### Phase 3: Advanced Tool Features (MEDIUM)
1. Parallel tool calling
2. Tool call format validation
3. ToolCall/ToolResult hydration
4. Schema generation

### Phase 4: Edge Cases & Compatibility (LOW)
1. XML regression tests
2. Mixed tool types
3. Generator patterns
4. Edge cases

---

## Running Tests

```bash
# Run all integration tests (requires API keys)
cargo test --test integration_tests -- --ignored

# Run specific provider
cargo test --test integration_tests test_anthropic -- --ignored
cargo test --test integration_tests test_google -- --ignored
cargo test --test integration_tests test_openrouter -- --ignored

# Run non-API tests (no keys required)
cargo test --test integration_tests

# Run with output
cargo test --test integration_tests -- --ignored --nocapture
```

---

## Test File Organization

```
reson-rust/tests/
├── integration_tests.rs          # Main integration tests (current)
├── test_native_tools.rs          # Native tool calling tests
├── test_streaming.rs             # Streaming tests
├── test_reasoning.rs             # Reasoning/thinking tests
├── test_parallel_tools.rs        # Parallel tool calling
├── test_tool_formats.rs          # Tool call format validation
├── test_hydration.rs             # ToolCall/ToolResult hydration
├── macro_tests.rs                # Macro tests (existing)
└── test_anthropic_client.rs      # Anthropic-specific tests (existing)
```
