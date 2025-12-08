# Native Tool Hydration & Structured Outputs Spec

## Status: ✅ COMPLETE (December 2024)

All features described in this spec have been implemented and tested.

## Overview

This spec covers wiring up two related features in reson-py that exist in skeleton form but aren't functional:

1. **Tool Call Hydration**: When LLM returns a tool call, instantiate the registered `tool_type` class (Pydantic or Deserializable) from the JSON args before calling the tool function ✅ **DONE**
2. **Structured Outputs via `output_type`**: When user specifies `output_type`, use native provider structured output APIs to get typed responses ✅ **DONE**

Both features avoid XML parsing entirely - they use native provider JSON formats.

### Provider Support Summary

| Provider | Structured Outputs | Tool Hydration | Status |
|----------|-------------------|----------------|--------|
| **OpenAI** | `response_format` with `json_schema` | ✅ | ✅ Complete |
| **Anthropic** | `output_format` + `anthropic-beta: structured-outputs-2025-11-13` | ✅ | ✅ Complete |
| **Google Gemini** | `response_schema` in generation config | ✅ | ✅ Complete |
| **OpenRouter** | Pass-through to underlying provider | ✅ | ✅ Complete |
| **Bedrock** | `output_format` (same as Anthropic) | ✅ | ✅ Complete |

---

## Part 1: Tool Call Hydration

### Current State

```python
# User registers a tool with a type
class WeatherQuery(Deserializable):
    city: str
    units: str = "celsius"

def get_weather(query: WeatherQuery) -> str:
    return f"Weather in {query.city}: 22°{query.units[0].upper()}"

runtime.tool(get_weather, tool_type=WeatherQuery)
```

**What happens now:**
- `tool_type` is stored in `runtime.tool_types` registry
- Schema is generated from the type for the LLM
- LLM returns tool call with JSON args: `{"city": "Paris", "units": "celsius"}`
- `execute_tool()` passes raw `ToolCall` object to `get_weather()`
- **FAILS** because function expects `WeatherQuery`, not `ToolCall`

**What should happen:**
- Look up `tool_types["get_weather"]` → `WeatherQuery`
- Instantiate: `WeatherQuery(**{"city": "Paris", "units": "celsius"})`
- Pass typed instance to `get_weather(query)`

### Implementation

#### File: `reson-py/src/runtime.rs`

Modify `execute_tool()` (around line 373):

```rust
fn execute_tool<'py>(
    &self,
    py: Python<'py>,
    tool_result: PyObject,
) -> PyResult<Bound<'py, PyAny>> {
    let tools = self.tools.clone();
    let tool_types = self.tool_types.clone();  // ADD: clone tool_types
    let tool_result_clone = tool_result.clone_ref(py);

    // Get tool name
    let tool_name: String = {
        let bound = tool_result.bind(py);
        if let Ok(name) = bound.getattr("tool_name") {
            name.extract()?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tool result must have tool_name attribute"
            ));
        }
    };

    // Get args dict from ToolCall
    let args_dict: PyObject = {
        let bound = tool_result.bind(py);
        bound.getattr("args")?.extract()?
    };

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
        // Look up the tool function
        let tool_fn = {
            let guard = tools.read().await;
            guard.get(&tool_name).cloned()
        };

        let tool_fn = tool_fn.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("Tool '{}' not found", tool_name))
        })?;

        // Look up the tool type (if registered)
        let tool_type_opt = {
            let guard = tool_types.read().await;
            guard.get(&tool_name).cloned()
        };

        Python::with_gil(|py| -> PyResult<PyObject> {
            let asyncio = py.import("asyncio")?;

            // Hydrate args into typed instance if tool_type registered
            let call_arg = if let Some(tool_type) = tool_type_opt {
                let args_bound = args_dict.bind(py);

                // Check if Pydantic (has model_validate classmethod)
                if tool_type.bind(py).hasattr("model_validate")? {
                    // Pydantic V2: MyClass.model_validate(args_dict)
                    tool_type.call_method1(py, "model_validate", (args_bound,))?
                } else {
                    // Deserializable/dataclass: MyClass(**args_dict)
                    tool_type.call(py, (), Some(args_bound.downcast()?))?
                }
            } else {
                // No type registered, pass raw ToolCall
                tool_result_clone.clone_ref(py)
            };

            let result = tool_fn.call1(py, (call_arg,))?;

            // Check if it's a coroutine and await it
            if asyncio.call_method1("iscoroutine", (&result,))?.is_truthy()? {
                Ok(result)
            } else {
                Ok(result)
            }
        })
    })
}
```

### Supported Types

The hydration should support:

1. **Pydantic BaseModel** (V1 and V2)
   - Detection: `hasattr(type, "model_validate")` (V2) or `hasattr(type, "parse_obj")` (V1)
   - Instantiation: `type.model_validate(args_dict)` or `type.parse_obj(args_dict)`

2. **Deserializable** (reson's own base class)
   - Detection: `issubclass(type, Deserializable)` or just fallback
   - Instantiation: `type(**args_dict)`

3. **dataclasses**
   - Detection: `dataclasses.is_dataclass(type)`
   - Instantiation: `type(**args_dict)`

4. **Regular classes**
   - Fallback: `type(**args_dict)`

### Helper Function

Add a Python helper that Rust can call:

```python
# reson/utils/hydration.py
def hydrate_tool_args(tool_type, args_dict):
    """Instantiate a tool type from args dict."""
    import dataclasses

    # Pydantic V2
    if hasattr(tool_type, "model_validate"):
        return tool_type.model_validate(args_dict)

    # Pydantic V1
    if hasattr(tool_type, "parse_obj"):
        return tool_type.parse_obj(args_dict)

    # dataclass
    if dataclasses.is_dataclass(tool_type):
        return tool_type(**args_dict)

    # Deserializable or regular class
    return tool_type(**args_dict)
```

Or implement entirely in Rust - either works.

---

## Part 2: Structured Outputs via `output_type`

### Current State

```python
class Person(BaseModel):
    name: str
    age: int

result = await runtime.run(
    prompt="Extract: Alice is 30 years old",
    output_type=Person
)
# Currently: output_type is ignored (_output_type with underscore)
```

### What Should Happen

When `output_type` is specified:
1. Generate JSON schema from the type
2. Pass to provider's native structured output API
3. Parse response JSON into the type

### Provider Support

| Provider | Structured Output API | How to Use |
|----------|----------------------|------------|
| **OpenAI** | `response_format: { type: "json_schema", json_schema: {...} }` | Add to request |
| **Anthropic** | `output_format: { type: "json_schema", schema: {...} }` + beta header | Native structured outputs (Dec 2025) |
| **Google Gemini** | `response_mime_type: "application/json"` + `response_schema: {...}` | Add to generation config |
| **OpenRouter** | Passes through to underlying provider | Same as underlying |

**Note:** Anthropic requires beta header: `anthropic-beta: structured-outputs-2025-11-13` ✅ Implemented in anthropic.rs

### Implementation

#### Step 1: Schema Generation

Reuse existing schema generation from tool registration:

```rust
fn generate_output_schema(py: Python, output_type: &PyObject) -> PyResult<serde_json::Value> {
    // Use existing schema generator infrastructure
    let schema_gen = py.import("reson.utils.schema_generators")?;
    let schema = schema_gen.call_method1("generate_type_schema", (output_type,))?;
    pythonize::depythonize(&schema)
}
```

#### Step 2: Modify Provider Requests

In `reson-rust/src/providers/` for each provider:

**OpenAI/OpenRouter** (`oai.rs`):
```rust
if let Some(schema) = output_schema {
    request["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {
            "name": output_type_name,
            "schema": schema,
            "strict": true
        }
    });
}
```

**Anthropic** (`anthropic.rs`):
```rust
if let Some(schema) = output_schema {
    // Use native structured outputs (Dec 2025)
    // Requires beta header: anthropic-beta: structured-outputs-2025-11-13
    request["output_format"] = json!({
        "type": "json_schema",
        "schema": schema
    });
}
```

Add beta header to request:
```rust
headers.insert("anthropic-beta", "structured-outputs-2025-11-13");
```

**Google Gemini** (`google_genai.rs`):
```rust
if let Some(schema) = output_schema {
    generation_config["response_mime_type"] = json!("application/json");
    generation_config["response_schema"] = schema;
}
```

#### Step 3: Parse Response

After getting response, parse into typed object:

```rust
fn parse_structured_response(
    py: Python,
    response_json: &str,
    output_type: &PyObject,
) -> PyResult<PyObject> {
    let json_module = py.import("json")?;
    let parsed = json_module.call_method1("loads", (response_json,))?;

    // Hydrate into type (same logic as tool hydration)
    hydrate_tool_args(output_type, parsed)
}
```

#### Step 4: Wire Up in Runtime

In `runtime.rs` `run()` and `run_stream()`:

```rust
// Remove underscore - actually use output_type
fn run(
    &self,
    py: Python<'py>,
    prompt: Option<String>,
    system: Option<String>,
    history: Option<Vec<PyObject>>,
    output_type: Option<PyObject>,  // Now used!
    // ...
) {
    // Generate schema if output_type provided
    let output_schema = if let Some(ref ot) = output_type {
        Some(generate_output_schema(py, ot)?)
    } else {
        None
    };

    // Pass to LLM call
    let result = call_llm(
        // ...
        output_schema,
        // ...
    ).await?;

    // Parse response if output_type provided
    if let Some(ref ot) = output_type {
        parse_structured_response(py, &result.content, ot)
    } else {
        // Return raw string
        Ok(result.content.into_py(py))
    }
}
```

---

## Part 3: Files to Modify

### reson-py/src/runtime.rs
- [x] `execute_tool()`: Add tool type lookup and hydration ✅ DONE (lines 469-569)
- [x] `run()`: Wire up `output_type` schema generation and response parsing ✅ DONE (lines 621-716)
- [x] `run_stream()`: Same for streaming ✅ DONE (lines 760-768)
- [x] Remove underscore from `_output_type` parameters ✅ DONE

### reson-rust/src/runtime/inference.rs
- [x] `call_llm()`: Accept output schema, pass to provider ✅ DONE (lines 379-437)
- [x] `call_llm_stream()`: Same for streaming ✅ DONE (lines 529-588)
- [x] Remove underscore from `_output_type` parameters ✅ DONE

### reson-rust/src/providers/*.rs
- [x] `anthropic.rs`: Add `output_format` + beta header for structured outputs ✅ DONE (lines 122-128, 221-226)
- [x] `oai.rs`: Add `response_format` with JSON schema ✅ DONE (lines 156-167)
- [x] `google.rs`: Add `response_schema` to generation config ✅ DONE (lines 545-549)
- [x] `openrouter.rs`: Pass through to underlying provider ✅ (delegates to OAIClient)
- [x] `bedrock.rs`: Add structured output support ✅ DONE (lines 139-146)

### reson-py/src/types.rs
- [x] Keep `Deserializable` class ✅ Already present

### New file: reson/utils/hydration.py (optional)
- Not needed - hydration is implemented in Rust (reson-py/src/runtime.rs)

---

## Part 4: Files to Delete

These are gasp/XML parsing files that were copied but aren't needed:

- [ ] `reson-py/parser.rs` - XML stream parser
- [ ] `reson-py/python_types.rs` - Type introspection for XML
- [ ] `reson-py/type_string_parser.rs` - Parse type strings like "list[str]"
- [ ] `reson-py/lib-gasp-py.rs` - gasp Python module definition

**Keep:**
- `reson-py/deserializable.py` - Python Deserializable base class (useful for tool types)

---

## Part 5: README Updates

### Root README.md
- [ ] Keep structured output example but ensure it matches actual API
- [ ] Document that `output_type` uses native provider APIs
- [ ] Document `tool_type` parameter for typed tool arguments

### reson-py/README.md
- [ ] Add section on Deserializable vs Pydantic for tool types
- [ ] Document structured output support per provider
- [ ] Remove any XML/gasp references

### reson-rust/README.md
- [ ] Document `#[derive(Tool)]` and `#[derive(Deserializable)]` macros
- [ ] Show Rust structured output examples

---

## Part 6: Testing

### Existing Tests to Verify
- `integration_tests/test_native_tool_deserializable.py` - Should pass after hydration works
- `integration_tests/test_comprehensive_native_tools.py` - Tool calling with types
- `integration_tests/test_toolcall_hydration.py` - Hydration workflows

### New Tests Needed
- Test Pydantic V1 model hydration
- Test Pydantic V2 model hydration
- Test Deserializable hydration
- Test dataclass hydration
- Test `output_type` with each provider (OpenAI, Anthropic, Google, OpenRouter)
- Test streaming with `output_type`

---

## Implementation Order

1. **Tool Call Hydration** (Part 1)
   - Modify `execute_tool()` to hydrate args
   - Run existing tests to verify

2. **Structured Outputs** (Part 2)
   - Wire up `output_type` in runtime
   - Add schema to provider requests
   - Parse responses into types

3. **Cleanup** (Parts 4-5)
   - Delete gasp XML files
   - Update READMEs

4. **Testing** (Part 6)
   - Run existing tests
   - Add new coverage

---

## Open Questions

1. **Streaming with `output_type`**: Should we buffer the entire response before parsing, or attempt incremental parsing? Buffering is simpler and more reliable.

2. **Error handling**: What if hydration fails (missing required field, wrong type)? Raise exception or return None?

3. **Nested Deserializable**: If a tool type has nested Deserializable fields, should we recursively hydrate? The old Python code did this.

4. **tool_type inference**: Should we auto-infer `tool_type` from function signature if not provided? The old code did this with `_create_deserializable_tool_class()`.
