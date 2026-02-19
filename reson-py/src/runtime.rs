//! Python Runtime wrapper

use futures::Stream;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::errors::to_py_err;
use crate::types::{ChatMessage, ReasoningSegment, ToolResult};

type ChunkStream = Pin<
    Box<dyn Stream<Item = Result<(String, serde_json::Value), reson_agentic::error::Error>> + Send>,
>;

/// Generate JSON schema from a Python type (Pydantic model, dataclass, Deserializable, etc.)
/// Note: Provider-specific schema fixes are applied in the Rust inference layer
fn generate_output_schema(py: Python<'_>, output_type: &PyObject) -> Option<serde_json::Value> {
    // Try Pydantic V2 first (model_json_schema)
    if let Ok(schema) = output_type.call_method0(py, "model_json_schema") {
        if let Ok(py_dict) = schema.extract::<Bound<'_, PyDict>>(py) {
            if let Ok(json_val) = pythonize::depythonize::<serde_json::Value>(&py_dict) {
                return Some(json_val);
            }
        }
    }

    // Try Pydantic V1 (schema)
    if let Ok(schema) = output_type.call_method0(py, "schema") {
        if let Ok(py_dict) = schema.extract::<Bound<'_, PyDict>>(py) {
            if let Ok(json_val) = pythonize::depythonize::<serde_json::Value>(&py_dict) {
                return Some(json_val);
            }
        }
    }

    // Try dataclasses (use __dataclass_fields__)
    if output_type
        .bind(py)
        .hasattr("__dataclass_fields__")
        .unwrap_or(false)
    {
        // For dataclasses, construct a basic schema from fields
        if let Ok(fields) = output_type.getattr(py, "__dataclass_fields__") {
            if let Ok(fields_dict) = fields.extract::<Bound<'_, PyDict>>(py) {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();

                for (key, _value) in fields_dict.iter() {
                    if let Ok(field_name) = key.extract::<String>() {
                        // Default to string type for now - dataclasses don't provide rich type info
                        properties
                            .insert(field_name.clone(), serde_json::json!({"type": "string"}));
                        required.push(serde_json::Value::String(field_name));
                    }
                }

                return Some(serde_json::json!({
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": false
                }));
            }
        }
    }

    // Try Deserializable classes (have __annotations__ and __gasp_from_partial__)
    // This handles custom classes that inherit from Deserializable
    if output_type
        .bind(py)
        .hasattr("__annotations__")
        .unwrap_or(false)
    {
        if let Ok(annotations) = output_type.getattr(py, "__annotations__") {
            if let Ok(annotations_dict) = annotations.extract::<Bound<'_, PyDict>>(py) {
                let mut properties = serde_json::Map::new();
                let mut required = Vec::new();

                for (key, type_hint) in annotations_dict.iter() {
                    if let Ok(field_name) = key.extract::<String>() {
                        // Get the JSON schema type from Python type hint
                        let json_type = python_type_to_json_schema(py, &type_hint);
                        properties.insert(field_name.clone(), json_type.clone());

                        // Check if field is optional (has default value or is Optional[T])
                        let is_optional = is_optional_type(py, &type_hint)
                            || has_class_default(py, output_type, &field_name);

                        if !is_optional {
                            required.push(serde_json::Value::String(field_name));
                        }
                    }
                }

                // Get class name for title
                let title = output_type
                    .getattr(py, "__name__")
                    .and_then(|n| n.extract::<String>(py))
                    .unwrap_or_else(|_| "Object".to_string());

                return Some(serde_json::json!({
                    "type": "object",
                    "title": title,
                    "properties": properties,
                    "required": required
                }));
            }
        }
    }

    None
}

/// Convert Python type hint to JSON schema type
fn python_type_to_json_schema(
    py: Python<'_>,
    type_hint: &Bound<'_, pyo3::PyAny>,
) -> serde_json::Value {
    // Get the type name or repr
    let type_repr = type_hint
        .repr()
        .and_then(|r| r.extract::<String>())
        .unwrap_or_else(|_| "Any".to_string());

    // Check for Optional types (Union[X, None] or X | None)
    if type_repr.starts_with("typing.Optional[")
        || type_repr.contains(" | None")
        || type_repr.contains("None |")
    {
        // Extract inner type from Optional[X]
        if let Ok(args) = type_hint.getattr("__args__") {
            if let Ok(args_tuple) = args.extract::<Vec<Bound<'_, pyo3::PyAny>>>() {
                // Find the non-None type
                for arg in args_tuple {
                    let arg_repr = arg
                        .repr()
                        .and_then(|r| r.extract::<String>())
                        .unwrap_or_default();
                    if arg_repr != "<class 'NoneType'>" && arg_repr != "None" {
                        let inner_schema = python_type_to_json_schema(py, &arg);
                        // Return anyOf with inner type and null
                        return serde_json::json!({
                            "anyOf": [inner_schema, {"type": "null"}]
                        });
                    }
                }
            }
        }
        return serde_json::json!({"type": "string"});
    }

    // Check for List types
    if type_repr.starts_with("typing.List[") || type_repr.starts_with("list[") {
        if let Ok(args) = type_hint.getattr("__args__") {
            if let Ok(args_tuple) = args.extract::<Vec<Bound<'_, pyo3::PyAny>>>() {
                if let Some(item_type) = args_tuple.first() {
                    let item_schema = python_type_to_json_schema(py, item_type);
                    return serde_json::json!({
                        "type": "array",
                        "items": item_schema
                    });
                }
            }
        }
        return serde_json::json!({"type": "array"});
    }

    // Check for Dict types
    if type_repr.starts_with("typing.Dict[") || type_repr.starts_with("dict[") {
        return serde_json::json!({"type": "object"});
    }

    // Get the actual type name
    let type_name = if let Ok(name) = type_hint.getattr("__name__") {
        name.extract::<String>()
            .unwrap_or_else(|_| type_repr.clone())
    } else {
        // Extract from repr like "<class 'str'>" or "typing.List[str]"
        if type_repr.starts_with("<class '") && type_repr.ends_with("'>") {
            type_repr[8..type_repr.len() - 2].to_string()
        } else {
            type_repr.clone()
        }
    };

    // Map Python types to JSON schema types
    match type_name.as_str() {
        "str" | "string" => serde_json::json!({"type": "string"}),
        "int" | "integer" => serde_json::json!({"type": "integer"}),
        "float" | "number" => serde_json::json!({"type": "number"}),
        "bool" | "boolean" => serde_json::json!({"type": "boolean"}),
        "list" | "List" => serde_json::json!({"type": "array"}),
        "dict" | "Dict" => serde_json::json!({"type": "object"}),
        "None" | "NoneType" => serde_json::json!({"type": "null"}),
        _ => serde_json::json!({"type": "string"}), // Default to string for unknown types
    }
}

/// Check if a Python type hint is Optional
fn is_optional_type(_py: Python<'_>, type_hint: &Bound<'_, pyo3::PyAny>) -> bool {
    let type_repr = type_hint
        .repr()
        .and_then(|r| r.extract::<String>())
        .unwrap_or_default();

    // Check common Optional patterns
    type_repr.starts_with("typing.Optional[")
        || type_repr.contains(" | None")
        || type_repr.contains("None |")
        || type_repr.starts_with("typing.Union[") && type_repr.contains("NoneType")
}

/// Check if a class has a default value for a field
fn has_class_default(py: Python<'_>, class_obj: &PyObject, field_name: &str) -> bool {
    // Check if the class has an attribute with this name (default value)
    if let Ok(has_attr) = class_obj.bind(py).hasattr(field_name) {
        if has_attr {
            // Also check it's not just the annotation
            if let Ok(attr) = class_obj.getattr(py, field_name) {
                // If we can get the attribute, it has a default
                // Skip if it's a classmethod, property, or callable
                let attr_repr = attr
                    .bind(py)
                    .repr()
                    .and_then(|r| r.extract::<String>())
                    .unwrap_or_default();
                return !attr_repr.contains("method")
                    && !attr_repr.contains("property")
                    && !attr_repr.contains("function");
            }
        }
    }
    false
}

/// Parameters needed to lazily create a stream
#[derive(Clone)]
struct StreamParams {
    prompt: Option<String>,
    model: String,
    api_key: Option<String>,
    system: Option<String>,
    history: Option<Vec<reson_agentic::utils::ConversationMessage>>,
    output_type_name: Option<String>,
    output_schema: Option<serde_json::Value>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    max_tokens: Option<u32>,
}

/// Async iterator for streaming chunks
#[pyclass]
pub struct StreamIterator {
    // Stream is created lazily on first __anext__ call
    stream: Arc<tokio::sync::Mutex<Option<ChunkStream>>>,
    params: Arc<tokio::sync::Mutex<Option<StreamParams>>>,
    initialized: Arc<std::sync::atomic::AtomicBool>,
    raw_response_acc: Arc<RwLock<Vec<String>>>,
    reasoning_acc: Arc<RwLock<Vec<String>>>,
}

#[pymethods]
impl StreamIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        use futures::StreamExt;

        let stream = self.stream.clone();
        let params = self.params.clone();
        let initialized = self.initialized.clone();
        let raw_response_acc = self.raw_response_acc.clone();
        let reasoning_acc = self.reasoning_acc.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Initialize stream on first call
            if !initialized.load(std::sync::atomic::Ordering::SeqCst) {
                let mut params_guard = params.lock().await;
                if let Some(p) = params_guard.take() {
                    // Create the stream
                    let rust_tools: Arc<
                        RwLock<HashMap<String, reson_agentic::runtime::ToolFunction>>,
                    > = Arc::new(RwLock::new(HashMap::new()));
                    let rust_tool_schemas: Arc<
                        RwLock<HashMap<String, reson_agentic::runtime::ToolSchemaInfo>>,
                    > = Arc::new(RwLock::new(HashMap::new()));
                    let call_context: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>> =
                        Arc::new(RwLock::new(None));
                    let accumulators =
                        Arc::new(RwLock::new(reson_agentic::runtime::Accumulators::default()));

                    let new_stream = reson_agentic::runtime::inference::call_llm_stream(
                        p.prompt.as_deref(),
                        &p.model,
                        rust_tools,
                        rust_tool_schemas,
                        p.output_type_name,
                        p.output_schema,
                        p.api_key.as_deref(),
                        p.system.as_deref(),
                        p.history,
                        p.temperature,
                        p.top_p,
                        p.max_tokens,
                        None,
                        call_context,
                        accumulators,
                    )
                    .await
                    .map_err(to_py_err)?;

                    let mut stream_guard = stream.lock().await;
                    *stream_guard = Some(new_stream);
                    initialized.store(true, std::sync::atomic::Ordering::SeqCst);
                }
            }

            let mut guard = stream.lock().await;
            let stream_ref = guard.as_mut();

            match stream_ref {
                Some(s) => {
                    match s.next().await {
                        Some(Ok((chunk_type, chunk_value))) => {
                            // Update accumulators based on chunk type
                            match chunk_type.as_str() {
                                "content" => {
                                    if let Some(text) = chunk_value.as_str() {
                                        raw_response_acc.write().await.push(text.to_string());
                                    }
                                }
                                "reasoning" => {
                                    if let Some(text) = chunk_value.as_str() {
                                        reasoning_acc.write().await.push(text.to_string());
                                    }
                                }
                                _ => {}
                            }

                            // Return tuple (chunk_type, chunk_value)
                            Python::with_gil(|py| -> PyResult<PyObject> {
                                let py_value = pythonize::pythonize(py, &chunk_value)
                                    .map(|b| b.unbind())
                                    .unwrap_or_else(|_| py.None());
                                let tuple = (chunk_type, py_value).into_pyobject(py)?;
                                Ok(tuple.unbind().into())
                            })
                        }
                        Some(Err(e)) => Err(to_py_err(e)),
                        None => {
                            // Stream exhausted - raise StopAsyncIteration
                            Err(pyo3::exceptions::PyStopAsyncIteration::new_err(""))
                        }
                    }
                }
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err("")),
            }
        })
    }
}

/// Python wrapper for the Rust Runtime
#[pyclass]
pub struct Runtime {
    // We store Python tool functions separately since they can't go in the Rust Runtime
    tools: Arc<RwLock<HashMap<String, PyObject>>>,
    tool_types: Arc<RwLock<HashMap<String, PyObject>>>,
    tool_schemas: Arc<RwLock<HashMap<String, reson_agentic::runtime::ToolSchemaInfo>>>,
    model: Option<String>,
    api_key: Option<String>,
    native_tools: bool,
    used: Arc<RwLock<bool>>,
    // Accumulators
    raw_response: Arc<RwLock<Vec<String>>>,
    reasoning: Arc<RwLock<Vec<String>>>,
    reasoning_segments: Arc<RwLock<Vec<reson_agentic::types::ReasoningSegment>>>,
}

#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature = (model=None, store=None, used=false, api_key=None, native_tools=false))]
    fn new(
        model: Option<String>,
        store: Option<PyObject>,
        used: bool,
        api_key: Option<String>,
        native_tools: bool,
    ) -> PyResult<Self> {
        let _ = store; // Accept but ignore for now

        // Validate native_tools support
        if native_tools {
            if let Some(ref model_str) = model {
                if !crate::utils::supports_native_tools(model_str) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Native tools not supported for provider: {}",
                        model_str
                    )));
                }
            }
        }

        Ok(Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            tool_types: Arc::new(RwLock::new(HashMap::new())),
            tool_schemas: Arc::new(RwLock::new(HashMap::new())),
            model,
            api_key,
            native_tools,
            used: Arc::new(RwLock::new(used)),
            raw_response: Arc::new(RwLock::new(Vec::new())),
            reasoning: Arc::new(RwLock::new(Vec::new())),
            reasoning_segments: Arc::new(RwLock::new(Vec::new())),
        })
    }

    #[getter]
    fn raw_response(&self) -> String {
        let raw = self.raw_response.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            let acc = raw.read().await;
            acc.join("")
        })
    }

    #[getter]
    fn reasoning(&self) -> String {
        let reasoning = self.reasoning.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            let acc = reasoning.read().await;
            acc.join("")
        })
    }

    #[getter]
    fn reasoning_segments(&self) -> Vec<ReasoningSegment> {
        let segments = self.reasoning_segments.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            let acc = segments.read().await;
            acc.iter()
                .map(|s| ReasoningSegment::from(s.clone()))
                .collect()
        })
    }

    #[getter]
    fn native_tools(&self) -> bool {
        self.native_tools
    }

    #[getter]
    fn model(&self) -> Option<String> {
        self.model.clone()
    }

    /// Return the registered tools as a dict (for compatibility with tests)
    #[getter]
    fn _tools(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tools = self.tools.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let dict = PyDict::new(py);
        rt.block_on(async {
            let guard = tools.read().await;
            for (name, func) in guard.iter() {
                let _ = dict.set_item(name, func);
            }
        });
        Ok(dict.into())
    }

    /// Return the registered tool types as a dict (for compatibility with tests)
    #[getter]
    fn _tool_types(&self, py: Python<'_>) -> PyResult<PyObject> {
        let tool_types = self.tool_types.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let dict = PyDict::new(py);
        rt.block_on(async {
            let guard = tool_types.read().await;
            for (name, type_obj) in guard.iter() {
                let _ = dict.set_item(name, type_obj);
            }
        });
        Ok(dict.into())
    }

    fn clear_raw_response<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let raw = self.raw_response.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut acc = raw.write().await;
            acc.clear();
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn clear_reasoning<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let reasoning = self.reasoning.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut acc = reasoning.write().await;
            acc.clear();
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    fn clear_reasoning_segments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let segments = self.reasoning_segments.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut acc = segments.write().await;
            acc.clear();
            Ok::<_, PyErr>(Python::with_gil(|py| py.None()))
        })
    }

    /// Register a Python tool function
    #[pyo3(signature = (func, *, name=None, tool_type=None))]
    fn tool(
        &self,
        py: Python<'_>,
        func: PyObject,
        name: Option<String>,
        tool_type: Option<PyObject>,
    ) -> PyResult<()> {
        // Get function name if not provided
        let tool_name = match name {
            Some(n) => n,
            None => {
                let func_name = func.getattr(py, "__name__")?;
                func_name.extract::<String>(py)?
            }
        };

        // Attach tool_type to function as __reson_tool_type__ attribute
        // (Python implementation uses this for schema generation)
        if let Some(ref tt) = tool_type {
            func.setattr(py, "__reson_tool_type__", tt)?;
        }

        // Store the Python callable
        let tools = self.tools.clone();
        let tool_types = self.tool_types.clone();
        let tool_name_clone = tool_name.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            let mut tools_guard = tools.write().await;
            tools_guard.insert(tool_name, func);
        });

        // Store tool type if provided
        if let Some(tt) = tool_type {
            rt.block_on(async {
                let mut types_guard = tool_types.write().await;
                types_guard.insert(tool_name_clone, tt);
            });
        }

        Ok(())
    }

    /// Check if result is a tool call
    fn is_tool_call(&self, result: &Bound<'_, PyAny>) -> PyResult<bool> {
        // Check for _tool_name attribute or key
        if let Ok(dict) = result.downcast::<PyDict>() {
            return Ok(dict.get_item("_tool_name")?.is_some());
        }
        // Check for attribute
        if result.hasattr("_tool_name")? {
            return Ok(true);
        }
        Ok(false)
    }

    /// Get tool name from result
    fn get_tool_name(&self, result: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
        if let Ok(dict) = result.downcast::<PyDict>() {
            if let Some(name) = dict.get_item("_tool_name")? {
                return Ok(Some(name.extract()?));
            }
        }
        if result.hasattr("_tool_name")? {
            let name = result.getattr("_tool_name")?;
            return Ok(Some(name.extract()?));
        }
        Ok(None)
    }

    /// Execute a tool - calls the registered Python function
    ///
    /// If a `tool_type` was registered for this tool, the JSON args are hydrated
    /// into a typed instance (Pydantic, Deserializable, dataclass, or regular class)
    /// before being passed to the tool function.
    fn execute_tool<'py>(
        &self,
        py: Python<'py>,
        tool_result: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tools = self.tools.clone();
        let tool_types = self.tool_types.clone();
        let tool_result_clone = tool_result.clone_ref(py);

        // Get tool name
        let tool_name: String = {
            let bound = tool_result.bind(py);
            self.get_tool_name(bound)?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No _tool_name in result"))?
        };

        // Extract args dict from the tool result
        let args_dict: PyObject = {
            let bound = tool_result.bind(py);
            // First try to get 'args' attribute (ToolCall object)
            if let Ok(args) = bound.getattr("args") {
                args.unbind()
            // Then try to get it as a dict key
            } else if let Ok(dict) = bound.downcast::<pyo3::types::PyDict>() {
                if let Some(args) = dict.get_item("args")? {
                    args.unbind()
                } else {
                    // No args key, filter out metadata keys and use remaining as args
                    let filtered = pyo3::types::PyDict::new(py);
                    for (k, v) in dict.iter() {
                        let key: String = k.extract()?;
                        if !key.starts_with("_tool") {
                            filtered.set_item(k, v)?;
                        }
                    }
                    filtered.unbind().into()
                }
            } else {
                // Fallback: create empty dict
                pyo3::types::PyDict::new(py).unbind().into()
            }
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Look up the tool function
            let tool_fn = {
                let tools_guard = tools.read().await;
                let func = tools_guard.get(&tool_name).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Tool '{}' not found",
                        tool_name
                    ))
                })?;
                Python::with_gil(|py| func.clone_ref(py))
            };

            // Look up the tool type (if registered)
            let tool_type_opt = {
                let guard = tool_types.read().await;
                guard
                    .get(&tool_name)
                    .map(|t| Python::with_gil(|py| t.clone_ref(py)))
            };

            // Call the Python function with hydrated args or raw tool_result
            Python::with_gil(|py| -> PyResult<PyObject> {
                let asyncio = py.import("asyncio")?;

                // Hydrate args into typed instance if tool_type is registered
                let call_arg = if let Some(tool_type) = tool_type_opt {
                    let args_bound = args_dict.bind(py);

                    // Check if Pydantic V2 (has model_validate classmethod)
                    if tool_type.bind(py).hasattr("model_validate")? {
                        // Pydantic V2: MyClass.model_validate(args_dict)
                        tool_type.call_method1(py, "model_validate", (args_bound,))?
                    // Check if Pydantic V1 (has parse_obj classmethod)
                    } else if tool_type.bind(py).hasattr("parse_obj")? {
                        // Pydantic V1: MyClass.parse_obj(args_dict)
                        tool_type.call_method1(py, "parse_obj", (args_bound,))?
                    } else {
                        // Deserializable, dataclass, or regular class: MyClass(**args_dict)
                        let kwargs = args_bound.downcast::<pyo3::types::PyDict>()?;
                        tool_type.call(py, (), Some(kwargs))?
                    }
                } else {
                    // No type registered, pass raw tool_result
                    tool_result_clone.clone_ref(py)
                };

                let result = tool_fn.call1(py, (call_arg,))?;

                // Check if it's a coroutine and await it
                if asyncio
                    .call_method1("iscoroutine", (&result,))?
                    .is_truthy()?
                {
                    // For now, just return the coroutine and let Python handle it
                    Ok(result)
                } else {
                    Ok(result)
                }
            })
        })
    }

    /// Main inference call (async)
    #[pyo3(signature = (*, prompt=None, system=None, history=None, output_type=None, temperature=None, top_p=None, max_tokens=None, model=None, api_key=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        prompt: Option<String>,
        system: Option<String>,
        history: Option<&Bound<'_, PyList>>,
        output_type: Option<PyObject>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Mark as used
        let used = self.used.clone();
        let raw_response_acc = self.raw_response.clone();
        let reasoning_acc = self.reasoning.clone();

        // Get effective model
        let effective_model = model
            .or_else(|| self.model.clone())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No model specified"))?;

        let effective_api_key = api_key.or_else(|| self.api_key.clone());

        // Convert history to Rust types
        let rust_history: Option<Vec<reson_agentic::utils::ConversationMessage>> =
            if let Some(hist) = history {
                let mut messages = Vec::new();
                for item in hist.iter() {
                    // Try to convert each item
                    if let Ok(msg) = item.extract::<ChatMessage>() {
                        messages.push(reson_agentic::utils::ConversationMessage::Chat(msg.into()));
                    } else if let Ok(tr) = item.extract::<ToolResult>() {
                        messages.push(reson_agentic::utils::ConversationMessage::ToolResult(
                            tr.into(),
                        ));
                    } else if let Ok(rs) = item.extract::<ReasoningSegment>() {
                        messages.push(reson_agentic::utils::ConversationMessage::Reasoning(
                            rs.into(),
                        ));
                    }
                    // Skip items we can't convert
                }
                Some(messages)
            } else {
                None
            };

        // Extract output type name and schema if provided
        let (output_type_name, output_schema, output_type_clone): (
            Option<String>,
            Option<serde_json::Value>,
            Option<PyObject>,
        ) = if let Some(ref ot) = output_type {
            let type_name = ot
                .getattr(py, "__name__")
                .ok()
                .and_then(|n| n.extract(py).ok());

            // Try to generate JSON schema from the type
            let schema = generate_output_schema(py, ot);

            (type_name, schema, Some(ot.clone_ref(py)))
        } else {
            (None, None, None)
        };

        // Clone for move into async
        let prompt_clone = prompt.clone();
        let system_clone = system.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            // Mark as used
            *used.write().await = true;

            // Clear accumulators
            raw_response_acc.write().await.clear();
            reasoning_acc.write().await.clear();

            // Convert Python tools to Rust tool functions (empty for now - tools are Python side)
            let rust_tools: Arc<RwLock<HashMap<String, reson_agentic::runtime::ToolFunction>>> =
                Arc::new(RwLock::new(HashMap::new()));

            // Convert tool schemas
            let rust_tool_schemas: Arc<
                RwLock<HashMap<String, reson_agentic::runtime::ToolSchemaInfo>>,
            > = Arc::new(RwLock::new(HashMap::new()));

            // Create call context
            let call_context: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>> =
                Arc::new(RwLock::new(None));

            // Call the Rust inference engine
            let result = reson_agentic::runtime::inference::call_llm(
                prompt_clone.as_deref(),
                &effective_model,
                rust_tools,
                rust_tool_schemas,
                output_type_name,
                output_schema,
                effective_api_key.as_deref(),
                system_clone.as_deref(),
                rust_history,
                temperature,
                top_p,
                max_tokens,
                None,
                call_context,
            )
            .await
            .map_err(to_py_err)?;

            // Update accumulators
            if let Some(raw) = &result.raw_response {
                raw_response_acc.write().await.push(raw.clone());
            }
            if let Some(reasoning) = &result.reasoning {
                reasoning_acc.write().await.push(reasoning.clone());
            }

            // Convert result to Python, hydrating into output_type if provided
            Python::with_gil(|py| -> PyResult<PyObject> {
                let py_value = pythonize::pythonize(py, &result.parsed_value)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

                // If output_type was provided, hydrate the response into the typed instance
                if let Some(output_type) = output_type_clone {
                    // Check if the result is a dict (structured output)
                    if let Ok(dict) = py_value.downcast::<pyo3::types::PyDict>() {
                        // Check if Pydantic V2 (has model_validate classmethod)
                        if output_type.bind(py).hasattr("model_validate")? {
                            // Pydantic V2: MyClass.model_validate(dict)
                            return output_type.call_method1(py, "model_validate", (dict,));
                        // Check if Pydantic V1 (has parse_obj classmethod)
                        } else if output_type.bind(py).hasattr("parse_obj")? {
                            // Pydantic V1: MyClass.parse_obj(dict)
                            return output_type.call_method1(py, "parse_obj", (dict,));
                        } else {
                            // Deserializable, dataclass, or regular class: MyClass(**dict)
                            return output_type.call(py, (), Some(dict));
                        }
                    }
                }

                Ok(py_value.unbind())
            })
        })
    }

    /// Streaming inference call (async generator)
    #[pyo3(signature = (*, prompt=None, system=None, history=None, output_type=None, temperature=None, top_p=None, max_tokens=None, model=None, api_key=None))]
    fn run_stream<'py>(
        &self,
        py: Python<'py>,
        prompt: Option<String>,
        system: Option<String>,
        history: Option<&Bound<'_, PyList>>,
        output_type: Option<PyObject>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Get effective model
        let effective_model = model
            .or_else(|| self.model.clone())
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No model specified"))?;

        let effective_api_key = api_key.or_else(|| self.api_key.clone());

        // Convert history to Rust types
        let rust_history: Option<Vec<reson_agentic::utils::ConversationMessage>> =
            if let Some(hist) = history {
                let mut messages = Vec::new();
                for item in hist.iter() {
                    if let Ok(msg) = item.extract::<ChatMessage>() {
                        messages.push(reson_agentic::utils::ConversationMessage::Chat(msg.into()));
                    } else if let Ok(tr) = item.extract::<ToolResult>() {
                        messages.push(reson_agentic::utils::ConversationMessage::ToolResult(
                            tr.into(),
                        ));
                    } else if let Ok(rs) = item.extract::<ReasoningSegment>() {
                        messages.push(reson_agentic::utils::ConversationMessage::Reasoning(
                            rs.into(),
                        ));
                    }
                }
                Some(messages)
            } else {
                None
            };

        // Extract output type name and schema
        let (output_type_name, output_schema): (Option<String>, Option<serde_json::Value>) =
            if let Some(ref ot) = output_type {
                let type_name = ot
                    .getattr(py, "__name__")
                    .ok()
                    .and_then(|n| n.extract(py).ok());
                let schema = generate_output_schema(py, ot);
                (type_name, schema)
            } else {
                (None, None)
            };

        // Mark runtime as used synchronously
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        let used = self.used.clone();
        let raw_response_acc = self.raw_response.clone();
        let reasoning_acc = self.reasoning.clone();
        let reasoning_segments_acc = self.reasoning_segments.clone();

        rt.block_on(async {
            *used.write().await = true;
            raw_response_acc.write().await.clear();
            reasoning_acc.write().await.clear();
            reasoning_segments_acc.write().await.clear();
        });

        // Create StreamIterator with params for lazy stream initialization
        let params = StreamParams {
            prompt: prompt,
            model: effective_model,
            api_key: effective_api_key,
            system: system,
            history: rust_history,
            output_type_name,
            output_schema,
            temperature,
            top_p,
            max_tokens,
        };

        let iterator = StreamIterator {
            stream: Arc::new(tokio::sync::Mutex::new(None)),
            params: Arc::new(tokio::sync::Mutex::new(Some(params))),
            initialized: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            raw_response_acc: self.raw_response.clone(),
            reasoning_acc: self.reasoning.clone(),
        };

        // Return the iterator directly (it's an async iterable)
        let bound = Py::new(py, iterator)?;
        Ok(bound.into_pyobject(py)?.into_any())
    }
}

// Internal methods - called by decorator
impl Runtime {
    /// Create a new Runtime (for use from Rust code like decorators)
    pub fn create(model: Option<String>, api_key: Option<String>, native_tools: bool) -> Self {
        Self {
            tools: Arc::new(RwLock::new(HashMap::new())),
            tool_types: Arc::new(RwLock::new(HashMap::new())),
            tool_schemas: Arc::new(RwLock::new(HashMap::new())),
            model,
            api_key,
            native_tools,
            used: Arc::new(RwLock::new(false)),
            raw_response: Arc::new(RwLock::new(Vec::new())),
            reasoning: Arc::new(RwLock::new(Vec::new())),
            reasoning_segments: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn mark_used(&self) {
        let used = self.used.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async {
            *used.write().await = true;
        });
    }

    pub fn is_used(&self) -> bool {
        let used = self.used.clone();
        let rt = pyo3_async_runtimes::tokio::get_runtime();
        rt.block_on(async { *used.read().await })
    }
}
