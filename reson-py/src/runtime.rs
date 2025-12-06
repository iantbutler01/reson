//! Python Runtime wrapper

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use futures::Stream;

use crate::errors::to_py_err;
use crate::stores::MemoryStore;
use crate::types::{ChatMessage, ReasoningSegment, ToolCall, ToolResult};

type ChunkStream = Pin<Box<dyn Stream<Item = Result<(String, serde_json::Value), reson_agentic::error::Error>> + Send>>;

/// Parameters needed to lazily create a stream
#[derive(Clone)]
struct StreamParams {
    prompt: Option<String>,
    model: String,
    api_key: Option<String>,
    system: Option<String>,
    history: Option<Vec<reson_agentic::utils::ConversationMessage>>,
    output_type_name: Option<String>,
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
                    let rust_tools: Arc<RwLock<HashMap<String, reson_agentic::runtime::ToolFunction>>> =
                        Arc::new(RwLock::new(HashMap::new()));
                    let rust_tool_schemas: Arc<RwLock<HashMap<String, reson_agentic::runtime::ToolSchemaInfo>>> =
                        Arc::new(RwLock::new(HashMap::new()));
                    let store: Arc<dyn reson_agentic::storage::Storage> =
                        Arc::new(reson_agentic::storage::MemoryStore::new());
                    let call_context: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>> =
                        Arc::new(RwLock::new(None));
                    let accumulators = Arc::new(RwLock::new(reson_agentic::runtime::Accumulators::default()));

                    let new_stream = reson_agentic::runtime::inference::call_llm_stream(
                        p.prompt.as_deref(),
                        &p.model,
                        rust_tools,
                        rust_tool_schemas,
                        p.output_type_name,
                        store,
                        p.api_key.as_deref(),
                        p.system.as_deref(),
                        p.history,
                        p.temperature,
                        p.top_p,
                        p.max_tokens,
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
    // Store
    store: Arc<RwLock<reson_agentic::storage::MemoryKVStore>>,
}

#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature = (model=None, store=None, used=false, api_key=None, native_tools=false))]
    fn new(
        model: Option<String>,
        store: Option<&MemoryStore>,
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
            store: Arc::new(RwLock::new(reson_agentic::storage::MemoryKVStore::new())),
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
    fn execute_tool<'py>(
        &self,
        py: Python<'py>,
        tool_result: PyObject,
    ) -> PyResult<Bound<'py, PyAny>> {
        let tools = self.tools.clone();
        let tool_result_clone = tool_result.clone_ref(py);

        // Get tool name
        let tool_name: String = {
            let bound = tool_result.bind(py);
            self.get_tool_name(bound)?.ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("No _tool_name in result")
            })?
        };

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let tool_fn = {
                let tools_guard = tools.read().await;
                let func = tools_guard.get(&tool_name).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!("Tool '{}' not found", tool_name))
                })?;
                Python::with_gil(|py| func.clone_ref(py))
            };

            // Call the Python function
            Python::with_gil(|py| -> PyResult<PyObject> {
                let asyncio = py.import("asyncio")?;
                let result = tool_fn.call1(py, (tool_result_clone.clone_ref(py),))?;

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
        let tools = self.tools.clone();
        let tool_schemas = self.tool_schemas.clone();

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
                        messages.push(reson_agentic::utils::ConversationMessage::ToolResult(tr.into()));
                    } else if let Ok(rs) = item.extract::<ReasoningSegment>() {
                        messages.push(reson_agentic::utils::ConversationMessage::Reasoning(rs.into()));
                    }
                    // Skip items we can't convert
                }
                Some(messages)
            } else {
                None
            };

        // Extract output type name if provided
        let output_type_name: Option<String> = if let Some(ref ot) = output_type {
            ot.getattr(py, "__name__")
                .ok()
                .and_then(|n| n.extract(py).ok())
        } else {
            None
        };

        // Create storage wrapper
        let store: Arc<dyn reson_agentic::storage::Storage> =
            Arc::new(reson_agentic::storage::MemoryStore::new());

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
            let rust_tool_schemas: Arc<RwLock<HashMap<String, reson_agentic::runtime::ToolSchemaInfo>>> =
                Arc::new(RwLock::new(HashMap::new()));

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
                store,
                effective_api_key.as_deref(),
                system_clone.as_deref(),
                rust_history,
                temperature,
                top_p,
                max_tokens,
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

            // Convert result to Python
            Python::with_gil(|py| -> PyResult<PyObject> {
                pythonize::pythonize(py, &result.parsed_value)
                    .map(|b| b.unbind())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
                        messages.push(reson_agentic::utils::ConversationMessage::ToolResult(tr.into()));
                    } else if let Ok(rs) = item.extract::<ReasoningSegment>() {
                        messages.push(reson_agentic::utils::ConversationMessage::Reasoning(rs.into()));
                    }
                }
                Some(messages)
            } else {
                None
            };

        // Extract output type name
        let output_type_name: Option<String> = if let Some(ref ot) = output_type {
            ot.getattr(py, "__name__")
                .ok()
                .and_then(|n| n.extract(py).ok())
        } else {
            None
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
            store: Arc::new(RwLock::new(reson_agentic::storage::MemoryKVStore::new())),
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
