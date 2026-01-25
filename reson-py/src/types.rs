//! Python type wrappers for reson core types

use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use once_cell::sync::Lazy;

// Global registry to store original Python objects for ToolCall.tool_obj
// Key is a unique ID, value is the Python object
static TOOL_OBJ_REGISTRY: Lazy<Arc<RwLock<HashMap<String, PyObject>>>> =
    Lazy::new(|| Arc::new(RwLock::new(HashMap::new())));

/// Chat role enum exposed to Python
#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum ChatRole {
    #[pyo3(name = "SYSTEM")]
    System,
    #[pyo3(name = "USER")]
    User,
    #[pyo3(name = "ASSISTANT")]
    Assistant,
    #[pyo3(name = "TOOL")]
    Tool,
}

#[pymethods]
impl ChatRole {
    #[getter]
    fn value(&self) -> &'static str {
        match self {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        }
    }

    fn __repr__(&self) -> String {
        format!("ChatRole.{}", match self {
            ChatRole::System => "SYSTEM",
            ChatRole::User => "USER",
            ChatRole::Assistant => "ASSISTANT",
            ChatRole::Tool => "TOOL",
        })
    }
}

impl From<reson_agentic::types::ChatRole> for ChatRole {
    fn from(role: reson_agentic::types::ChatRole) -> Self {
        match role {
            reson_agentic::types::ChatRole::System => ChatRole::System,
            reson_agentic::types::ChatRole::User => ChatRole::User,
            reson_agentic::types::ChatRole::Assistant => ChatRole::Assistant,
            reson_agentic::types::ChatRole::Tool => ChatRole::Tool,
        }
    }
}

impl From<ChatRole> for reson_agentic::types::ChatRole {
    fn from(role: ChatRole) -> Self {
        match role {
            ChatRole::System => reson_agentic::types::ChatRole::System,
            ChatRole::User => reson_agentic::types::ChatRole::User,
            ChatRole::Assistant => reson_agentic::types::ChatRole::Assistant,
            ChatRole::Tool => reson_agentic::types::ChatRole::Tool,
        }
    }
}

/// Chat message exposed to Python
#[pyclass]
#[derive(Clone)]
pub struct ChatMessage {
    #[pyo3(get, set)]
    pub role: ChatRole,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub cache_marker: bool,
    #[pyo3(get, set)]
    pub signature: Option<String>,
}

#[pymethods]
impl ChatMessage {
    #[new]
    #[pyo3(signature = (role, content, cache_marker=false, signature=None))]
    fn new(role: ChatRole, content: String, cache_marker: bool, signature: Option<String>) -> Self {
        Self { role, content, cache_marker, signature }
    }

    #[staticmethod]
    fn user(content: String) -> Self {
        Self { role: ChatRole::User, content, cache_marker: false, signature: None }
    }

    #[staticmethod]
    fn assistant(content: String) -> Self {
        Self { role: ChatRole::Assistant, content, cache_marker: false, signature: None }
    }

    #[staticmethod]
    fn system(content: String) -> Self {
        Self { role: ChatRole::System, content, cache_marker: false, signature: None }
    }

    fn model_dump(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("role", self.role.value())?;
        dict.set_item("content", &self.content)?;
        dict.set_item("cache_marker", self.cache_marker)?;
        if let Some(ref sig) = self.signature {
            dict.set_item("signature", sig)?;
        }
        Ok(dict.into())
    }

    #[staticmethod]
    fn model_validate(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let role_str: String = data.get_item("role")?.unwrap().extract()?;
        let role = match role_str.as_str() {
            "system" => ChatRole::System,
            "user" => ChatRole::User,
            "assistant" => ChatRole::Assistant,
            "tool" => ChatRole::Tool,
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid role: {}", role_str))),
        };
        let content: String = data.get_item("content")?.unwrap().extract()?;
        let cache_marker: bool = data.get_item("cache_marker")
            .ok()
            .flatten()
            .map(|v| v.extract().unwrap_or(false))
            .unwrap_or(false);
        let signature: Option<String> = data.get_item("signature")
            .ok()
            .flatten()
            .and_then(|v| v.extract().ok());

        Ok(Self { role, content, cache_marker, signature })
    }

    fn __repr__(&self) -> String {
        format!("ChatMessage(role={:?}, content={:?})", self.role.value(), self.content)
    }
}

impl From<reson_agentic::types::ChatMessage> for ChatMessage {
    fn from(msg: reson_agentic::types::ChatMessage) -> Self {
        Self {
            role: msg.role.into(),
            content: msg.content,
            cache_marker: msg.cache_marker.is_some(),
            signature: msg.signature,
        }
    }
}

impl From<ChatMessage> for reson_agentic::types::ChatMessage {
    fn from(msg: ChatMessage) -> Self {
        Self {
            role: msg.role.into(),
            content: msg.content,
            cache_marker: if msg.cache_marker { Some(reson_agentic::types::CacheMarker::Ephemeral) } else { None },
            model_families: None,
            signature: msg.signature,
        }
    }
}

/// Tool call exposed to Python
#[pyclass]
#[derive(Clone)]
pub struct ToolCall {
    #[pyo3(get, set)]
    pub tool_use_id: String,
    #[pyo3(get, set)]
    pub tool_name: String,
    #[pyo3(get, set)]
    pub raw_arguments: Option<String>,
    #[pyo3(get, set)]
    pub signature: Option<String>,
    // args stored as JSON string internally
    args_json: String,
    // Original tool call object for preserving provider format (JSON fallback)
    tool_obj_json: Option<String>,
    // Registry key for the original Python object (if stored)
    tool_obj_registry_key: Option<String>,
}

#[pymethods]
impl ToolCall {
    #[new]
    #[pyo3(signature = (tool_use_id, tool_name, args=None, raw_arguments=None, signature=None, tool_obj=None))]
    fn new(
        tool_use_id: String,
        tool_name: String,
        args: Option<&Bound<'_, PyAny>>,
        raw_arguments: Option<String>,
        signature: Option<String>,
        tool_obj: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let args_json = if let Some(any) = args {
            pythonize::depythonize::<serde_json::Value>(any)
                .map(|v| serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()))
                .unwrap_or_else(|_| "{}".to_string())
        } else {
            "{}".to_string()
        };

        let tool_obj_json = if let Some(obj) = tool_obj {
            pythonize::depythonize::<serde_json::Value>(obj)
                .ok()
                .and_then(|v| serde_json::to_string(&v).ok())
        } else {
            None
        };

        Ok(Self {
            tool_use_id,
            tool_name,
            args_json,
            raw_arguments,
            signature,
            tool_obj_json,
            tool_obj_registry_key: None,
        })
    }

    /// Factory method to create ToolCall(s) from provider-format tool call objects.
    /// Automatically detects the provider format and parses accordingly.
    ///
    /// Returns a single ToolCall or list of ToolCalls depending on input.
    #[staticmethod]
    #[pyo3(signature = (tool_call_obj_or_list, signature=None))]
    fn create(py: Python<'_>, tool_call_obj_or_list: &Bound<'_, PyAny>, signature: Option<String>) -> PyResult<PyObject> {
        // Store original Python object in registry before conversion
        let registry_key = uuid::Uuid::new_v4().to_string();
        {
            let mut registry = TOOL_OBJ_REGISTRY.write().unwrap();
            registry.insert(registry_key.clone(), tool_call_obj_or_list.clone().unbind());
        }

        // Try to convert Python object to JSON
        // If direct conversion fails and it's an object with __dict__, extract attributes
        let json_value: serde_json::Value = if let Ok(val) = pythonize::depythonize(tool_call_obj_or_list) {
            val
        } else if tool_call_obj_or_list.hasattr("__dict__")? {
            // It's a Python object with attributes - convert __dict__ to JSON
            let dict = tool_call_obj_or_list.getattr("__dict__")?;
            pythonize::depythonize(&dict)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Failed to convert object: {}", e)))?
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Failed to convert input: unsupported type"
            ));
        };

        // Call Rust create function
        let result = reson_agentic::types::ToolCall::create(json_value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Unsupported tool call format: {}", e)))?;

        // Convert result back to Python
        match result {
            reson_agentic::types::CreateResult::Single(mut tc) => {
                // Apply signature override if provided
                if signature.is_some() {
                    tc.signature = signature;
                }
                let mut py_tc: ToolCall = tc.into();
                // Store registry key so we can retrieve original object
                py_tc.tool_obj_registry_key = Some(registry_key);
                Ok(py_tc.into_pyobject(py)?.unbind().into_any())
            }
            reson_agentic::types::CreateResult::Multiple(tcs) => {
                let py_list: Vec<ToolCall> = tcs.into_iter().map(|mut tc| {
                    if signature.is_some() {
                        tc.signature = signature.clone();
                    }
                    let mut py_tc: ToolCall = tc.into();
                    py_tc.tool_obj_registry_key = Some(registry_key.clone());
                    py_tc
                }).collect();
                Ok(py_list.into_pyobject(py)?.unbind().into_any())
            }
            reson_agentic::types::CreateResult::Empty => {
                Err(pyo3::exceptions::PyValueError::new_err("No tool calls provided"))
            }
        }
    }

    #[getter]
    fn args(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value: serde_json::Value = serde_json::from_str(&self.args_json)
            .unwrap_or(serde_json::Value::Object(Default::default()));
        pythonize::pythonize(py, &value)
            .map(|bound| bound.unbind())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    #[setter]
    fn set_args(&mut self, args: &Bound<'_, PyAny>) -> PyResult<()> {
        let value: serde_json::Value = pythonize::depythonize(args)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        self.args_json = serde_json::to_string(&value)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    #[getter]
    fn tool_obj(&self, py: Python<'_>) -> PyResult<PyObject> {
        // First check the registry for the original Python object
        if let Some(ref key) = self.tool_obj_registry_key {
            let registry = TOOL_OBJ_REGISTRY.read().unwrap();
            if let Some(obj) = registry.get(key) {
                return Ok(obj.clone_ref(py));
            }
        }
        // Fall back to JSON representation
        if let Some(ref json_str) = self.tool_obj_json {
            let value: serde_json::Value = serde_json::from_str(json_str)
                .unwrap_or(serde_json::Value::Null);
            pythonize::pythonize(py, &value)
                .map(|bound| bound.unbind())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
        } else {
            Ok(py.None())
        }
    }

    #[setter]
    fn set_tool_obj(&mut self, tool_obj: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.tool_obj_json = if let Some(obj) = tool_obj {
            pythonize::depythonize::<serde_json::Value>(obj)
                .ok()
                .and_then(|v| serde_json::to_string(&v).ok())
        } else {
            None
        };
        Ok(())
    }

    /// Convert to provider-specific assistant message format.
    fn to_provider_assistant_message(&self, py: Python<'_>, provider: &crate::services::InferenceProvider) -> PyResult<PyObject> {
        // Convert to Rust type, call method, convert back
        let rust_provider: reson_agentic::types::Provider = (*provider).into();
        let rust_tc: reson_agentic::types::ToolCall = self.clone().into();
        let result = rust_tc.to_provider_assistant_message(rust_provider);

        pythonize::pythonize(py, &result)
            .map(|bound| bound.unbind())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("tool_use_id", &self.tool_use_id)?;
        dict.set_item("tool_call_id", &self.tool_use_id)?;  // Alias for compatibility
        dict.set_item("tool_name", &self.tool_name)?;
        dict.set_item("args", self.args(py)?)?;
        if let Some(ref raw) = self.raw_arguments {
            dict.set_item("raw_arguments", raw)?;
        }
        if let Some(ref sig) = self.signature {
            dict.set_item("signature", sig)?;
        }
        if let Some(ref obj_json) = self.tool_obj_json {
            let value: serde_json::Value = serde_json::from_str(obj_json)
                .unwrap_or(serde_json::Value::Null);
            let py_val = pythonize::pythonize(py, &value)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            dict.set_item("tool_obj", py_val)?;
        }
        Ok(dict.into())
    }

    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let tool_use_id: String = data.get_item("tool_use_id")?
            .or_else(|| data.get_item("tool_call_id").ok().flatten())
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("tool_use_id or tool_call_id required"))?
            .extract()?;
        let tool_name: String = data.get_item("tool_name")?.unwrap().extract()?;
        let args: Option<Bound<'_, PyAny>> = data.get_item("args")?;
        let raw_arguments: Option<String> = data.get_item("raw_arguments")
            .ok().flatten().and_then(|v| v.extract().ok());
        let signature: Option<String> = data.get_item("signature")
            .ok().flatten().and_then(|v| v.extract().ok());
        let tool_obj: Option<Bound<'_, PyAny>> = data.get_item("tool_obj").ok().flatten();

        Self::new(tool_use_id, tool_name, args.as_ref(), raw_arguments, signature, tool_obj.as_ref())
    }

    fn __repr__(&self) -> String {
        format!("ToolCall(tool_name={:?}, tool_use_id={:?})", self.tool_name, self.tool_use_id)
    }
}

impl From<reson_agentic::types::ToolCall> for ToolCall {
    fn from(tc: reson_agentic::types::ToolCall) -> Self {
        Self {
            tool_use_id: tc.tool_use_id,
            tool_name: tc.tool_name,
            args_json: serde_json::to_string(&tc.args).unwrap_or_else(|_| "{}".to_string()),
            raw_arguments: tc.raw_arguments,
            signature: tc.signature,
            tool_obj_json: tc.tool_obj.and_then(|v| serde_json::to_string(&v).ok()),
            tool_obj_registry_key: None,  // Set separately after conversion when needed
        }
    }
}

impl From<ToolCall> for reson_agentic::types::ToolCall {
    fn from(tc: ToolCall) -> Self {
        Self {
            tool_use_id: tc.tool_use_id,
            tool_name: tc.tool_name,
            args: serde_json::from_str(&tc.args_json).unwrap_or(serde_json::Value::Object(Default::default())),
            raw_arguments: tc.raw_arguments,
            signature: tc.signature,
            tool_obj: tc.tool_obj_json.and_then(|s| serde_json::from_str(&s).ok()),
        }
    }
}

/// Tool result exposed to Python
#[pyclass]
#[derive(Clone)]
pub struct ToolResult {
    #[pyo3(get, set)]
    pub tool_use_id: String,
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub is_error: bool,
    #[pyo3(get, set)]
    pub signature: Option<String>,
    #[pyo3(get, set)]
    pub tool_name: Option<String>,
}

#[pymethods]
impl ToolResult {
    #[new]
    #[pyo3(signature = (tool_use_id, content, is_error=false, signature=None, tool_name=None))]
    fn new(
        tool_use_id: String,
        content: String,
        is_error: bool,
        signature: Option<String>,
        tool_name: Option<String>,
    ) -> Self {
        Self { tool_use_id, content, is_error, signature, tool_name }
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("tool_use_id", &self.tool_use_id)?;
        dict.set_item("content", &self.content)?;
        dict.set_item("is_error", self.is_error)?;
        if let Some(ref sig) = self.signature {
            dict.set_item("signature", sig)?;
        }
        if let Some(ref name) = self.tool_name {
            dict.set_item("tool_name", name)?;
        }
        Ok(dict.into())
    }

    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let tool_use_id: String = data.get_item("tool_use_id")?.unwrap().extract()?;
        let content: String = data.get_item("content")?.unwrap().extract()?;
        let is_error: bool = data.get_item("is_error")
            .ok().flatten().map(|v| v.extract().unwrap_or(false)).unwrap_or(false);
        let signature: Option<String> = data.get_item("signature")
            .ok().flatten().and_then(|v| v.extract().ok());
        let tool_name: Option<String> = data.get_item("tool_name")
            .ok().flatten().and_then(|v| v.extract().ok());

        Ok(Self { tool_use_id, content, is_error, signature, tool_name })
    }

    fn __repr__(&self) -> String {
        format!("ToolResult(tool_use_id={:?}, is_error={})", self.tool_use_id, self.is_error)
    }
}

impl From<reson_agentic::types::ToolResult> for ToolResult {
    fn from(tr: reson_agentic::types::ToolResult) -> Self {
        Self {
            tool_use_id: tr.tool_use_id,
            content: tr.content,
            is_error: tr.is_error,
            signature: tr.signature,
            tool_name: tr.tool_name,
        }
    }
}

impl From<ToolResult> for reson_agentic::types::ToolResult {
    fn from(tr: ToolResult) -> Self {
        Self {
            tool_use_id: tr.tool_use_id,
            tool_name: tr.tool_name,
            content: tr.content,
            is_error: tr.is_error,
            signature: tr.signature,
            tool_obj: None,
        }
    }
}

/// Reasoning segment exposed to Python
#[pyclass]
#[derive(Clone)]
pub struct ReasoningSegment {
    #[pyo3(get, set)]
    pub content: String,
    #[pyo3(get, set)]
    pub signature: Option<String>,
    #[pyo3(get, set)]
    pub segment_index: usize,
    // provider_metadata stored as JSON
    metadata_json: String,
}

#[pymethods]
impl ReasoningSegment {
    #[new]
    #[pyo3(signature = (content, signature=None, provider_metadata=None, segment_index=0))]
    fn new(
        content: String,
        signature: Option<String>,
        provider_metadata: Option<&Bound<'_, PyDict>>,
        segment_index: usize,
    ) -> PyResult<Self> {
        let metadata_json = if let Some(dict) = provider_metadata {
            pythonize::depythonize::<serde_json::Value>(dict)
                .map(|v| serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()))
                .unwrap_or_else(|_| "{}".to_string())
        } else {
            "{}".to_string()
        };

        Ok(Self { content, signature, segment_index, metadata_json })
    }

    #[getter]
    fn provider_metadata(&self, py: Python<'_>) -> PyResult<PyObject> {
        let value: serde_json::Value = serde_json::from_str(&self.metadata_json)
            .unwrap_or(serde_json::Value::Object(Default::default()));
        pythonize::pythonize(py, &value)
            .map(|bound| bound.unbind())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let preview: String = self.content.chars().take(50).collect();
        format!("ReasoningSegment(content={:?}...)", preview)
    }

    /// Convert to provider-specific format
    fn to_provider_format(&self, py: Python<'_>, provider: &crate::services::InferenceProvider) -> PyResult<PyObject> {
        let dict = PyDict::new(py);

        // Parse existing metadata
        let metadata: serde_json::Value = serde_json::from_str(&self.metadata_json)
            .unwrap_or(serde_json::Value::Object(Default::default()));

        match provider {
            crate::services::InferenceProvider::ANTHROPIC
            | crate::services::InferenceProvider::BEDROCK
            | crate::services::InferenceProvider::GOOGLE_ANTHROPIC => {
                dict.set_item("type", "thinking")?;
                dict.set_item("thinking", &self.content)?;
                if let Some(ref sig) = self.signature {
                    dict.set_item("signature", sig)?;
                }
                // Add any extra metadata
                if let serde_json::Value::Object(map) = metadata {
                    for (k, v) in map {
                        let py_val = pythonize::pythonize(py, &v)
                            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                        dict.set_item(k, py_val)?;
                    }
                }
            }
            crate::services::InferenceProvider::OPENAI
            | crate::services::InferenceProvider::OPENAI_RESPONSES
            | crate::services::InferenceProvider::OPENROUTER
            | crate::services::InferenceProvider::OPENROUTER_RESPONSES => {
                dict.set_item("type", "reasoning")?;
                dict.set_item("content", &self.content)?;
                if let Some(ref sig) = self.signature {
                    dict.set_item("signature", sig)?;
                }
            }
            crate::services::InferenceProvider::GOOGLE_GENAI => {
                dict.set_item("thought", true)?;
                dict.set_item("text", &self.content)?;
                if let Some(ref sig) = self.signature {
                    dict.set_item("thought_signature", sig)?;
                }
            }
        }

        Ok(dict.into())
    }
}

impl From<reson_agentic::types::ReasoningSegment> for ReasoningSegment {
    fn from(rs: reson_agentic::types::ReasoningSegment) -> Self {
        Self {
            content: rs.content,
            signature: rs.signature,
            segment_index: rs.segment_index,
            metadata_json: rs.provider_metadata
                .map(|v| serde_json::to_string(&v).unwrap_or_else(|_| "{}".to_string()))
                .unwrap_or_else(|| "{}".to_string()),
        }
    }
}

impl From<ReasoningSegment> for reson_agentic::types::ReasoningSegment {
    fn from(rs: ReasoningSegment) -> Self {
        Self {
            content: rs.content,
            signature: rs.signature,
            segment_index: rs.segment_index,
            provider_metadata: serde_json::from_str(&rs.metadata_json).ok(),
        }
    }
}

/// Marker class for types that can be deserialized from tool call responses.
/// In the Rust implementation, this is implemented as a trait marker.
#[pyclass(subclass)]
#[derive(Clone)]
pub struct Deserializable;

#[pymethods]
impl Deserializable {
    #[new]
    fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        "Deserializable()".to_string()
    }
}
