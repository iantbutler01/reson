//! Utils module - schema generators and helper functions

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

fn resolve_provider_name(provider: &str) -> String {
    let parts: Vec<&str> = provider.split(':').collect();
    if parts.len() >= 2 && parts[1] == "resp" {
        match parts[0] {
            "openai" => "openai-responses".to_string(),
            "openrouter" => "openrouter-responses".to_string(),
            other => other.to_string(),
        }
    } else {
        parts.first().unwrap_or(&provider).to_string()
    }
}

/// Check if a provider supports native tools
#[pyfunction]
pub fn supports_native_tools(provider: &str) -> bool {
    let provider_name = resolve_provider_name(provider);

    match provider_name.as_str() {
        "openai" | "anthropic" | "google-gemini" | "google-genai" | "vertex-gemini"
        | "openrouter" | "bedrock" | "custom-openai" | "google-anthropic"
        | "openai-responses" | "openrouter-responses" => true,
        _ => false,
    }
}

/// Convert a Python type annotation to JSON schema type
fn python_type_to_json_schema(py: Python<'_>, type_obj: &Bound<'_, PyAny>) -> serde_json::Value {
    // Get type name
    let type_name = if let Ok(name) = type_obj.getattr("__name__") {
        name.extract::<String>().unwrap_or_default()
    } else if let Ok(origin) = type_obj.getattr("__origin__") {
        // Handle generic types like List[str], Optional[int]
        origin.getattr("__name__")
            .and_then(|n| n.extract::<String>())
            .unwrap_or_default()
    } else {
        type_obj.str().map(|s| s.to_string()).unwrap_or_default()
    };

    match type_name.as_str() {
        "str" => serde_json::json!({"type": "string"}),
        "int" => serde_json::json!({"type": "integer"}),
        "float" => serde_json::json!({"type": "number"}),
        "bool" => serde_json::json!({"type": "boolean"}),
        "list" | "List" => {
            // Try to get inner type from __args__
            let items = if let Ok(args) = type_obj.getattr("__args__") {
                if let Ok(tuple) = args.downcast::<PyTuple>() {
                    if let Some(inner) = tuple.get_item(0).ok() {
                        python_type_to_json_schema(py, &inner)
                    } else {
                        serde_json::json!({"type": "string"})
                    }
                } else {
                    serde_json::json!({"type": "string"})
                }
            } else {
                serde_json::json!({"type": "string"})
            };
            serde_json::json!({"type": "array", "items": items})
        }
        "dict" | "Dict" => {
            // Try to get value type from __args__ (Dict[K, V])
            if let Ok(args) = type_obj.getattr("__args__") {
                if let Ok(tuple) = args.downcast::<PyTuple>() {
                    // args[1] is the value type
                    if let Some(value_type) = tuple.get_item(1).ok() {
                        let value_schema = python_type_to_json_schema(py, &value_type);
                        return serde_json::json!({
                            "type": "object",
                            "additionalProperties": value_schema
                        });
                    }
                }
            }
            serde_json::json!({"type": "object", "additionalProperties": true})
        }
        _ => {
            // Check if it's a class with __annotations__ (like a Pydantic model or dataclass)
            if let Ok(annotations) = type_obj.getattr("__annotations__") {
                if let Ok(ann_dict) = annotations.downcast::<PyDict>() {
                    let mut properties = serde_json::Map::new();
                    let mut required = Vec::new();

                    for (key, value) in ann_dict.iter() {
                        let key_str: String = key.extract().unwrap_or_default();
                        let prop_schema = python_type_to_json_schema(py, &value);
                        properties.insert(key_str.clone(), prop_schema);
                        required.push(serde_json::Value::String(key_str));
                    }

                    return serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": required
                    });
                }
            }
            // Default to object
            serde_json::json!({"type": "object"})
        }
    }
}

/// Introspect a Python function and extract parameter schema
fn introspect_function(py: Python<'_>, func: &Bound<'_, PyAny>) -> PyResult<(String, serde_json::Value)> {
    // Get function name
    let _name: String = func.getattr("__name__")?.extract()?;

    // Check if there's a tool_type attached (takes priority for description)
    let description: String = if let Ok(tool_type) = func.getattr("__reson_tool_type__") {
        // Use tool_type's __doc__ if available
        tool_type.getattr("__doc__")
            .and_then(|d| d.extract::<Option<String>>())
            .unwrap_or(None)
            .unwrap_or_else(|| {
                // Fall back to function's __doc__
                func.getattr("__doc__")
                    .and_then(|d| d.extract::<Option<String>>())
                    .unwrap_or(None)
                    .unwrap_or_default()
            })
    } else {
        // No tool_type, use function's __doc__
        func.getattr("__doc__")
            .and_then(|d| d.extract::<Option<String>>())
            .unwrap_or(None)
            .unwrap_or_default()
    };

    // Get type annotations
    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    if let Ok(annotations) = func.getattr("__annotations__") {
        if let Ok(ann_dict) = annotations.downcast::<PyDict>() {
            for (key, value) in ann_dict.iter() {
                let key_str: String = key.extract()?;

                // Skip 'return' annotation and 'runtime' parameter
                if key_str == "return" || key_str == "runtime" {
                    continue;
                }

                let prop_schema = python_type_to_json_schema(py, &value);
                properties.insert(key_str.clone(), prop_schema);
                required.push(serde_json::Value::String(key_str));
            }
        }
    }

    let params_schema = serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required
    });

    Ok((description, params_schema))
}

/// Get a schema generator for the given provider
#[pyclass]
pub struct SchemaGenerator {
    provider: String,
}

#[pymethods]
impl SchemaGenerator {
    #[new]
    fn new(provider: String) -> Self {
        Self { provider }
    }

    /// Generate tool schemas for registered tools
    fn generate_tool_schemas(&self, py: Python<'_>, tools: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let mut schemas = Vec::new();

        for (name, func) in tools.iter() {
            let name_str: String = name.extract()?;

            // Introspect the function to get description and parameters
            let (description, params_schema) = introspect_function(py, &func)?;

            // Build provider-specific schema
            let schema = self.generate_schema_with_params(&name_str, &description, params_schema);
            schemas.push(schema);
        }

        // Wrap in provider-specific container format
        let provider_name = resolve_provider_name(&self.provider);
        let final_schemas = match provider_name.as_str() {
            "google-gemini" | "google-genai" | "vertex-gemini" => {
                // Google expects schemas wrapped in function_declarations
                vec![serde_json::json!({
                    "function_declarations": schemas
                })]
            }
            _ => schemas,
        };

        Ok(pythonize::pythonize(py, &final_schemas)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .unbind())
    }
}

impl SchemaGenerator {
    /// Generate schema with introspected parameters
    fn generate_schema_with_params(&self, name: &str, description: &str, params: serde_json::Value) -> serde_json::Value {
        let provider_name = resolve_provider_name(&self.provider);

        match provider_name.as_str() {
            "anthropic" | "bedrock" | "google-anthropic" => {
                serde_json::json!({
                    "name": name,
                    "description": description,
                    "input_schema": params,
                })
            }
            "openai" | "openrouter" | "custom-openai" => {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": params,
                    }
                })
            }
            "openai-responses" | "openrouter-responses" => {
                serde_json::json!({
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": params,
                })
            }
            "google-gemini" | "google-genai" | "vertex-gemini" => {
                serde_json::json!({
                    "name": name,
                    "description": description,
                    "parameters": params,
                })
            }
            _ => {
                serde_json::json!({
                    "name": name,
                    "description": description,
                    "parameters": params,
                })
            }
        }
    }
}

/// Get a schema generator for the given provider
#[pyfunction]
pub fn get_schema_generator(provider: &str) -> PyResult<SchemaGenerator> {
    if !supports_native_tools(provider) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Native tools not supported for provider: {}",
            provider
        )));
    }
    Ok(SchemaGenerator::new(provider.to_string()))
}

/// Generate native tool schemas (used by reson.reson module)
#[pyfunction]
pub fn _generate_native_tool_schemas(py: Python<'_>, tools: &Bound<'_, PyDict>, provider: &str) -> PyResult<PyObject> {
    let generator = get_schema_generator(provider)?;
    generator.generate_tool_schemas(py, tools)
}
