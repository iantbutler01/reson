//! Type introspection utilities for schema generation
//!
//! Provides helpers to extract type information and generate JSON schemas
//! from Rust types at compile-time or runtime.

use serde_json::{json, Value};
use std::collections::HashMap;

/// Extract parameter schema from a type name string
///
/// This is a basic implementation that maps common type names to JSON schema.
/// For full introspection, we'd need proc macros or reflection.
pub fn type_name_to_schema(type_name: &str) -> Value {
    match type_name {
        "String" | "str" | "&str" => json!({
            "type": "string",
            "description": "A string value"
        }),
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => json!({
            "type": "integer",
            "description": "An integer value"
        }),
        "f32" | "f64" => json!({
            "type": "number",
            "description": "A floating point number"
        }),
        "bool" => json!({
            "type": "boolean",
            "description": "A boolean value"
        }),
        s if s.starts_with("Vec<") || s.starts_with("&[") => {
            // Extract inner type
            let inner = if s.starts_with("Vec<") {
                s.strip_prefix("Vec<")
                    .and_then(|s| s.strip_suffix(">"))
                    .unwrap_or("String")
            } else {
                s.strip_prefix("&[")
                    .and_then(|s| s.strip_suffix("]"))
                    .unwrap_or("String")
            };

            json!({
                "type": "array",
                "items": type_name_to_schema(inner),
                "description": format!("An array of {}", inner)
            })
        },
        s if s.starts_with("Option<") => {
            // Extract inner type - optional fields are not required
            let inner = s.strip_prefix("Option<")
                .and_then(|s| s.strip_suffix(">"))
                .unwrap_or("String");
            type_name_to_schema(inner)
        },
        s if s.starts_with("HashMap<") || s.starts_with("BTreeMap<") => {
            json!({
                "type": "object",
                "additionalProperties": true,
                "description": "A map of key-value pairs"
            })
        },
        _ => {
            // Unknown type - assume it's a custom struct/object
            json!({
                "type": "object",
                "description": format!("An object of type {}", type_name)
            })
        }
    }
}

/// Build a tool schema from parameter information
///
/// # Arguments
/// * `name` - Tool name
/// * `description` - Tool description
/// * `parameters` - Map of parameter names to their type names
/// * `required` - List of required parameter names
pub fn build_tool_schema(
    name: &str,
    description: &str,
    parameters: &HashMap<String, String>,
    required: &[String],
) -> Value {
    let mut properties = serde_json::Map::new();

    for (param_name, type_name) in parameters {
        let schema = type_name_to_schema(type_name);
        properties.insert(param_name.clone(), schema);
    }

    json!({
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    })
}

/// Extract description from doc comments
///
/// This is a placeholder - in a real implementation, we'd use proc macros
/// to extract actual doc comments at compile time.
pub fn extract_doc_comment(_item_name: &str) -> Option<String> {
    // TODO: Implement with proc macros
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_schema() {
        let schema = type_name_to_schema("String");
        assert_eq!(schema["type"], "string");
        assert!(schema["description"].is_string());
    }

    #[test]
    fn test_integer_schema() {
        let schema = type_name_to_schema("i32");
        assert_eq!(schema["type"], "integer");
    }

    #[test]
    fn test_float_schema() {
        let schema = type_name_to_schema("f64");
        assert_eq!(schema["type"], "number");
    }

    #[test]
    fn test_bool_schema() {
        let schema = type_name_to_schema("bool");
        assert_eq!(schema["type"], "boolean");
    }

    #[test]
    fn test_vec_schema() {
        let schema = type_name_to_schema("Vec<String>");
        assert_eq!(schema["type"], "array");
        assert_eq!(schema["items"]["type"], "string");
    }

    #[test]
    fn test_option_schema() {
        let schema = type_name_to_schema("Option<i32>");
        assert_eq!(schema["type"], "integer");
    }

    #[test]
    fn test_hashmap_schema() {
        let schema = type_name_to_schema("HashMap<String, i32>");
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["additionalProperties"], true);
    }

    #[test]
    fn test_build_tool_schema() {
        let mut params = HashMap::new();
        params.insert("name".to_string(), "String".to_string());
        params.insert("age".to_string(), "i32".to_string());
        params.insert("emails".to_string(), "Vec<String>".to_string());

        let required = vec!["name".to_string(), "age".to_string()];

        let schema = build_tool_schema(
            "create_person",
            "Create a new person",
            &params,
            &required,
        );

        assert_eq!(schema["name"], "create_person");
        assert_eq!(schema["description"], "Create a new person");
        assert!(schema["parameters"]["properties"].is_object());
        assert_eq!(schema["parameters"]["properties"]["name"]["type"], "string");
        assert_eq!(schema["parameters"]["properties"]["age"]["type"], "integer");
        assert_eq!(schema["parameters"]["properties"]["emails"]["type"], "array");
        assert_eq!(schema["parameters"]["required"][0], "name");
        assert_eq!(schema["parameters"]["required"][1], "age");
    }
}
