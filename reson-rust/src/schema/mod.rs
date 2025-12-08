//! Schema generation for tools and structured outputs
//!
//! Converts Rust types to provider-specific tool schemas and output schemas.

use serde_json::Value;
use crate::error::{Error, Result};

/// Fix output schema for provider-specific requirements
///
/// Different providers have different JSON Schema requirements:
/// - OpenAI: Requires `additionalProperties: false` and all properties in `required`
/// - Google: Does NOT support `additionalProperties`, uses standard JSON Schema
/// - Anthropic: Similar to OpenAI but with different structured output format
///
/// This function modifies the schema in-place to match provider requirements.
pub fn fix_output_schema_for_provider(schema: &mut Value, provider: &str) {
    match provider {
        "openai" | "openrouter" => fix_schema_for_openai(schema),
        "anthropic" | "bedrock" | "google-anthropic" | "vertexai" => fix_schema_for_anthropic(schema),
        "google" | "google_gemini" | "vertex_gemini" | "gemini" | "google-genai" | "google-gemini" => {
            fix_schema_for_google(schema)
        }
        _ => {
            // Default: apply OpenAI-style fixes (most restrictive)
            fix_schema_for_openai(schema)
        }
    }
}

/// Fix schema for OpenAI's strict structured outputs
/// - Add `additionalProperties: false` to all objects
/// - Make all properties required (OpenAI requires this)
fn fix_schema_for_openai(schema: &mut Value) {
    if let Value::Object(map) = schema {
        // If this is an object type with properties
        if map.get("type") == Some(&Value::String("object".to_string())) {
            // Add additionalProperties: false
            map.insert("additionalProperties".to_string(), Value::Bool(false));

            // Make all properties required (OpenAI strict mode requirement)
            if let Some(Value::Object(props)) = map.get("properties") {
                let all_keys: Vec<Value> = props.keys()
                    .map(|k| Value::String(k.clone()))
                    .collect();
                map.insert("required".to_string(), Value::Array(all_keys));
            }
        }

        // Recurse into nested schemas
        recurse_schema_fix(schema, fix_schema_for_openai);
    }
}

/// Fix schema for Anthropic's structured outputs
/// - Add `additionalProperties: false` to all objects
/// - Keep original required fields (Anthropic doesn't require all fields)
fn fix_schema_for_anthropic(schema: &mut Value) {
    if let Value::Object(map) = schema {
        // If this is an object type with properties
        if map.get("type") == Some(&Value::String("object".to_string())) {
            // Add additionalProperties: false
            map.insert("additionalProperties".to_string(), Value::Bool(false));
        }

        // Recurse into nested schemas
        recurse_schema_fix(schema, fix_schema_for_anthropic);
    }
}

/// Fix schema for Google's structured outputs
/// - Remove `additionalProperties` (Google doesn't support it)
/// - Remove `$defs` and inline references (Google doesn't support $ref)
/// - Keep original required fields
fn fix_schema_for_google(schema: &mut Value) {
    // First, inline any $defs references
    inline_schema_refs(schema);

    if let Value::Object(map) = schema {
        // Remove additionalProperties (Google doesn't support it)
        map.remove("additionalProperties");

        // Remove $defs (already inlined)
        map.remove("$defs");
        map.remove("definitions");

        // Remove title (Google may not need it)
        // map.remove("title"); // Keep title, it's informative

        // Recurse into nested schemas
        recurse_schema_fix(schema, fix_schema_for_google);
    }
}

/// Inline $ref references by copying from $defs
fn inline_schema_refs(schema: &mut Value) {
    // Extract $defs if present
    let defs = if let Value::Object(map) = schema {
        map.get("$defs").cloned()
            .or_else(|| map.get("definitions").cloned())
    } else {
        None
    };

    if let Some(defs) = defs {
        inline_refs_recursive(schema, &defs);
    }
}

/// Recursively replace $ref with actual definitions
fn inline_refs_recursive(schema: &mut Value, defs: &Value) {
    match schema {
        Value::Object(map) => {
            // Check if this object has a $ref
            if let Some(Value::String(ref_path)) = map.get("$ref").cloned() {
                // Extract definition name from "#/$defs/Name" or "#/definitions/Name"
                let def_name = ref_path
                    .strip_prefix("#/$defs/")
                    .or_else(|| ref_path.strip_prefix("#/definitions/"));

                if let Some(name) = def_name {
                    if let Some(def) = defs.get(name) {
                        // Replace the entire object with the definition
                        *schema = def.clone();
                        // Recurse into the inlined definition
                        inline_refs_recursive(schema, defs);
                        return;
                    }
                }
            }

            // Recurse into all values
            for (_, v) in map.iter_mut() {
                inline_refs_recursive(v, defs);
            }
        }
        Value::Array(arr) => {
            for item in arr.iter_mut() {
                inline_refs_recursive(item, defs);
            }
        }
        _ => {}
    }
}

/// Helper to recurse into nested schema structures
fn recurse_schema_fix(schema: &mut Value, fix_fn: fn(&mut Value)) {
    if let Value::Object(map) = schema {
        // Recurse into properties
        if let Some(Value::Object(props)) = map.get_mut("properties") {
            for (_, prop_schema) in props.iter_mut() {
                fix_fn(prop_schema);
            }
        }

        // Recurse into items (for arrays)
        if let Some(items) = map.get_mut("items") {
            fix_fn(items);
        }

        // Recurse into $defs
        if let Some(Value::Object(defs)) = map.get_mut("$defs") {
            for (_, def_schema) in defs.iter_mut() {
                fix_fn(def_schema);
            }
        }

        // Recurse into definitions
        if let Some(Value::Object(defs)) = map.get_mut("definitions") {
            for (_, def_schema) in defs.iter_mut() {
                fix_fn(def_schema);
            }
        }

        // Recurse into anyOf/oneOf/allOf
        for key in &["anyOf", "oneOf", "allOf"] {
            if let Some(Value::Array(variants)) = map.get_mut(*key) {
                for variant in variants.iter_mut() {
                    fix_fn(variant);
                }
            }
        }
    }
}

/// Trait for generating provider-specific schemas
pub trait SchemaGenerator: Send + Sync {
    /// Generate a schema for a tool
    ///
    /// # Arguments
    /// * `name` - Tool name
    /// * `description` - Tool description
    /// * `parameters` - JSON schema for parameters
    ///
    /// # Returns
    /// Provider-specific schema JSON
    fn generate_schema(&self, name: &str, description: &str, parameters: Value) -> Value;
}

/// Anthropic schema generator
pub struct AnthropicSchemaGenerator;

impl SchemaGenerator for AnthropicSchemaGenerator {
    fn generate_schema(&self, name: &str, description: &str, parameters: Value) -> Value {
        serde_json::json!({
            "name": name,
            "description": description,
            "input_schema": parameters,
        })
    }
}

/// OpenAI schema generator
pub struct OpenAISchemaGenerator;

impl SchemaGenerator for OpenAISchemaGenerator {
    fn generate_schema(&self, name: &str, description: &str, parameters: Value) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        })
    }
}

/// Google GenAI schema generator
pub struct GoogleSchemaGenerator;

impl SchemaGenerator for GoogleSchemaGenerator {
    fn generate_schema(&self, name: &str, description: &str, parameters: Value) -> Value {
        serde_json::json!({
            "name": name,
            "description": description,
            "parameters": parameters,
        })
    }
}

/// Get appropriate schema generator for provider
pub fn get_schema_generator(provider: &str) -> Result<Box<dyn SchemaGenerator>> {
    match provider {
        "anthropic" => Ok(Box::new(AnthropicSchemaGenerator)),
        "openai" => Ok(Box::new(OpenAISchemaGenerator)),
        "openrouter" => Ok(Box::new(OpenAISchemaGenerator)), // OpenRouter uses OpenAI format
        "google" | "google_gemini" | "vertex_gemini" | "gemini" | "google-genai" | "google-gemini" => {
            Ok(Box::new(GoogleSchemaGenerator))
        }
        "bedrock" => Ok(Box::new(AnthropicSchemaGenerator)), // Bedrock uses Anthropic format
        "google-anthropic" | "vertexai" => Ok(Box::new(AnthropicSchemaGenerator)), // Vertex AI with Claude
        _ => Err(Error::NonRetryable(format!(
            "Native tools not supported for provider: {}",
            provider
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_schema() {
        let generator = AnthropicSchemaGenerator;
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        });

        let schema = generator.generate_schema("get_weather", "Get weather for location", params);

        assert_eq!(schema["name"], "get_weather");
        assert_eq!(schema["description"], "Get weather for location");
        assert!(schema["input_schema"].is_object());
    }

    #[test]
    fn test_openai_schema() {
        let generator = OpenAISchemaGenerator;
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        });

        let schema = generator.generate_schema("get_weather", "Get weather for location", params);

        assert_eq!(schema["type"], "function");
        assert_eq!(schema["function"]["name"], "get_weather");
        assert_eq!(schema["function"]["description"], "Get weather for location");
        assert!(schema["function"]["parameters"].is_object());
    }

    #[test]
    fn test_google_schema() {
        let generator = GoogleSchemaGenerator;
        let params = serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        });

        let schema = generator.generate_schema("get_weather", "Get weather for location", params);

        assert_eq!(schema["name"], "get_weather");
        assert_eq!(schema["description"], "Get weather for location");
        assert!(schema["parameters"].is_object());
    }

    #[test]
    fn test_get_schema_generator_anthropic() {
        let result = get_schema_generator("anthropic");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_schema_generator_openai() {
        let result = get_schema_generator("openai");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_schema_generator_openrouter() {
        let result = get_schema_generator("openrouter");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_schema_generator_google() {
        let result = get_schema_generator("google_gemini");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_schema_generator_bedrock() {
        let result = get_schema_generator("bedrock");
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_schema_generator_unsupported() {
        let result = get_schema_generator("unsupported");
        assert!(result.is_err());
    }
}
