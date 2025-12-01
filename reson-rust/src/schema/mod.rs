//! Schema generation for tools
//!
//! Converts Rust types to provider-specific tool schemas.

use serde_json::Value;
use crate::error::{Error, Result};

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
        "google" | "google_gemini" | "vertex_gemini" => Ok(Box::new(GoogleSchemaGenerator)),
        "bedrock" => Ok(Box::new(AnthropicSchemaGenerator)), // Bedrock uses Anthropic format
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
