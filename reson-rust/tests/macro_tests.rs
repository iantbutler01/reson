//! Integration tests for proc macros
//!
//! These tests demonstrate that the #[agentic], #[tool], and #[deserializable]
//! macros compile and generate correct code.

use reson_agentic::Deserializable;
use reson_agentic::parsers::Deserializable as DeserializableTrait;
use serde::{Deserialize, Serialize};

#[test]
fn test_deserializable_macro() {
    #[derive(Deserializable, Serialize, Deserialize, Debug)]
    struct Person {
        /// The person's name
        name: String,
        /// The person's age
        age: u32,
        /// Optional email address
        email: Option<String>,
    }

    // Test from_partial
    let json = serde_json::json!({
        "name": "Alice",
        "age": 30,
        "email": "alice@example.com"
    });
    let person = Person::from_partial(json).unwrap();
    assert_eq!(person.name, "Alice");
    assert_eq!(person.age, 30);
    assert_eq!(person.email, Some("alice@example.com".to_string()));

    // Test field_descriptions
    let descriptions = Person::field_descriptions();
    assert_eq!(descriptions.len(), 3);
    assert_eq!(descriptions[0].name, "name");
    assert_eq!(descriptions[0].description, "The person's name");
    assert!(descriptions[0].required);

    assert_eq!(descriptions[1].name, "age");
    assert_eq!(descriptions[1].description, "The person's age");
    assert!(descriptions[1].required);

    assert_eq!(descriptions[2].name, "email");
    assert_eq!(descriptions[2].description, "Optional email address");
    assert!(!descriptions[2].required); // Optional

    // Test validate_complete
    assert!(person.validate_complete().is_ok());
}

#[test]
fn test_deserializable_macro_partial() {
    #[derive(Deserializable, Serialize, Deserialize)]
    struct Config {
        host: String,
        port: Option<u16>,
    }

    // Test with missing optional field
    let json = serde_json::json!({
        "host": "localhost"
    });
    let config = Config::from_partial(json).unwrap();
    assert_eq!(config.host, "localhost");
    assert_eq!(config.port, None);
}

#[test]
fn test_tool_macro_snake_case_conversion() {
    // Test that the Tool derive macro generates proper methods

    use reson_agentic::Tool;
    use serde::{Serialize, Deserialize};

    /// A calculator tool for basic math operations
    #[derive(Tool, Serialize, Deserialize)]
    struct CalculatorTool {
        /// The operation to perform (add, subtract, multiply, divide)
        operation: String,
        /// First number
        a: f64,
        /// Second number
        b: f64,
        /// Optional precision
        precision: Option<u32>,
    }

    // Test that the macro generates tool_name()
    assert_eq!(CalculatorTool::tool_name(), "calculator_tool");

    // Test that description() returns doc comment
    assert_eq!(CalculatorTool::description(), "A calculator tool for basic math operations");

    // Test that schema() generates proper JSON schema
    let schema = CalculatorTool::schema();
    assert_eq!(schema["type"], "object");

    let props = schema["properties"].as_object().unwrap();
    assert!(props.contains_key("operation"));
    assert!(props.contains_key("a"));
    assert!(props.contains_key("b"));
    assert!(props.contains_key("precision"));

    // Check types
    assert_eq!(props["operation"]["type"], "string");
    assert_eq!(props["a"]["type"], "number"); // f64 maps to "number"
    assert_eq!(props["b"]["type"], "number");
    assert_eq!(props["precision"]["type"], "integer"); // u32 inside Option<u32>

    // Check descriptions
    assert_eq!(props["operation"]["description"], "The operation to perform (add, subtract, multiply, divide)");
    assert_eq!(props["a"]["description"], "First number");

    // Check required fields (non-Option fields)
    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&serde_json::json!("operation")));
    assert!(required.contains(&serde_json::json!("a")));
    assert!(required.contains(&serde_json::json!("b")));
    assert!(!required.contains(&serde_json::json!("precision"))); // Optional, not required
}

#[test]
fn test_tool_macro_with_schema_generator() {
    use reson_agentic::Tool;
    use reson_agentic::schema::AnthropicSchemaGenerator;
    use serde::{Serialize, Deserialize};

    /// Get current weather for a location
    #[derive(Tool, Serialize, Deserialize)]
    struct GetWeather {
        /// City name
        location: String,
        /// Temperature unit (celsius or fahrenheit)
        unit: Option<String>,
    }

    // Test tool_schema() with Anthropic generator
    let generator = AnthropicSchemaGenerator;
    let schema = GetWeather::tool_schema(&generator);

    // Anthropic format uses "input_schema"
    assert_eq!(schema["name"], "get_weather");
    assert_eq!(schema["description"], "Get current weather for a location");
    assert!(schema["input_schema"].is_object());
    assert_eq!(schema["input_schema"]["type"], "object");
}

// Note: Testing the #[agentic] macro requires careful setup since it transforms
// the function signature. Below we test that the macro generates valid code.

#[cfg(test)]
mod agentic_macro_tests {
    use reson_agentic::agentic;
    use reson_agentic::runtime::Runtime;
    use reson_agentic::error::Result;

    // Test that the macro compiles and generates a function without runtime param
    #[agentic(model = "anthropic:claude-3-5-sonnet-20241022")]
    async fn simple_agentic_fn(input: String, runtime: Runtime) -> Result<serde_json::Value> {
        // Mark runtime as used by calling run()
        // Note: This won't actually call an LLM in tests, it will fail without API key
        // but the point is to verify the macro generates valid code
        runtime.run(
            Some(&input),
            None, None, None, None, None, None, None, None, None
        ).await
    }

    #[test]
    fn test_agentic_macro_compiles() {
        // This test just verifies the macro generates valid code
        // The function signature should NOT include `runtime` - it's injected

        // We can check that the function exists and has the right signature
        // by creating a function pointer (this will fail to compile if signature is wrong)
        let _fn_ptr: fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send>> = |input| {
            Box::pin(simple_agentic_fn(input))
        };
    }

    // Test with no model attribute (should use None)
    #[agentic]
    async fn no_model_agentic_fn(data: i32, runtime: Runtime) -> Result<serde_json::Value> {
        let _ = data; // Use the parameter
        runtime.run(
            Some("test"),
            None, None, None, None, None, None, None, None, None
        ).await
    }

    #[test]
    fn test_agentic_macro_no_model_compiles() {
        // Just verify it compiles
        let _fn_ptr: fn(i32) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send>> = |data| {
            Box::pin(no_model_agentic_fn(data))
        };
    }

    // Test with multiple parameters
    #[agentic(model = "openai:gpt-4")]
    async fn multi_param_fn(name: String, count: u32, runtime: Runtime) -> Result<serde_json::Value> {
        let prompt = format!("Name: {}, Count: {}", name, count);
        runtime.run(
            Some(&prompt),
            None, None, None, None, None, None, None, None, None
        ).await
    }

    #[test]
    fn test_agentic_macro_multi_params_compiles() {
        // Verify multi-param function compiles correctly
        let _fn_ptr: fn(String, u32) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<serde_json::Value>> + Send>> = |name, count| {
            Box::pin(multi_param_fn(name, count))
        };
    }
}
