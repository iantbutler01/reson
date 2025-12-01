//! Output parsing system
//!
//! Types and traits for parsing structured outputs and native tool calls from LLMs.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;

/// Field description for schema generation
#[derive(Debug, Clone)]
pub struct FieldDescription {
    pub name: String,
    pub field_type: String,
    pub description: String,
    pub required: bool,
}

/// Trait for types that can be constructed from partial/streaming data
///
/// This trait enables progressive construction of structured outputs
/// as tokens arrive from streaming LLM responses (gasp-like behavior).
pub trait Deserializable: Serialize + for<'de> Deserialize<'de> + Send + Sync {
    /// Construct from partial JSON data
    ///
    /// This method should handle incomplete JSON gracefully,
    /// using default values or Option types for missing fields.
    fn from_partial(partial: serde_json::Value) -> Result<Self>
    where
        Self: Sized;

    /// Validate that the object is complete
    ///
    /// Returns Ok(()) if all required fields are present and valid,
    /// otherwise returns an error describing what's missing.
    fn validate_complete(&self) -> Result<()>;

    /// Get field descriptions for schema generation
    ///
    /// Returns metadata about fields that can be used to generate
    /// provider-specific tool schemas.
    fn field_descriptions() -> Vec<FieldDescription>
    where
        Self: Sized;
}

/// A parsed tool with metadata (tool_name and tool_use_id)
///
/// This wraps a Deserializable tool object along with the metadata
/// needed to track and execute it.
#[derive(Debug)]
pub struct ParsedTool {
    /// The tool name
    pub tool_name: String,

    /// The tool use ID from the LLM
    pub tool_use_id: String,

    /// The parsed tool arguments as JSON
    pub value: serde_json::Value,
}

/// Type alias for a tool constructor function
///
/// Takes partial JSON and returns a ParsedTool wrapper
pub type ToolConstructor = Box<dyn Fn(serde_json::Value) -> Result<ParsedTool> + Send + Sync>;

/// Native tool parser for handling tool call deltas from streaming responses
///
/// Accumulates JSON arguments as they arrive in chunks and constructs
/// Deserializable tool objects progressively.
#[derive(Clone)]
pub struct NativeToolParser {
    /// Mapping of tool names to constructor functions
    tool_constructors: Arc<HashMap<String, Arc<ToolConstructor>>>,
}

impl NativeToolParser {
    /// Create a new NativeToolParser with a registry of tool constructors
    pub fn new(tool_constructors: HashMap<String, Arc<ToolConstructor>>) -> Self {
        Self {
            tool_constructors: Arc::new(tool_constructors),
        }
    }

    /// Create a parser with a single tool registered (useful for testing)
    pub fn with_tool<T: Deserializable + Serialize + 'static>(tool_name: &str) -> Self {
        let mut constructors = HashMap::new();
        let name = tool_name.to_string();
        let constructor: ToolConstructor = Box::new(move |json: serde_json::Value| {
            T::from_partial(json.clone()).map(|tool| ParsedTool {
                tool_name: name.clone(),
                tool_use_id: String::new(),
                value: serde_json::to_value(&tool).unwrap_or(json),
            })
        });
        constructors.insert(tool_name.to_string(), Arc::new(constructor));
        Self::new(constructors)
    }

    /// Parse a tool call from streaming delta JSON, returning a ParsedTool
    ///
    /// Takes the tool name, accumulated JSON arguments string, and tool ID,
    /// and uses the registered constructor to build a ParsedTool with metadata.
    pub fn parse_tool(
        &self,
        tool_name: &str,
        delta_json: &str,
        tool_id: &str,
    ) -> ParsedToolResult {
        // Check if tool exists in registry
        let constructor = match self.tool_constructors.get(tool_name) {
            Some(ctor) => ctor,
            None => {
                return ParsedToolResult {
                    value: None,
                    error: Some(crate::error::Error::NonRetryable(format!(
                        "Tool '{}' not found in registry",
                        tool_name
                    ))),
                    is_partial: true,
                    raw_output: delta_json.to_string(),
                };
            }
        };

        // Try to parse the JSON (may be incomplete)
        let partial_data = match serde_json::from_str::<serde_json::Value>(delta_json) {
            Ok(data) => data,
            Err(_) => {
                // JSON parse failed - incomplete, try with empty object
                match constructor(serde_json::json!({})) {
                    Ok(mut empty_tool) => {
                        empty_tool.tool_use_id = tool_id.to_string();
                        return ParsedToolResult {
                            value: Some(empty_tool),
                            error: None,
                            is_partial: true,
                            raw_output: delta_json.to_string(),
                        };
                    }
                    Err(e) => {
                        return ParsedToolResult {
                            value: None,
                            error: Some(e),
                            is_partial: true,
                            raw_output: delta_json.to_string(),
                        };
                    }
                }
            }
        };

        // Use the constructor to build the ParsedTool
        match constructor(partial_data) {
            Ok(mut parsed_tool) => {
                // Set the tool_use_id from the streaming data
                parsed_tool.tool_use_id = tool_id.to_string();
                ParsedToolResult {
                    value: Some(parsed_tool),
                    error: None,
                    is_partial: true,
                    raw_output: delta_json.to_string(),
                }
            }
            Err(e) => {
                // Construction failed, try with empty object as fallback
                match constructor(serde_json::json!({})) {
                    Ok(mut empty_tool) => {
                        empty_tool.tool_use_id = tool_id.to_string();
                        ParsedToolResult {
                            value: Some(empty_tool),
                            error: Some(e),
                            is_partial: true,
                            raw_output: delta_json.to_string(),
                        }
                    }
                    Err(e2) => ParsedToolResult {
                        value: None,
                        error: Some(e2),
                        is_partial: true,
                        raw_output: delta_json.to_string(),
                    },
                }
            }
        }
    }

    /// Extract tool name from OpenAI-style tool call delta format
    pub fn extract_tool_name(&self, tool_call_data: &serde_json::Value) -> String {
        if let Some(function) = tool_call_data.get("function") {
            if let Some(name) = function.get("name") {
                return name.as_str().unwrap_or("").to_string();
            }
        }
        String::new()
    }

    /// Extract tool ID from tool call delta
    pub fn extract_tool_id(&self, tool_call_data: &serde_json::Value) -> Option<String> {
        tool_call_data
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Extract arguments JSON from OpenAI-style tool call delta format
    pub fn extract_arguments(&self, tool_call_data: &serde_json::Value) -> String {
        if let Some(function) = tool_call_data.get("function") {
            if let Some(args) = function.get("arguments") {
                return args.as_str().unwrap_or("{}").to_string();
            }
        }
        "{}".to_string()
    }
}

impl Default for NativeToolParser {
    fn default() -> Self {
        Self::new(HashMap::new())
    }
}

/// Result from parsing a tool call (dynamic, returns ParsedTool)
#[derive(Debug)]
pub struct ParsedToolResult {
    /// The parsed tool with metadata (if successful)
    pub value: Option<ParsedTool>,

    /// Error if parsing failed
    pub error: Option<crate::error::Error>,

    /// Whether this is a partial result (streaming)
    pub is_partial: bool,

    /// Raw output string
    pub raw_output: String,
}

impl ParsedToolResult {
    /// Check if parsing was successful
    pub fn success(&self) -> bool {
        self.value.is_some()
    }
}

/// Result from parsing a tool call (generic, for specific types)
#[derive(Debug)]
pub struct ParserResult<T> {
    /// The parsed value (if successful)
    pub value: Option<T>,

    /// Error if parsing failed
    pub error: Option<crate::error::Error>,

    /// Whether this is a partial result (streaming)
    pub is_partial: bool,

    /// Raw output string
    pub raw_output: String,
}

impl<T> ParserResult<T> {
    /// Check if parsing was successful
    pub fn success(&self) -> bool {
        self.value.is_some()
    }
}

/// Type parser for extracting structured data from text
pub struct TypeParser<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> TypeParser<T>
where
    T: for<'de> Deserialize<'de>,
{
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Parse from complete JSON
    pub fn parse(&self, json: &str) -> Result<T> {
        serde_json::from_str(json).map_err(|e| {
            crate::error::Error::NonRetryable(format!("Failed to parse JSON: {}", e))
        })
    }

    /// Parse from JSON value
    pub fn parse_value(&self, value: serde_json::Value) -> Result<T> {
        serde_json::from_value(value).map_err(|e| {
            crate::error::Error::NonRetryable(format!("Failed to parse JSON value: {}", e))
        })
    }
}

impl<T> Default for TypeParser<T>
where
    T: for<'de> Deserialize<'de>,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct TestStruct {
        #[serde(default)]
        name: String,
        #[serde(default)]
        age: u32,
        #[serde(default)]
        optional: Option<String>,
    }

    impl Deserializable for TestStruct {
        fn from_partial(partial: serde_json::Value) -> Result<Self> {
            // For testing, just use serde with defaults
            serde_json::from_value(partial).map_err(|e| {
                crate::error::Error::NonRetryable(format!("Parse error: {}", e))
            })
        }

        fn validate_complete(&self) -> Result<()> {
            if self.name.is_empty() {
                return Err(crate::error::Error::NonRetryable(
                    "name is required".to_string(),
                ));
            }
            Ok(())
        }

        fn field_descriptions() -> Vec<FieldDescription> {
            vec![
                FieldDescription {
                    name: "name".to_string(),
                    field_type: "string".to_string(),
                    description: "Person's name".to_string(),
                    required: true,
                },
                FieldDescription {
                    name: "age".to_string(),
                    field_type: "number".to_string(),
                    description: "Person's age".to_string(),
                    required: true,
                },
                FieldDescription {
                    name: "optional".to_string(),
                    field_type: "string".to_string(),
                    description: "Optional field".to_string(),
                    required: false,
                },
            ]
        }
    }

    #[test]
    fn test_native_tool_parser_new() {
        let parser = NativeToolParser::new(HashMap::new());
        assert!(parser.tool_constructors.is_empty());
    }

    #[test]
    fn test_native_tool_parser_extract_tool_name() {
        let parser = NativeToolParser::new(HashMap::new());
        let tool_data = serde_json::json!({
            "function": {"name": "GetWeather"}
        });

        let name = parser.extract_tool_name(&tool_data);
        assert_eq!(name, "GetWeather");
    }

    #[test]
    fn test_native_tool_parser_extract_arguments() {
        let parser = NativeToolParser::new(HashMap::new());
        let tool_data = serde_json::json!({
            "function": {"arguments": "{\"city\":\"SF\"}"}
        });

        let args = parser.extract_arguments(&tool_data);
        assert_eq!(args, "{\"city\":\"SF\"}");
    }

    #[test]
    fn test_native_tool_parser_extract_tool_id() {
        let parser = NativeToolParser::new(HashMap::new());
        let tool_data = serde_json::json!({
            "id": "tool_123"
        });

        let id = parser.extract_tool_id(&tool_data);
        assert_eq!(id, Some("tool_123".to_string()));
    }

    #[test]
    fn test_native_tool_parser_parse_tool_complete() {
        let parser = NativeToolParser::with_tool::<TestStruct>("TestTool");

        let json = r#"{"name":"Alice","age":30}"#;
        let result = parser.parse_tool("TestTool", json, "tool_1");

        assert!(result.success());
        let parsed = result.value.unwrap();
        assert_eq!(parsed.tool_name, "TestTool");
        assert_eq!(parsed.tool_use_id, "tool_1");

        // Verify the value contains the correct data
        let obj: TestStruct = serde_json::from_value(parsed.value).unwrap();
        assert_eq!(obj.name, "Alice");
        assert_eq!(obj.age, 30);
    }

    #[test]
    fn test_native_tool_parser_parse_tool_incomplete() {
        let parser = NativeToolParser::with_tool::<TestStruct>("TestTool");

        let json = r#"{"name":"Alice","age":"#;
        let result = parser.parse_tool("TestTool", json, "tool_1");

        // Should fall back to empty object
        assert!(result.is_partial);
        assert!(result.value.is_some());
    }

    #[test]
    fn test_native_tool_parser_parse_tool_not_found() {
        let parser = NativeToolParser::new(HashMap::new());

        let json = r#"{"name":"Alice"}"#;
        let result = parser.parse_tool("UnknownTool", json, "tool_1");

        assert!(!result.success());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_type_parser_parse() {
        let parser = TypeParser::<TestStruct>::new();
        let json = r#"{"name":"Bob","age":25}"#;

        let result = parser.parse(json);
        assert!(result.is_ok());

        let obj = result.unwrap();
        assert_eq!(obj.name, "Bob");
        assert_eq!(obj.age, 25);
    }

    #[test]
    fn test_type_parser_parse_value() {
        let parser = TypeParser::<TestStruct>::new();
        let value = serde_json::json!({"name":"Charlie","age":35});

        let result = parser.parse_value(value);
        assert!(result.is_ok());

        let obj = result.unwrap();
        assert_eq!(obj.name, "Charlie");
        assert_eq!(obj.age, 35);
    }

    #[test]
    fn test_type_parser_parse_invalid() {
        let parser = TypeParser::<TestStruct>::new();
        let json = r#"{"invalid":"json"#;

        let result = parser.parse(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_deserializable_from_partial() {
        let partial = serde_json::json!({"name":"Dave","age":40});
        let result = TestStruct::from_partial(partial);

        assert!(result.is_ok());
        let obj = result.unwrap();
        assert_eq!(obj.name, "Dave");
        assert_eq!(obj.age, 40);
    }

    #[test]
    fn test_deserializable_validate_complete() {
        let valid = TestStruct {
            name: "Eve".to_string(),
            age: 28,
            optional: None,
        };
        assert!(valid.validate_complete().is_ok());

        let invalid = TestStruct {
            name: "".to_string(),
            age: 28,
            optional: None,
        };
        assert!(invalid.validate_complete().is_err());
    }

    #[test]
    fn test_deserializable_field_descriptions() {
        let descs = TestStruct::field_descriptions();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].name, "name");
        assert_eq!(descs[1].name, "age");
        assert_eq!(descs[2].name, "optional");
        assert!(descs[0].required);
        assert!(descs[1].required);
        assert!(!descs[2].required);
    }

    #[test]
    fn test_parsed_tool_metadata() {
        let parser = NativeToolParser::with_tool::<TestStruct>("GetPerson");

        let json = r#"{"name":"Frank","age":42}"#;
        let result = parser.parse_tool("GetPerson", json, "call_abc123");

        assert!(result.success());
        let parsed = result.value.unwrap();

        // Verify metadata
        assert_eq!(parsed.tool_name, "GetPerson");
        assert_eq!(parsed.tool_use_id, "call_abc123");

        // Verify the value can be deserialized back to TestStruct
        let obj: TestStruct = serde_json::from_value(parsed.value).unwrap();
        assert_eq!(obj.name, "Frank");
        assert_eq!(obj.age, 42);
    }

    #[test]
    fn test_parser_with_multiple_tools() {
        let mut constructors = HashMap::new();

        // Register TestStruct as "Tool1"
        let name1 = "Tool1".to_string();
        let constructor1: ToolConstructor = Box::new(move |json: serde_json::Value| {
            TestStruct::from_partial(json.clone()).map(|tool| ParsedTool {
                tool_name: name1.clone(),
                tool_use_id: String::new(),
                value: serde_json::to_value(&tool).unwrap_or(json),
            })
        });
        constructors.insert("Tool1".to_string(), Arc::new(constructor1));

        let parser = NativeToolParser::new(constructors);

        // Can parse Tool1
        let result = parser.parse_tool("Tool1", r#"{"name":"George","age":50}"#, "id1");
        assert!(result.success());
        assert_eq!(result.value.unwrap().tool_name, "Tool1");

        // Cannot parse Tool2 (not registered)
        let result = parser.parse_tool("Tool2", r#"{}"#, "id2");
        assert!(!result.success());
        assert!(result.error.is_some());
    }
}
