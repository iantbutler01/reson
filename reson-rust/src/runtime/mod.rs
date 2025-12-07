//! Runtime - Core execution environment for agentic functions
//!
//! The Runtime is the main interface for Reson. It manages:
//! - LLM client lifecycle and API calls
//! - Tool registration and execution
//! - Message history and accumulators
//! - Context/state storage
//! - Structured output parsing

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Error, Result};
use crate::parsers::{Deserializable, ParsedTool, ToolConstructor};
use crate::storage::{MemoryStore, Storage};
use crate::types::ReasoningSegment;
use crate::utils::ConversationMessage;
use futures::future::BoxFuture;

pub mod context;
pub mod decorators;
pub mod inference;

pub use context::ContextApi;

/// Accumulated state during runtime execution
#[derive(Debug, Default, Clone)]
pub struct Accumulators {
    raw_response: Vec<String>,
    reasoning: Vec<String>,
    reasoning_segments: Vec<ReasoningSegment>,
    current_reasoning_segment: Option<ReasoningSegment>,
}

/// Stored schema information for a tool
#[derive(Debug, Clone)]
pub struct ToolSchemaInfo {
    pub name: String,
    pub description: String,
    pub fields: Vec<crate::parsers::FieldDescription>,
}

/// Runtime - Main execution environment for agentic functions
pub struct Runtime {
    // Public configuration
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub used: bool,

    // Private state (using interior mutability)
    store: Arc<dyn Storage>,
    tools: Arc<RwLock<HashMap<String, ToolFunction>>>,
    tool_types: Arc<RwLock<HashMap<String, String>>>, // tool_name -> type_name mapping
    tool_schemas: Arc<RwLock<HashMap<String, ToolSchemaInfo>>>, // tool_name -> schema info
    tool_constructors: Arc<RwLock<HashMap<String, Arc<ToolConstructor>>>>, // For NativeToolParser
    default_prompt: Arc<RwLock<String>>,
    return_type: Arc<RwLock<Option<String>>>, // Store type name as string
    accumulators: Arc<RwLock<Accumulators>>,
    messages: Arc<RwLock<Vec<ConversationMessage>>>,
    current_call_args: Arc<RwLock<Option<HashMap<String, serde_json::Value>>>>,
}

/// Wrapper for tool functions (sync or async)
pub enum ToolFunction {
    Sync(Box<dyn Fn(serde_json::Value) -> Result<String> + Send + Sync>),
    Async(Box<dyn Fn(serde_json::Value) -> futures::future::BoxFuture<'static, Result<String>> + Send + Sync>),
}

impl Runtime {
    /// Create a new Runtime with default memory storage
    pub fn new() -> Self {
        Self {
            model: None,
            api_key: None,
            used: false,
            store: Arc::new(MemoryStore::new()),
            tools: Arc::new(RwLock::new(HashMap::new())),
            tool_types: Arc::new(RwLock::new(HashMap::new())),
            tool_schemas: Arc::new(RwLock::new(HashMap::new())),
            tool_constructors: Arc::new(RwLock::new(HashMap::new())),
            default_prompt: Arc::new(RwLock::new(String::new())),
            return_type: Arc::new(RwLock::new(None)),
            accumulators: Arc::new(RwLock::new(Accumulators::default())),
            messages: Arc::new(RwLock::new(Vec::new())),
            current_call_args: Arc::new(RwLock::new(None)),
        }
    }

    /// Create a Runtime with specific configuration
    pub fn with_config(
        model: Option<String>,
        api_key: Option<String>,
        store: Arc<dyn Storage>,
    ) -> Self {
        Self {
            model,
            api_key,
            used: false,
            store,
            tools: Arc::new(RwLock::new(HashMap::new())),
            tool_types: Arc::new(RwLock::new(HashMap::new())),
            tool_schemas: Arc::new(RwLock::new(HashMap::new())),
            tool_constructors: Arc::new(RwLock::new(HashMap::new())),
            default_prompt: Arc::new(RwLock::new(String::new())),
            return_type: Arc::new(RwLock::new(None)),
            accumulators: Arc::new(RwLock::new(Accumulators::default())),
            messages: Arc::new(RwLock::new(Vec::new())),
            current_call_args: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a tool function
    ///
    /// # Arguments
    /// * `name` - Tool name for LLM
    /// * `tool_fn` - Sync or async function
    /// * `tool_type` - Optional type name for marshalling
    pub async fn register_tool(
        &self,
        name: impl Into<String>,
        tool_fn: ToolFunction,
        tool_type: Option<String>,
    ) -> Result<()> {
        let name = name.into();

        // Store tool function
        let mut tools = self.tools.write().await;
        if tools.contains_key(&name) {
            return Err(Error::NonRetryable(format!(
                "Tool '{}' is already registered",
                name
            )));
        }
        tools.insert(name.clone(), tool_fn);
        drop(tools);

        // Store tool type if provided
        if let Some(type_name) = tool_type {
            let mut tool_types = self.tool_types.write().await;
            tool_types.insert(name, type_name);
        }

        Ok(())
    }

    /// Register a tool function with schema information
    ///
    /// This method is useful when you have a tool struct that provides its schema
    /// via `#[derive(Tool)]`, and you want to register it with a handler function.
    ///
    /// # Arguments
    /// * `name` - Tool name for LLM
    /// * `description` - Tool description
    /// * `schema` - JSON schema for the tool parameters (from Tool::schema())
    /// * `tool_fn` - Sync or async function to handle the tool call
    ///
    /// # Example
    /// ```ignore
    /// runtime.register_tool_with_schema(
    ///     MyTool::tool_name(),
    ///     MyTool::description(),
    ///     MyTool::schema(),
    ///     ToolFunction::Sync(Box::new(|args| {
    ///         // Handle the tool call
    ///         Ok("result".to_string())
    ///     })),
    /// ).await?;
    /// ```
    pub async fn register_tool_with_schema(
        &self,
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        tool_fn: ToolFunction,
    ) -> Result<()> {
        let name = name.into();
        let description = description.into();

        // Store tool function
        let mut tools = self.tools.write().await;
        if tools.contains_key(&name) {
            return Err(Error::NonRetryable(format!(
                "Tool '{}' is already registered",
                name
            )));
        }
        tools.insert(name.clone(), tool_fn);
        drop(tools);

        // Extract field info from schema and store schema info
        let fields = Self::extract_fields_from_schema(&schema);
        let schema_info = ToolSchemaInfo {
            name: name.clone(),
            description,
            fields,
        };
        let mut schemas = self.tool_schemas.write().await;
        schemas.insert(name, schema_info);

        Ok(())
    }

    /// Helper to extract field descriptions from a JSON schema
    fn extract_fields_from_schema(schema: &serde_json::Value) -> Vec<crate::parsers::FieldDescription> {
        let mut fields = Vec::new();

        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            let required = schema.get("required")
                .and_then(|r| r.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
                .unwrap_or_default();

            for (name, prop) in properties {
                let field_type = prop.get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("object")
                    .to_string();
                let description = prop.get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                let is_required = required.contains(&name.as_str());

                fields.push(crate::parsers::FieldDescription {
                    name: name.clone(),
                    field_type,
                    description,
                    required: is_required,
                });
            }
        }

        fields
    }

    /// Register a tool with type and handler (Python: runtime.tool(fn, name=..., tool_type=...))
    ///
    /// This is the primary API for registering tools for native tool calling.
    /// The type parameter T must implement Deserializable.
    ///
    /// # Arguments
    /// * `handler` - Async function that handles tool execution
    /// * `name` - Optional tool name (defaults to type name if None)
    ///
    /// # Example
    /// ```ignore
    /// // Register a tool with a type parameter
    /// runtime.tool::<MyToolType, _>(my_handler, Some("my_tool")).await?;
    /// // Or let it derive the name from the type
    /// runtime.tool::<MyToolType, _>(my_handler, None).await?;
    /// ```
    pub async fn tool<T, F>(&self, handler: F, name: Option<&str>) -> Result<()>
    where
        T: Deserializable + serde::Serialize + 'static,
        F: Fn(ParsedTool) -> BoxFuture<'static, Result<String>> + Send + Sync + 'static,
    {
        // Get tool name - either from parameter or type name
        let type_name = std::any::type_name::<T>();
        let tool_name_str = name.unwrap_or_else(|| {
            type_name
                .split("::")
                .last()
                .unwrap_or(type_name)
        });
        let tool_name = tool_name_str.to_string(); // Convert to owned String

        // Store handler function (wrap to match ToolFunction signature)
        let mut tools = self.tools.write().await;
        if tools.contains_key(&tool_name) {
            return Err(Error::NonRetryable(format!(
                "Tool '{}' is already registered",
                tool_name
            )));
        }

        // Wrap the handler to convert ParsedTool -> serde_json::Value
        let tool_name_for_handler = tool_name.clone();
        let wrapped_handler = Box::new(move |json_value: serde_json::Value| {
            // For now, create a ParsedTool from the JSON
            // TODO: This will be populated properly by NativeToolParser
            let parsed_tool = ParsedTool {
                tool_name: tool_name_for_handler.clone(),
                tool_use_id: String::new(),
                value: json_value,
            };
            handler(parsed_tool)
        });

        tools.insert(tool_name.clone(), ToolFunction::Async(wrapped_handler));
        drop(tools);

        // Store type name
        let mut tool_types = self.tool_types.write().await;
        tool_types.insert(tool_name.to_string(), type_name.to_string());
        drop(tool_types);

        // Extract and store schema information from Deserializable type
        let field_descriptions = T::field_descriptions();
        let schema_info = ToolSchemaInfo {
            name: tool_name.clone(),
            description: format!("Tool: {}", tool_name), // Default description
            fields: field_descriptions,
        };
        let mut schemas = self.tool_schemas.write().await;
        schemas.insert(tool_name.clone(), schema_info);
        drop(schemas);

        // Store constructor for NativeToolParser
        let tool_name_clone = tool_name.clone();
        let constructor: ToolConstructor = Box::new(move |json: serde_json::Value| {
            T::from_partial(json.clone()).map(|tool| {
                ParsedTool {
                    tool_name: tool_name_clone.clone(),
                    tool_use_id: String::new(), // Will be set by parser
                    value: serde_json::to_value(&tool).unwrap(),
                }
            })
        });

        let mut constructors = self.tool_constructors.write().await;
        constructors.insert(tool_name.clone(), Arc::new(constructor));

        Ok(())
    }

    /// Get a NativeToolParser instance with all registered tool constructors
    ///
    /// This parser can be used during streaming to dynamically construct ParsedTool
    /// instances from tool call deltas.
    pub async fn get_parser(&self) -> crate::parsers::NativeToolParser {
        let constructors = self.tool_constructors.read().await;
        crate::parsers::NativeToolParser::new(constructors.clone())
    }

    /// Get tool schema information for all registered tools
    pub async fn get_tool_schemas(&self) -> HashMap<String, ToolSchemaInfo> {
        let schemas = self.tool_schemas.read().await;
        schemas.clone()
    }

    /// Execute a non-streaming LLM call
    ///
    /// # Arguments
    /// * `prompt` - User prompt (optional, uses default if None)
    /// * `system` - System message (optional)
    /// * `history` - Previous messages (optional)
    /// * `output_type` - Expected output type name (optional)
    /// * `temperature`, `top_p`, `max_tokens` - Generation parameters (optional)
    /// * `model` - Override model (optional)
    /// * `api_key` - Override API key (optional)
    ///
    /// # Returns
    /// Parsed response as JSON value
    pub async fn run(
        &mut self,
        prompt: Option<&str>,
        system: Option<&str>,
        history: Option<Vec<ConversationMessage>>,
        output_type: Option<String>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> Result<serde_json::Value> {
        // Mark as used
        self.used = true;

        // Clear accumulators
        self.clear_raw_response().await;
        self.clear_reasoning().await;

        // Get effective values
        let prompt_text = match prompt {
            Some(p) => p.to_string(),
            None => self.default_prompt.read().await.clone(),
        };

        let effective_model = model
            .or_else(|| self.model.clone())
            .ok_or_else(|| Error::NonRetryable("No model specified".to_string()))?;

        let effective_api_key = api_key.or_else(|| self.api_key.clone());

        // Call inference utilities
        let result = inference::call_llm(
            Some(&prompt_text),
            &effective_model,
            self.tools.clone(),
            self.tool_schemas.clone(),
            output_type,
            self.store.clone(),
            effective_api_key.as_deref(),
            system,
            history,
            temperature,
            top_p,
            max_tokens,
            self.current_call_args.clone(),
        )
        .await?;

        // Update accumulators
        if let Some(raw) = &result.raw_response {
            let mut acc = self.accumulators.write().await;
            acc.raw_response.push(raw.clone());
        }

        if let Some(reasoning) = &result.reasoning {
            let mut acc = self.accumulators.write().await;
            acc.reasoning.push(reasoning.clone());
        }

        Ok(result.parsed_value)
    }

    /// Execute a streaming LLM call
    ///
    /// Returns an async stream of (chunk_type, chunk_value) tuples
    pub async fn run_stream(
        &mut self,
        prompt: Option<&str>,
        system: Option<&str>,
        history: Option<Vec<ConversationMessage>>,
        output_type: Option<String>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<u32>,
        model: Option<String>,
        api_key: Option<String>,
    ) -> Result<impl futures::stream::Stream<Item = Result<(String, serde_json::Value)>>> {
        // Mark as used
        self.used = true;

        // Clear accumulators
        self.clear_raw_response().await;
        self.clear_reasoning().await;
        self.clear_reasoning_segments().await;

        // Get effective values
        let prompt_text = match prompt {
            Some(p) => p.to_string(),
            None => self.default_prompt.read().await.clone(),
        };

        let effective_model = model
            .or_else(|| self.model.clone())
            .ok_or_else(|| Error::NonRetryable("No model specified".to_string()))?;

        let effective_api_key = api_key.or_else(|| self.api_key.clone());

        // Call streaming inference
        inference::call_llm_stream(
            Some(&prompt_text),
            &effective_model,
            self.tools.clone(),
            self.tool_schemas.clone(),
            output_type,
            self.store.clone(),
            effective_api_key.as_deref(),
            system,
            history,
            temperature,
            top_p,
            max_tokens,
            self.current_call_args.clone(),
            self.accumulators.clone(),
        )
        .await
    }

    /// Check if a result is a tool call
    pub fn is_tool_call(&self, result: &serde_json::Value) -> bool {
        // Check for _tool_name field in JSON
        result
            .get("_tool_name")
            .and_then(|v| v.as_str())
            .is_some()
    }

    /// Get tool name from result
    pub fn get_tool_name(&self, result: &serde_json::Value) -> Option<String> {
        result
            .get("_tool_name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Execute a tool call
    pub async fn execute_tool(&self, tool_result: &serde_json::Value) -> Result<String> {
        let tool_name = self
            .get_tool_name(tool_result)
            .ok_or_else(|| Error::NonRetryable("No tool name in result".to_string()))?;

        let tools = self.tools.read().await;
        let tool_fn = tools
            .get(&tool_name)
            .ok_or_else(|| Error::NonRetryable(format!("Tool '{}' not found", tool_name)))?;

        // Extract arguments
        let args = tool_result.clone();

        // Execute based on function type
        match tool_fn {
            ToolFunction::Sync(f) => f(args),
            ToolFunction::Async(f) => f(args).await,
        }
    }

    /// Get accumulated raw response
    pub async fn raw_response(&self) -> String {
        let acc = self.accumulators.read().await;
        acc.raw_response.join("")
    }

    /// Get accumulated reasoning
    pub async fn reasoning(&self) -> String {
        let acc = self.accumulators.read().await;
        acc.reasoning.join("")
    }

    /// Get reasoning segments
    pub async fn reasoning_segments(&self) -> Vec<ReasoningSegment> {
        let acc = self.accumulators.read().await;
        acc.reasoning_segments.clone()
    }

    /// Clear raw response accumulator
    pub async fn clear_raw_response(&self) {
        let mut acc = self.accumulators.write().await;
        acc.raw_response.clear();
    }

    /// Clear reasoning accumulator
    pub async fn clear_reasoning(&self) {
        let mut acc = self.accumulators.write().await;
        acc.reasoning.clear();
    }

    /// Clear reasoning segments
    pub async fn clear_reasoning_segments(&self) {
        let mut acc = self.accumulators.write().await;
        acc.reasoning_segments.clear();
        acc.current_reasoning_segment = None;
    }

    /// Get context API for key-value storage
    pub fn context(&self) -> ContextApi {
        ContextApi::new(self.store.clone())
    }

    /// Set default prompt (from function docstring)
    pub async fn set_default_prompt(&self, prompt: impl Into<String>) {
        let mut default_prompt = self.default_prompt.write().await;
        *default_prompt = prompt.into();
    }

    /// Set return type
    pub async fn set_return_type(&self, type_name: Option<String>) {
        let mut return_type = self.return_type.write().await;
        *return_type = type_name;
    }

    /// Set current call arguments (for prompt enhancement)
    pub async fn set_current_call_args(&self, args: Option<HashMap<String, serde_json::Value>>) {
        let mut current_args = self.current_call_args.write().await;
        *current_args = args;
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_runtime_new() {
        let runtime = Runtime::new();
        assert!(!runtime.used);
        assert!(runtime.model.is_none());
    }

    #[tokio::test]
    async fn test_runtime_with_config() {
        let store = Arc::new(MemoryStore::new());
        let runtime = Runtime::with_config(
            Some("test-model".to_string()),
            Some("test-key".to_string()),
            store,
        );

        assert_eq!(runtime.model, Some("test-model".to_string()));
        assert_eq!(runtime.api_key, Some("test-key".to_string()));
    }

    #[tokio::test]
    async fn test_register_tool() {
        let runtime = Runtime::new();

        let tool_fn = ToolFunction::Sync(Box::new(|_args| Ok("result".to_string())));

        runtime
            .register_tool("test_tool", tool_fn, None)
            .await
            .unwrap();

        let tools = runtime.tools.read().await;
        assert!(tools.contains_key("test_tool"));
    }

    #[tokio::test]
    async fn test_register_duplicate_tool() {
        let runtime = Runtime::new();

        let tool_fn1 = ToolFunction::Sync(Box::new(|_args| Ok("result1".to_string())));
        let tool_fn2 = ToolFunction::Sync(Box::new(|_args| Ok("result2".to_string())));

        runtime
            .register_tool("test_tool", tool_fn1, None)
            .await
            .unwrap();

        let result = runtime.register_tool("test_tool", tool_fn2, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_is_tool_call() {
        let runtime = Runtime::new();

        let tool_call = serde_json::json!({
            "_tool_name": "test_tool",
            "arg1": "value1"
        });

        assert!(runtime.is_tool_call(&tool_call));

        let not_tool_call = serde_json::json!({
            "result": "value"
        });

        assert!(!runtime.is_tool_call(&not_tool_call));
    }

    #[tokio::test]
    async fn test_get_tool_name() {
        let runtime = Runtime::new();

        let tool_call = serde_json::json!({
            "_tool_name": "my_tool",
            "arg1": "value1"
        });

        assert_eq!(runtime.get_tool_name(&tool_call), Some("my_tool".to_string()));
    }

    #[tokio::test]
    async fn test_execute_tool_sync() {
        let runtime = Runtime::new();

        let tool_fn = ToolFunction::Sync(Box::new(|args| {
            let name = args.get("name").and_then(|v| v.as_str()).unwrap_or("world");
            Ok(format!("Hello, {}!", name))
        }));

        runtime
            .register_tool("greet", tool_fn, None)
            .await
            .unwrap();

        let tool_call = serde_json::json!({
            "_tool_name": "greet",
            "name": "Alice"
        });

        let result = runtime.execute_tool(&tool_call).await.unwrap();
        assert_eq!(result, "Hello, Alice!");
    }

    #[tokio::test]
    async fn test_accumulators() {
        let runtime = Runtime::new();

        // Initially empty
        assert_eq!(runtime.raw_response().await, "");
        assert_eq!(runtime.reasoning().await, "");

        // Add some data
        {
            let mut acc = runtime.accumulators.write().await;
            acc.raw_response.push("Hello ".to_string());
            acc.raw_response.push("World".to_string());
            acc.reasoning.push("Think: ".to_string());
            acc.reasoning.push("Answer".to_string());
        }

        assert_eq!(runtime.raw_response().await, "Hello World");
        assert_eq!(runtime.reasoning().await, "Think: Answer");

        // Clear
        runtime.clear_raw_response().await;
        runtime.clear_reasoning().await;

        assert_eq!(runtime.raw_response().await, "");
        assert_eq!(runtime.reasoning().await, "");
    }

    #[tokio::test]
    #[ignore] // TODO: Requires Storage trait refactor for interior mutability
    async fn test_context() {
        let runtime = Runtime::new();
        let ctx = runtime.context();

        // Context is accessible
        ctx.set("test_key", serde_json::json!("test_value"))
            .await
            .unwrap();

        let value: Option<String> = ctx.get("test_key").await.unwrap();
        assert_eq!(value, Some("test_value".to_string()));
    }

    // Test types for tool registration
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    struct WeatherQuery {
        #[serde(default)]
        location: String,
        #[serde(default)]
        unit: Option<String>,
    }

    impl crate::parsers::Deserializable for WeatherQuery {
        fn from_partial(partial: serde_json::Value) -> crate::error::Result<Self> {
            serde_json::from_value(partial).map_err(|e| {
                crate::error::Error::NonRetryable(format!("Parse error: {}", e))
            })
        }

        fn validate_complete(&self) -> crate::error::Result<()> {
            if self.location.is_empty() {
                return Err(crate::error::Error::NonRetryable(
                    "location is required".to_string(),
                ));
            }
            Ok(())
        }

        fn field_descriptions() -> Vec<crate::parsers::FieldDescription> {
            vec![
                crate::parsers::FieldDescription {
                    name: "location".to_string(),
                    field_type: "string".to_string(),
                    description: "The city to get weather for".to_string(),
                    required: true,
                },
                crate::parsers::FieldDescription {
                    name: "unit".to_string(),
                    field_type: "string".to_string(),
                    description: "Temperature unit (celsius or fahrenheit)".to_string(),
                    required: false,
                },
            ]
        }
    }

    #[tokio::test]
    async fn test_tool_registration_with_schema() {
        use futures::future::BoxFuture;

        let runtime = Runtime::new();

        // Register tool with type parameter
        runtime
            .tool::<WeatherQuery, _>(
                |parsed_tool| -> BoxFuture<'static, crate::error::Result<String>> {
                    Box::pin(async move {
                        let query: WeatherQuery = serde_json::from_value(parsed_tool.value)?;
                        Ok(format!("Weather in {}: Sunny", query.location))
                    })
                },
                Some("get_weather"),
            )
            .await
            .unwrap();

        // Check that tool is registered
        let tools = runtime.tools.read().await;
        assert!(tools.contains_key("get_weather"));
        drop(tools);

        // Check that schema info is captured
        let schemas = runtime.tool_schemas.read().await;
        let schema_info = schemas.get("get_weather").unwrap();
        assert_eq!(schema_info.name, "get_weather");
        assert_eq!(schema_info.fields.len(), 2);
        assert_eq!(schema_info.fields[0].name, "location");
        assert_eq!(schema_info.fields[0].required, true);
        assert_eq!(schema_info.fields[1].name, "unit");
        assert_eq!(schema_info.fields[1].required, false);
    }

    #[tokio::test]
    async fn test_tool_schema_generation() {
        use futures::future::BoxFuture;

        let runtime = Runtime::new();

        // Register tool
        runtime
            .tool::<WeatherQuery, _>(
                |parsed_tool| -> BoxFuture<'static, crate::error::Result<String>> {
                    Box::pin(async move {
                        let _query: WeatherQuery = serde_json::from_value(parsed_tool.value)?;
                        Ok("Sunny".to_string())
                    })
                },
                Some("get_weather"),
            )
            .await
            .unwrap();

        // Get tool schemas and verify structure
        let schemas = runtime.get_tool_schemas().await;
        assert_eq!(schemas.len(), 1);

        let weather_schema = schemas.get("get_weather").unwrap();
        assert!(weather_schema.fields.iter().any(|f| f.name == "location" && f.required));
        assert!(weather_schema.fields.iter().any(|f| f.name == "unit" && !f.required));
    }
}
