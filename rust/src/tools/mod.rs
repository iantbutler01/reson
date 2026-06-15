//! Tool calling system
//!
//! Tool trait, registry, and execution logic.

use async_trait::async_trait;
use futures::future::BoxFuture;
use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::schema::SchemaGenerator;

pub mod execution;

/// Trait for callable tools
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name for LLM
    fn tool_name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Execute the tool
    fn execute(&self) -> BoxFuture<'_, Result<String>>;

    /// Get schema for this tool (provider-specific)
    fn schema(&self, generator: &dyn SchemaGenerator) -> serde_json::Value;
}

/// Tool registry for managing available tools
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register<T: Tool + 'static>(&mut self, tool: T) -> Result<()> {
        let name = tool.tool_name().to_string();

        if self.tools.contains_key(&name) {
            return Err(Error::NonRetryable(format!(
                "Tool '{}' is already registered",
                name
            )));
        }

        self.tools.insert(name, Box::new(tool));
        Ok(())
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    /// Get all tool names
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Generate schemas for all tools
    pub fn generate_schemas(&self, generator: &dyn SchemaGenerator) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| tool.schema(generator))
            .collect()
    }

    /// Check if a tool exists
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get the number of registered tools
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTool {
        name: String,
        description: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn tool_name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            &self.description
        }

        fn execute(&self) -> BoxFuture<'_, Result<String>> {
            Box::pin(async { Ok("mock result".to_string()) })
        }

        fn schema(&self, _generator: &dyn SchemaGenerator) -> serde_json::Value {
            serde_json::json!({
                "name": self.name,
                "description": self.description,
            })
        }
    }

    #[test]
    fn test_registry_new() {
        let registry = ToolRegistry::new();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_register() {
        let mut registry = ToolRegistry::new();
        let tool = MockTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
        };

        let result = registry.register(tool);
        assert!(result.is_ok());
        assert_eq!(registry.len(), 1);
        assert!(registry.contains("test_tool"));
    }

    #[test]
    fn test_registry_register_duplicate() {
        let mut registry = ToolRegistry::new();

        let tool1 = MockTool {
            name: "test_tool".to_string(),
            description: "First".to_string(),
        };
        registry.register(tool1).unwrap();

        let tool2 = MockTool {
            name: "test_tool".to_string(),
            description: "Second".to_string(),
        };
        let result = registry.register(tool2);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already registered"));
    }

    #[test]
    fn test_registry_get() {
        let mut registry = ToolRegistry::new();
        let tool = MockTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
        };
        registry.register(tool).unwrap();

        let retrieved = registry.get("test_tool");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().tool_name(), "test_tool");
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = ToolRegistry::new();
        let retrieved = registry.get("nonexistent");
        assert!(retrieved.is_none());
    }

    #[test]
    fn test_registry_tool_names() {
        let mut registry = ToolRegistry::new();

        registry
            .register(MockTool {
                name: "tool1".to_string(),
                description: "First".to_string(),
            })
            .unwrap();

        registry
            .register(MockTool {
                name: "tool2".to_string(),
                description: "Second".to_string(),
            })
            .unwrap();

        let mut names = registry.tool_names();
        names.sort();

        assert_eq!(names, vec!["tool1", "tool2"]);
    }

    #[tokio::test]
    async fn test_tool_execute() {
        let tool = MockTool {
            name: "test_tool".to_string(),
            description: "A test tool".to_string(),
        };

        let result = tool.execute().await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "mock result");
    }
}
