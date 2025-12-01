//! Tool execution logic
//!
//! Handles executing tools from ToolCall objects.

use crate::error::{Error, Result};
use crate::tools::{Tool, ToolRegistry};
use crate::types::ToolCall;

/// Execute a tool from a ToolCall
///
/// # Arguments
/// * `registry` - The tool registry containing registered tools
/// * `tool_call` - The tool call from the LLM
///
/// # Returns
/// The string result from tool execution
pub async fn execute_tool(registry: &ToolRegistry, tool_call: &ToolCall) -> Result<String> {
    let tool = registry
        .get(&tool_call.tool_name)
        .ok_or_else(|| Error::NonRetryable(format!("Tool '{}' not found", tool_call.tool_name)))?;

    // Execute the tool (arguments are already in tool_call.args)
    tool.execute().await
}

/// Execute multiple tools concurrently
///
/// # Arguments
/// * `registry` - The tool registry containing registered tools
/// * `tool_calls` - Multiple tool calls from the LLM
///
/// # Returns
/// Results from all tool executions (in order)
pub async fn execute_tools(
    registry: &ToolRegistry,
    tool_calls: &[ToolCall],
) -> Vec<Result<String>> {
    // Execute all tools concurrently
    let futures: Vec<_> = tool_calls
        .iter()
        .map(|tc| execute_tool(registry, tc))
        .collect();

    futures::future::join_all(futures).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::Tool;
    use async_trait::async_trait;
    use futures::future::BoxFuture;

    struct MockTool {
        name: String,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn tool_name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "Mock tool"
        }

        fn execute(&self) -> BoxFuture<'_, Result<String>> {
            Box::pin(async { Ok(format!("executed {}", self.name)) })
        }

        fn schema(&self, _generator: &dyn crate::schema::SchemaGenerator) -> serde_json::Value {
            serde_json::json!({"name": self.name})
        }
    }

    #[tokio::test]
    async fn test_execute_tool() {
        let mut registry = ToolRegistry::new();
        registry
            .register(MockTool {
                name: "test_tool".to_string(),
            })
            .unwrap();

        let tool_call = ToolCall {
            tool_use_id: "call_123".to_string(),
            tool_name: "test_tool".to_string(),
            args: serde_json::json!({}),
            raw_arguments: None,
            signature: None,
        };

        let result = execute_tool(&registry, &tool_call).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "executed test_tool");
    }

    #[tokio::test]
    async fn test_execute_tool_not_found() {
        let registry = ToolRegistry::new();

        let tool_call = ToolCall {
            tool_use_id: "call_123".to_string(),
            tool_name: "nonexistent".to_string(),
            args: serde_json::json!({}),
            raw_arguments: None,
            signature: None,
        };

        let result = execute_tool(&registry, &tool_call).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[tokio::test]
    async fn test_execute_tools_multiple() {
        let mut registry = ToolRegistry::new();
        registry
            .register(MockTool {
                name: "tool1".to_string(),
            })
            .unwrap();
        registry
            .register(MockTool {
                name: "tool2".to_string(),
            })
            .unwrap();

        let tool_calls = vec![
            ToolCall {
                tool_use_id: "call_1".to_string(),
                tool_name: "tool1".to_string(),
                args: serde_json::json!({}),
                raw_arguments: None,
                signature: None,
            },
            ToolCall {
                tool_use_id: "call_2".to_string(),
                tool_name: "tool2".to_string(),
                args: serde_json::json!({}),
                raw_arguments: None,
                signature: None,
            },
        ];

        let results = execute_tools(&registry, &tool_calls).await;
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert_eq!(results[0].as_ref().unwrap(), "executed tool1");
        assert_eq!(results[1].as_ref().unwrap(), "executed tool2");
    }

    #[tokio::test]
    async fn test_execute_tools_with_failures() {
        let mut registry = ToolRegistry::new();
        registry
            .register(MockTool {
                name: "tool1".to_string(),
            })
            .unwrap();

        let tool_calls = vec![
            ToolCall {
                tool_use_id: "call_1".to_string(),
                tool_name: "tool1".to_string(),
                args: serde_json::json!({}),
                raw_arguments: None,
                signature: None,
            },
            ToolCall {
                tool_use_id: "call_2".to_string(),
                tool_name: "nonexistent".to_string(),
                args: serde_json::json!({}),
                raw_arguments: None,
                signature: None,
            },
        ];

        let results = execute_tools(&registry, &tool_calls).await;
        assert_eq!(results.len(), 2);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }
}
