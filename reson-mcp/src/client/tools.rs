//! Tool-related types and utilities

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Information about a tool available from an MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInfo {
    /// The name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: Option<String>,
    /// JSON Schema for the tool's input parameters
    pub input_schema: Value,
}

impl From<&rmcp::model::Tool> for ToolInfo {
    fn from(tool: &rmcp::model::Tool) -> Self {
        Self {
            name: tool.name.to_string(),
            description: tool.description.as_ref().map(|s| s.to_string()),
            input_schema: serde_json::to_value(&tool.input_schema).unwrap_or(Value::Object(Default::default())),
        }
    }
}
