//! Bridge protocol for postMessage JSON-RPC communication (SEP-1865)
//!
//! Defines the JSON-RPC 2.0 message types exchanged between the host
//! and embedded UI apps via window.postMessage.
//!
//! The server's role is primarily to declare UI resources and serve HTML.
//! These types are provided so that host-side implementations can construct
//! and parse bridge messages correctly.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::types::{DisplayMode, UiPermissions, UiResourceCsp};

/// Capabilities declared by a UI app during initialization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpUiAppCapabilities {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_display_modes: Option<Vec<DisplayMode>>,
}

/// View -> Host: initialize the bridge connection
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiInitializeParams {
    pub protocol_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub app_capabilities: Option<McpUiAppCapabilities>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_info: Option<AppInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
}

/// Host -> View: response to ui/initialize
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiInitializeResult {
    pub protocol_version: String,
    pub host_capabilities: Value,
    pub host_info: AppInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_context: Option<HostContext>,
}

/// Context the host provides to the UI about its environment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HostContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_info: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub theme: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub styles: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display_mode: Option<DisplayMode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_dimensions: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub locale: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_zone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub platform: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_display_modes: Option<Vec<DisplayMode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_capabilities: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safe_area_insets: Option<Value>,
}

/// Host -> View: complete tool arguments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInputParams {
    pub arguments: Value,
}

/// Host -> View: tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
    #[serde(rename = "_meta", skip_serializing_if = "Option::is_none")]
    pub meta: Option<Value>,
}

/// Host -> View: tool was cancelled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCancelledParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// View -> Host: request to open a URL externally
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenLinkParams {
    pub url: String,
}

/// View -> Host: inject a message into the chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageParams {
    pub role: String,
    pub content: Value,
}

/// View -> Host: request a display mode change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDisplayModeParams {
    pub mode: DisplayMode,
}

/// Host -> View: response to ui/request-display-mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDisplayModeResult {
    pub mode: DisplayMode,
}

/// View -> Host: update model context for next LLM turn
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpdateModelContextParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_content: Option<Value>,
}

/// View -> Host: content size changed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeChangedParams {
    pub width: u32,
    pub height: u32,
}

/// Host -> View: graceful teardown request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTeardownParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Sandbox proxy -> Host: resource HTML and policy for sandbox iframe
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxResourceReadyParams {
    pub html: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sandbox: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub csp: Option<UiResourceCsp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<UiPermissions>,
}

/// All bridge method names as constants
pub mod methods {
    // Requests (require response)
    pub const INITIALIZE: &str = "ui/initialize";
    pub const OPEN_LINK: &str = "ui/open-link";
    pub const MESSAGE: &str = "ui/message";
    pub const REQUEST_DISPLAY_MODE: &str = "ui/request-display-mode";
    pub const UPDATE_MODEL_CONTEXT: &str = "ui/update-model-context";
    pub const RESOURCE_TEARDOWN: &str = "ui/resource-teardown";

    // Notifications (fire-and-forget)
    pub const INITIALIZED: &str = "ui/notifications/initialized";
    pub const TOOL_INPUT_PARTIAL: &str = "ui/notifications/tool-input-partial";
    pub const TOOL_INPUT: &str = "ui/notifications/tool-input";
    pub const TOOL_RESULT: &str = "ui/notifications/tool-result";
    pub const TOOL_CANCELLED: &str = "ui/notifications/tool-cancelled";
    pub const SIZE_CHANGED: &str = "ui/notifications/size-changed";
    pub const HOST_CONTEXT_CHANGED: &str = "ui/notifications/host-context-changed";

    // Sandbox proxy notifications
    pub const SANDBOX_PROXY_READY: &str = "ui/notifications/sandbox-proxy-ready";
    pub const SANDBOX_RESOURCE_READY: &str = "ui/notifications/sandbox-resource-ready";
}
