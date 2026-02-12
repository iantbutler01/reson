//! MCP Apps Extension (SEP-1865) - Interactive HTML UIs in MCP
//!
//! This module implements the MCP Apps extension which allows tools to return
//! interactive HTML UIs that are rendered in iframes within the conversation.

mod types;
mod resource;
mod bridge;

pub use types::{
    DisplayMode, UiPermissions, UiResource, UiResourceCsp, UiResourceMeta, UiToolMeta,
    Visibility, ui_uri, EXTENSION_ID, MCP_APP_MIME_TYPE, UI_SCHEME,
};
pub use resource::UiResourceRegistry;
pub use bridge::{
    methods as bridge_methods, AppInfo, HostContext, McpUiAppCapabilities, MessageParams,
    OpenLinkParams, RequestDisplayModeParams, RequestDisplayModeResult, ResourceTeardownParams,
    SandboxResourceReadyParams, SizeChangedParams, ToolCancelledParams, ToolInputParams,
    ToolResultParams, UiInitializeParams, UiInitializeResult, UpdateModelContextParams,
};
