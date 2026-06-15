//! MCP Apps Extension (SEP-1865) - Interactive HTML UIs in MCP
//!
//! This module implements the MCP Apps extension which allows tools to return
//! interactive HTML UIs that are rendered in iframes within the conversation.

mod bridge;
mod resource;
mod types;

pub use bridge::{
    methods as bridge_methods, AppInfo, HostContext, McpUiAppCapabilities, MessageParams,
    OpenLinkParams, RequestDisplayModeParams, RequestDisplayModeResult, ResourceTeardownParams,
    SandboxResourceReadyParams, SizeChangedParams, ToolCancelledParams, ToolInputParams,
    ToolResultParams, UiInitializeParams, UiInitializeResult, UpdateModelContextParams,
};
pub use resource::UiResourceRegistry;
pub use types::{
    ui_uri, DisplayMode, UiPermissions, UiResource, UiResourceCsp, UiResourceMeta, UiToolMeta,
    Visibility, EXTENSION_ID, MCP_APP_MIME_TYPE, UI_SCHEME,
};
