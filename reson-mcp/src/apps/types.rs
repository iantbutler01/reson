//! Types for MCP Apps extension (SEP-1865)
//!
//! Defines the core types for the MCP Apps UI extension including
//! resource metadata, CSP policies, permissions, and visibility controls.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

/// The MCP Apps extension identifier
pub const EXTENSION_ID: &str = "io.modelcontextprotocol/ui";

/// The URI scheme for MCP App resources
pub const UI_SCHEME: &str = "ui";

/// The MIME type for MCP App HTML resources
pub const MCP_APP_MIME_TYPE: &str = "text/html;profile=mcp-app";

/// Visibility of a tool to the model and/or UI app
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Visibility {
    /// Tool is visible to the LLM agent
    Model,
    /// Tool is callable from the iframe app on the same server
    App,
}

/// Build a `ui://` URI from a server name and resource name
///
/// Returns a properly constructed URI following RFC 3986 semantics.
pub fn ui_uri(server_name: &str, resource_name: &str) -> Url {
    // url crate handles custom schemes per RFC 3986
    Url::parse(&format!(
        "{}://{}/{}",
        UI_SCHEME, server_name, resource_name
    ))
    .expect("ui:// URI should always be valid")
}

/// Tool metadata that links a tool to a UI resource
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiToolMeta {
    /// URI pointing to the ui:// resource
    pub resource_uri: Url,
    /// Visibility of the tool
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<Vec<Visibility>>,
}

/// Content Security Policy domain allowlists for a UI resource
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiResourceCsp {
    /// Domains allowed for connect-src (fetch, XHR, WebSocket)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connect_domains: Option<Vec<String>>,
    /// Domains allowed for script-src, style-src, img-src, media-src, font-src
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resource_domains: Option<Vec<String>>,
    /// Domains allowed for frame-src
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_domains: Option<Vec<String>>,
    /// Domains allowed for base-uri
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_uri_domains: Option<Vec<String>>,
}

/// Permissions that a UI resource requests from the host
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiPermissions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub microphone: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub geolocation: Option<serde_json::Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clipboard_write: Option<serde_json::Map<String, Value>>,
}

/// Metadata attached to a UI resource's content response (_meta.ui on the resource)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UiResourceMeta {
    /// Content Security Policy domain allowlists
    #[serde(skip_serializing_if = "Option::is_none")]
    pub csp: Option<UiResourceCsp>,
    /// Requested iframe permissions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permissions: Option<UiPermissions>,
    /// Whether the host should render a visual border around the iframe
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefers_border: Option<bool>,
    /// Optional dedicated sandbox origin domain
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
}

/// Display mode for a UI resource
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DisplayMode {
    Inline,
    Fullscreen,
    Pip,
}

/// A UI resource that can be served to MCP hosts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiResource {
    /// The ui:// URI for this resource
    pub uri: Url,
    /// Human-readable name
    pub name: String,
    /// Description of what this UI shows
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// HTML content
    pub content: String,
    /// Resource metadata (CSP, permissions, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<UiResourceMeta>,
}

impl UiResource {
    /// Create a new UI resource from HTML content
    ///
    /// `server_name` is the MCP server identity (used as the host in `ui://server/name`).
    /// `name` is the resource identifier (used as the path).
    pub fn new(
        server_name: impl AsRef<str>,
        name: impl Into<String>,
        html: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let uri = ui_uri(server_name.as_ref(), &name);
        Self {
            uri,
            name,
            description: None,
            content: html.into(),
            meta: None,
        }
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Set CSP policy
    pub fn with_csp(mut self, csp: UiResourceCsp) -> Self {
        self.meta.get_or_insert_with(Default::default).csp = Some(csp);
        self
    }

    /// Set permissions
    pub fn with_permissions(mut self, permissions: UiPermissions) -> Self {
        self.meta.get_or_insert_with(Default::default).permissions = Some(permissions);
        self
    }

    /// Set border preference
    pub fn with_border(mut self, prefers_border: bool) -> Self {
        self.meta
            .get_or_insert_with(Default::default)
            .prefers_border = Some(prefers_border);
        self
    }
}
