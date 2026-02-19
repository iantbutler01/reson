//! UI resource registry
//!
//! Stores registered UI resources and provides lookups for serving
//! via MCP `resources/list` and `resources/read`.

use std::collections::HashMap;
use std::sync::Arc;

use rmcp::model::{Annotated, Meta, RawResource, ReadResourceResult, Resource, ResourceContents};
use serde_json::{json, Value};
use url::Url;

use super::types::{UiResource, UiResourceMeta, MCP_APP_MIME_TYPE};

/// Registry of UI resources available to MCP hosts.
///
/// Stores `UiResource` instances keyed by their `ui://` URI and provides
/// conversion to rmcp protocol types for `resources/list` and `resources/read`.
#[derive(Debug, Clone, Default)]
pub struct UiResourceRegistry {
    resources: HashMap<String, Arc<UiResource>>,
}

impl UiResourceRegistry {
    /// Create an empty registry
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a UI resource. Replaces any existing resource with the same URI.
    pub fn insert(&mut self, resource: UiResource) {
        self.resources
            .insert(resource.uri.to_string(), Arc::new(resource));
    }

    /// Look up a resource by URI string
    pub fn get(&self, uri: &str) -> Option<&Arc<UiResource>> {
        self.resources.get(uri)
    }

    /// Look up a resource by parsed URL
    pub fn get_by_url(&self, uri: &Url) -> Option<&Arc<UiResource>> {
        self.resources.get(uri.as_str())
    }

    /// Number of registered resources
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Whether the registry is empty
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// Convert all registered resources to rmcp `Resource` values for `resources/list`.
    ///
    /// Each resource is returned with its `ui://` URI, name, description, and
    /// the MCP App MIME type. The `_meta.ui` field carries CSP, permissions,
    /// and display preferences per SEP-1865.
    pub fn list_resources(&self) -> Vec<Resource> {
        self.resources
            .values()
            .map(|r| resource_to_mcp(r))
            .collect()
    }

    /// Read a single resource by URI, returning rmcp's `ReadResourceResult`.
    ///
    /// Returns `None` if the URI is not registered.
    pub fn read_resource(&self, uri: &str) -> Option<ReadResourceResult> {
        let resource = self.resources.get(uri)?;
        Some(read_resource_result(resource))
    }
}

/// Convert a `UiResource` to an rmcp `Resource` for listing.
fn resource_to_mcp(r: &UiResource) -> Resource {
    let mut meta = serde_json::Map::new();
    if let Some(ui_meta) = &r.meta {
        meta.insert("ui".to_string(), ui_meta_to_value(ui_meta));
    }

    Annotated {
        raw: RawResource {
            uri: r.uri.to_string(),
            name: r.name.clone(),
            title: None,
            description: r.description.clone(),
            mime_type: Some(MCP_APP_MIME_TYPE.to_string()),
            size: None,
            icons: None,
            meta: if meta.is_empty() {
                None
            } else {
                Some(Meta(meta))
            },
        },
        annotations: None,
    }
}

/// Build a `ReadResourceResult` for a `UiResource`.
fn read_resource_result(r: &UiResource) -> ReadResourceResult {
    let mut content_meta = serde_json::Map::new();
    if let Some(ui_meta) = &r.meta {
        content_meta.insert("ui".to_string(), ui_meta_to_value(ui_meta));
    }

    ReadResourceResult {
        contents: vec![ResourceContents::TextResourceContents {
            uri: r.uri.to_string(),
            mime_type: Some(MCP_APP_MIME_TYPE.to_string()),
            text: r.content.clone(),
            meta: if content_meta.is_empty() {
                None
            } else {
                Some(Meta(content_meta))
            },
        }],
    }
}

/// Serialize `UiResourceMeta` to a JSON `Value` for the `_meta.ui` field.
fn ui_meta_to_value(meta: &UiResourceMeta) -> Value {
    // Use serde to get the full representation, which already has
    // camelCase renaming and skip_serializing_if applied.
    serde_json::to_value(meta).unwrap_or(json!({}))
}
