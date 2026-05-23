// @dive-file: Feature-gated sandbox surface re-export for reson-agentic consumers.
// @dive-rel: Mirrors src/mcp/mod.rs style by exposing optional integration behind a feature flag.
// @dive-rel: Re-exports reson-sandbox so downstream APIs import via reson_agentic::sandbox.

pub use reson_sandbox::*;
