// @dive-file: Feature-gated sandbox surface re-export for chevalier-agentic consumers.
// @dive-rel: Mirrors src/mcp/mod.rs style by exposing optional integration behind a feature flag.
// @dive-rel: Re-exports chevalier-sandbox so downstream APIs import via chevalier_agentic::sandbox.

pub use chevalier_sandbox::*;
