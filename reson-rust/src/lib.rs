//! # Reson - Agents are just functions
//!
//! Production-grade LLM agent framework with structured outputs,
//! multi-provider support, and native tool calling.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use reson::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Coming soon: agentic macro support
//!     Ok(())
//! }
//! ```

pub mod error;
pub mod retry;
pub mod runtime;
pub mod types;

// Module declarations (implementations coming in phases)
pub mod providers;
pub mod parsers;
pub mod schema;
pub mod storage;
pub mod tools;
pub mod templating;
pub mod utils;

// Re-export proc macros from reson-macros crate
pub use reson_macros::{agentic, agentic_generator, Deserializable, Tool};

// Prelude for convenient imports
pub mod prelude {
    pub use crate::error::{Error, Result};
    pub use crate::runtime::{Runtime, ContextApi};
    pub use crate::types::{
        CacheMarker, ChatMessage, ChatRole, CreateResult, Provider, ReasoningSegment, TokenUsage,
        ToolCall, ToolResult,
    };
}
