# Reson Rust Rewrite - Comprehensive Specification

**Version:** 1.0
**Date:** 2025-10-06
**Status:** Draft for Review

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Goals](#project-goals)
3. [Architecture Overview](#architecture-overview)
4. [Module Structure](#module-structure)
5. [Core Type System](#core-type-system)
6. [Provider Abstraction Layer](#provider-abstraction-layer)
7. [Tool Calling System](#tool-calling-system)
8. [Streaming Architecture](#streaming-architecture)
9. [Storage Backend Design](#storage-backend-design)
10. [Error Handling Strategy](#error-handling-strategy)
11. [Async Runtime Design](#async-runtime-design)
12. [Macro System for Ergonomics](#macro-system-for-ergonomics)
13. [Dependencies](#dependencies)
14. [Migration Strategy](#migration-strategy)
15. [Testing Strategy](#testing-strategy)
16. [Performance Targets](#performance-targets)
17. [API Design Examples](#api-design-examples)
18. [Implementation Phases](#implementation-phases)
19. [Open Questions](#open-questions)

---

## Executive Summary

This document specifies the complete rewrite of the **reson** Python framework (~11K LOC) in Rust. Reson is a production-grade LLM agent framework that provides:

[THOUGHT we likely can't do decorators in rust right? Im fine with a different approach like moving it all to the runtime 'object' that users can 'instantiate' in whatever but only if we need to, do you think the macro approach will be feasible I like the idea a lot but I don't want to fight super hard to preserve it if its not a great fit break it down for me]
- **Decorator-based API** for defining agents as async functions
- **Structured outputs** via type-safe parsing (Deserializable/Pydantic)
- **Multi-provider support** (Anthropic, OpenAI, Google, Bedrock, OpenRouter)
- [SUGGESTION ditch the xml based parsing and seoly focus on native provider APIS]
- **Dual tool calling**: XML-based (universal) + native provider APIs
- **Streaming support** with progressive parsing
- **Pluggable storage** (Memory, Redis, PostgreSQL)
- **Production features**: OpenTelemetry tracing, cost tracking, fallback clients

The Rust rewrite will preserve all functionality while gaining:
- **10-100x performance** improvements (zero-copy parsing, native async)
- **Compile-time safety** (type-checked tool signatures, no runtime panics)
- **Lower memory footprint** (no GC, efficient data structures)
- **Better concurrency** (fearless async, no GIL)

---

## Project Goals

### Primary Goals
1. **Feature parity** with Python version (all 33+ key features preserved)
2. **API compatibility** (similar ergonomics, minimize learning curve)
3. **Production-ready** (comprehensive error handling, observability, testing)
4. **Performance** (10x faster parsing, 50x lower latency for streaming)
5. **Safety** (no panics, compile-time guarantees, exhaustive error handling)

### Secondary Goals
6. **Extensibility** (easy to add new providers, parsers, storage backends)
7. **Documentation** (comprehensive API docs, migration guide, examples)
8. **Ecosystem integration** (works with existing Rust async ecosystem)

### Non-Goals (for v1)
- Training data collection (ART/OpenPipe integration) - defer to v2
- Lua scripting support (lupa) - not used in core
- Python interop (pyo3 bindings) - post v1 if needed

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      User Application                        │
│                    (Rust async functions)                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       │ #[agentic] macro
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                         Runtime                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ • Tool registry (HashMap<String, Box<dyn Tool>>)     │   │
│  │ • Context/state (Arc<dyn Store>)                     │   │
│  │ • Message accumulator (Vec<ChatMessage>)             │   │
│  │ • Reasoning segments (Vec<ReasoningSegment>)         │   │
│  │ • Return type (TypeInfo)                             │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────┬───────────────────────────────────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
    ┌────────┐   ┌─────────┐   ┌────────┐
    │ Parser │   │ Provider│   │ Store  │
    │ System │   │ Clients │   │Backends│
    └────────┘   └─────────┘   └────────┘
         │             │             │
         │             ▼             │
         │     ┌──────────────┐     │
         │     │ Schema Gen   │     │
         │     └──────────────┘     │
         │                          │
         └──────────┬───────────────┘
                    │
            ┌───────┴────────┐
            │                │
            ▼                ▼
    ┌──────────────┐  ┌──────────────┐
    │ HTTP Client  │  │ OpenTelemetry│
    │  (reqwest)   │  │   Tracing    │
    └──────────────┘  └──────────────┘
```

### Key Principles

1. **Trait-based abstraction**: All core components implement traits for extensibility
2. **Zero-copy where possible**: Use `&str` and `Bytes` to minimize allocations [THOUGHT -- I'm fine with not striving for perfect 0 copy if it makes rust a bitch to get working I'd rather a bunch of clones first and we can work backwards to zero copy after a working impl]
3. **Compile-time safety**: Leverage Rust's type system to catch errors at compile time
4. **Async-first**: Built on `tokio`, all I/O is async
5. **Error as values**: `Result<T, E>` everywhere, no panics in public API

---

## Module Structure

```
reson/
├── Cargo.toml
├── README.md
├── MIGRATION_GUIDE.md
├── src/
│   ├── lib.rs                      # Public API exports
│   ├── macros.rs                   # #[agentic] proc macro
│   ├── runtime.rs                  # Runtime orchestrator
│   ├── types.rs                    # Core types (ChatMessage, ToolResult, etc.)
│   ├── error.rs                    # Error types with thiserror
│   │
│   ├── providers/                  # LLM provider clients
│   │   ├── mod.rs                  # InferenceClient trait
│   │   ├── anthropic.rs            # AnthropicClient
│   │   ├── openai.rs               # OpenAIClient
│   │   ├── bedrock.rs              # BedrockClient
│   │   ├── google_genai.rs         # GoogleGenAIClient
│   │   ├── google_anthropic.rs     # VertexAI Claude
│   │   ├── openrouter.rs           # OpenRouterClient
│   │   └── tracing_client.rs       # TracingInferenceClient wrapper
│   │
│   ├── parsers/                    # Output parsing system
│   │   ├── mod.rs                  # OutputParser trait
│   │   ├── type_parser.rs          # Primary parser (serde-based)
│   │   ├── xml_parser.rs           # XML tool parsing
│   │   ├── native_tool_parser.rs   # Native tool format parser
│   │   └── streaming.rs            # Streaming parser utilities
│   │
│   ├── schema/                     # Schema generation
│   │   ├── mod.rs                  # SchemaGenerator trait
│   │   ├── anthropic.rs            # Anthropic tool schemas
│   │   ├── openai.rs               # OpenAI tool schemas
│   │   ├── google.rs               # Google tool schemas
│   │   └── introspection.rs        # Type introspection helpers
│   │
│   ├── storage/                    # Storage backends
│   │   ├── mod.rs                  # Store trait
│   │   ├── memory.rs               # MemoryStore
│   │   ├── redis.rs                # RedisStore
│   │   └── postgres.rs             # PostgresStore (JSONB)
│   │
│   ├── tools/                      # Tool calling system
│   │   ├── mod.rs                  # Tool trait, registry
│   │   ├── marshalling.rs          # Argument marshalling
│   │   └── validation.rs           # Tool signature validation
│   │
│   ├── templating/                 # Jinja-like templating
│   │   ├── mod.rs                  # Template trait
│   │   └── tera_impl.rs            # Tera-based implementation
│   │
│   └── utils/                      # Utilities
│       ├── backoff.rs              # Retry logic
│       ├── cost_tracking.rs        # Token/cost accumulation
│       ├── streaming.rs            # SSE parsing helpers
│       └── utf8.rs                 # UTF-8 boundary handling
│
├── reson-macros/                   # Proc macro crate
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                  # #[agentic], #[tool] macros
│
├── examples/                       # Usage examples
│   ├── basic.rs
│   ├── streaming.rs
│   ├── tools.rs
│   ├── native_tools.rs
│   ├── postgres_store.rs
│   └── reasoning.rs
│
└── tests/                          # Integration tests
    ├── test_anthropic.rs
    ├── test_openai.rs
    ├── test_google.rs
    ├── test_streaming.rs
    ├── test_tools.rs
    └── test_storage.rs
```

---

## Core Type System

### Design Philosophy

Rust doesn't have Python's dynamic typing or runtime introspection. We'll use:
1. **`serde`** for serialization/deserialization (like Pydantic)
2. **Derive macros** for automatic trait implementations
3. **Trait objects** for dynamic dispatch when needed
4. **Type state pattern** for compile-time state tracking

### Primary Types

#### `ChatMessage`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_marker: Option<CacheMarker>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_families: Option<Vec<String>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    #[serde(rename = "tool")]
    ToolResult,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheMarker {
    Ephemeral,
}
```

#### `ToolResult`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    #[serde(skip)]
    pub tool_obj: Option<Box<dyn Tool>>,
}

impl ToolResult {
    pub fn create<T: Tool + 'static>(tool_obj: T, result: String) -> Self {
        Self {
            tool_use_id: tool_obj.tool_use_id().to_string(),
            content: result,
            is_error: false,
            signature: None,
            tool_obj: Some(Box::new(tool_obj)),
        }
    }

    pub fn to_provider_format(&self, provider: Provider) -> serde_json::Value {
        match provider {
            Provider::Anthropic | Provider::Bedrock => {
                json!({
                    "type": "tool_result",
                    "tool_use_id": self.tool_use_id,
                    "content": self.content,
                    "is_error": self.is_error
                })
            }
            Provider::OpenAI | Provider::OpenRouter => {
                json!({
                    "role": "tool",
                    "tool_call_id": self.tool_use_id,
                    "content": self.content
                })
            }
            Provider::GoogleGenAI => {
                json!({
                    "functionResponse": {
                        "name": self.tool_obj.as_ref()
                            .map(|t| t.tool_name())
                            .unwrap_or("unknown"),
                        "response": {
                            "result": self.content
                        }
                    }
                })
            }
        }
    }
}
```

#### `ToolCall`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_use_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_arguments: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    #[serde(skip)]
    pub tool_obj: Option<Box<dyn Tool>>,
}

impl ToolCall {
    pub fn create(provider_format: serde_json::Value, provider: Provider) -> Result<Self> {
        match provider {
            Provider::Anthropic | Provider::Bedrock => {
                Ok(Self {
                    tool_use_id: provider_format["id"].as_str().unwrap().to_string(),
                    tool_name: provider_format["name"].as_str().unwrap().to_string(),
                    args: provider_format["input"].clone(),
                    raw_arguments: None,
                    signature: None,
                    tool_obj: None,
                })
            }
            Provider::OpenAI | Provider::OpenRouter => {
                let function = &provider_format["function"];
                Ok(Self {
                    tool_use_id: provider_format["id"].as_str().unwrap().to_string(),
                    tool_name: function["name"].as_str().unwrap().to_string(),
                    args: serde_json::from_str(function["arguments"].as_str().unwrap())?,
                    raw_arguments: Some(function["arguments"].as_str().unwrap().to_string()),
                    signature: None,
                    tool_obj: None,
                })
            }
            Provider::GoogleGenAI => {
                Ok(Self {
                    tool_use_id: uuid::Uuid::new_v4().to_string(),
                    tool_name: provider_format["functionCall"]["name"].as_str().unwrap().to_string(),
                    args: provider_format["functionCall"]["args"].clone(),
                    raw_arguments: None,
                    signature: None,
                    tool_obj: None,
                })
            }
        }
    }
}
```

#### `ReasoningSegment`

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningSegment {
    pub content: String,
    pub segment_index: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider_metadata: Option<serde_json::Value>,
}

impl ReasoningSegment {
    pub fn to_provider_format(&self, provider: Provider) -> serde_json::Value {
        match provider {
            Provider::Anthropic | Provider::Bedrock => {
                json!({
                    "type": "thinking",
                    "thinking": self.content
                })
            }
            Provider::OpenAI | Provider::OpenRouter => {
                json!({
                    "type": "reasoning",
                    "content": self.content
                })
            }
            Provider::GoogleGenAI => {
                json!({
                    "thought": true,
                    "text": self.content,
                    "thought_signature": self.signature
                })
            }
        }
    }
}
```

### Deserializable Trait (gasp-like)

```rust
use serde::{Deserialize, Serialize};

/// Trait for types that can be constructed from partial data (streaming)
pub trait Deserializable: Serialize + for<'de> Deserialize<'de> + Send + Sync {
    /// Construct from partial JSON data
    fn from_partial(partial: serde_json::Value) -> Result<Self, ParseError>
    where
        Self: Sized;

    /// Validate that the object is complete
    fn validate_complete(&self) -> Result<(), ValidationError>;

    /// Get field descriptions for schema generation
    fn field_descriptions() -> Vec<FieldDescription>
    where
        Self: Sized;
}

// Derive macro for automatic implementation
#[derive(Deserializable, Serialize, Deserialize)]
pub struct Person {
    pub name: String,
    pub age: u32,

    #[serde(default)]
    pub skills: Vec<String>,
}
```

### Tool Trait

```rust
/// Trait for callable tools
pub trait Tool: Send + Sync {
    /// Tool name for LLM
    fn tool_name(&self) -> &str;

    /// Tool description
    fn description(&self) -> &str;

    /// Tool use ID (for tracking)
    fn tool_use_id(&self) -> &str;

    /// Execute the tool
    fn execute(&self, runtime: &Runtime) -> BoxFuture<'_, Result<String>>;

    /// Get schema for this tool
    fn schema(&self, generator: &dyn SchemaGenerator) -> serde_json::Value;
}

// Example implementation via derive macro
#[derive(Tool)]
struct CalculateTool {
    #[tool(id)]
    tool_use_id: String,

    operation: String,
    a: f64,
    b: f64,
}

#[async_trait]
impl ToolExecutor for CalculateTool {
    async fn execute_impl(&self, _runtime: &Runtime) -> Result<String> {
        let result = match self.operation.as_str() {
            "add" => self.a + self.b,
            "subtract" => self.a - self.b,
            "multiply" => self.a * self.b,
            "divide" => self.a / self.b,
            _ => return Err(Error::InvalidOperation(self.operation.clone())),
        };
        Ok(result.to_string())
    }
}
```

---

## Provider Abstraction Layer

### InferenceClient Trait

```rust
use async_trait::async_trait;
use futures::stream::Stream;

#[async_trait]
pub trait InferenceClient: Send + Sync {
    /// Get a single generation (non-streaming)
    async fn get_generation(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerationConfig,
    ) -> Result<GenerationResponse>;

    /// Connect and listen for streaming responses
    async fn connect_and_listen(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;

    /// Get provider type
    fn provider(&self) -> Provider;

    /// Check if native tools are supported
    fn supports_native_tools(&self) -> bool;

    /// Set trace callback
    fn set_trace_callback(&mut self, callback: TraceCallback);
}

pub struct GenerationConfig {
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub tools: Option<Vec<serde_json::Value>>,
    pub native_tools: bool,
    pub reasoning_effort: Option<String>, // For o-series
    pub thinking_budget: Option<u32>,     // For Claude/Gemini
}

pub struct GenerationResponse {
    pub content: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning_segments: Vec<ReasoningSegment>,
    pub usage: TokenUsage,
}

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Reasoning(String),
    Signature(String),
    Content(String),
    ToolCallPartial(serde_json::Value),
    ToolCallComplete(ToolCall),
}

pub struct TokenUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}
```

### Provider Implementations

Each provider will implement `InferenceClient`:

```rust
pub struct AnthropicClient {
    api_key: String,
    client: reqwest::Client,
    trace_callback: Option<TraceCallback>,
}

#[async_trait]
impl InferenceClient for AnthropicClient {
    async fn get_generation(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerationConfig,
    ) -> Result<GenerationResponse> {
        let request = self.build_request(messages, config)?;
        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error(response).await?);
        }

        let body: serde_json::Value = response.json().await?;
        self.parse_response(body)
    }

    async fn connect_and_listen(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let request = self.build_request(messages, config)?;
        let response = self.client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request)
            .send()
            .await?;

        Ok(Box::pin(parse_sse_stream(response)))
    }

    fn provider(&self) -> Provider {
        Provider::Anthropic
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    fn set_trace_callback(&mut self, callback: TraceCallback) {
        self.trace_callback = Some(callback);
    }
}
```

### TracingInferenceClient Wrapper

```rust
pub struct TracingInferenceClient {
    inner: Box<dyn InferenceClient>,
    fallback: Option<Box<dyn InferenceClient>>,
    trace_id: u64,
    cost_accumulator: Arc<RwLock<InferenceCost>>,
    fallback_until: Option<Instant>,
}

impl TracingInferenceClient {
    pub fn new(inner: Box<dyn InferenceClient>) -> Self {
        Self {
            inner,
            fallback: None,
            trace_id: rand::random(),
            cost_accumulator: Arc::new(RwLock::new(InferenceCost::default())),
            fallback_until: None,
        }
    }

    pub fn with_fallback(mut self, fallback: Box<dyn InferenceClient>) -> Self {
        self.fallback = Some(fallback);
        self
    }

    async fn try_with_fallback<F, T>(&mut self, f: F) -> Result<T>
    where
        F: Fn(&dyn InferenceClient) -> BoxFuture<'_, Result<T>>,
    {
        // Check if we should use fallback
        if let Some(until) = self.fallback_until {
            if Instant::now() < until {
                if let Some(ref fallback) = self.fallback {
                    return f(fallback.as_ref()).await;
                }
            } else {
                self.fallback_until = None;
            }
        }

        // Try primary
        match f(self.inner.as_ref()).await {
            Ok(result) => Ok(result),
            Err(e) if e.is_retries_exceeded() => {
                // Switch to fallback for 5 minutes
                if let Some(ref fallback) = self.fallback {
                    self.fallback_until = Some(Instant::now() + Duration::from_secs(300));
                    f(fallback.as_ref()).await
                } else {
                    Err(e)
                }
            }
            Err(e) => Err(e),
        }
    }
}

#[async_trait]
impl InferenceClient for TracingInferenceClient {
    async fn get_generation(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerationConfig,
    ) -> Result<GenerationResponse> {
        let start = Instant::now();
        let result = self.try_with_fallback(|client| {
            Box::pin(client.get_generation(messages.clone(), config.clone()))
        }).await;

        if let Ok(ref response) = result {
            // Update cost tracking
            let mut cost = self.cost_accumulator.write().await;
            cost.add(&response.usage);

            // Emit trace
            if let Some(ref callback) = self.inner.trace_callback {
                callback(self.trace_id, &messages, response, &cost).await;
            }
        }

        result
    }

    // ... similar for connect_and_listen
}
```

---

## Tool Calling System

### Tool Registry

```rust
use std::collections::HashMap;

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    tool_types: HashMap<String, TypeInfo>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            tool_types: HashMap::new(),
        }
    }

    pub fn register<T: Tool + 'static>(
        &mut self,
        tool: T,
        tool_type: Option<TypeInfo>,
    ) -> Result<()> {
        let name = tool.tool_name().to_string();

        // Validate tool type if provided
        if let Some(ref type_info) = tool_type {
            self.validate_tool_signature(&tool, type_info)?;
        }

        self.tools.insert(name.clone(), Box::new(tool));
        if let Some(type_info) = tool_type {
            self.tool_types.insert(name, type_info);
        }

        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.get(name).map(|b| b.as_ref())
    }

    pub fn generate_schemas(&self, generator: &dyn SchemaGenerator) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| tool.schema(generator))
            .collect()
    }

    fn validate_tool_signature(&self, tool: &dyn Tool, type_info: &TypeInfo) -> Result<()> {
        // Extract function signature
        let func_params = extract_function_params(tool)?;
        let type_fields = type_info.fields();

        // Check for mismatches (warnings, not errors)
        let func_only: Vec<_> = func_params.difference(&type_fields).collect();
        let type_only: Vec<_> = type_fields.difference(&func_params).collect();

        if !func_only.is_empty() {
            log::warn!(
                "Tool '{}': function has params not in tool_type: {:?}",
                tool.tool_name(),
                func_only
            );
        }

        if !type_only.is_empty() {
            log::warn!(
                "Tool '{}': tool_type has fields not in function: {:?}",
                tool.tool_name(),
                type_only
            );
        }

        Ok(())
    }
}
```

### Smart Argument Marshalling

```rust
pub async fn execute_tool(
    registry: &ToolRegistry,
    tool_result: &ToolCall,
    runtime: &Runtime,
) -> Result<String> {
    let tool = registry
        .get(&tool_result.tool_name)
        .ok_or_else(|| Error::ToolNotFound(tool_result.tool_name.clone()))?;

    // Marshal arguments
    let marshalled_args = marshall_arguments(tool, &tool_result.args, registry)?;

    // Execute (tool already has args bound)
    tool.execute(runtime).await
}

fn marshall_arguments(
    tool: &dyn Tool,
    raw_args: &serde_json::Value,
    registry: &ToolRegistry,
) -> Result<()> {
    let type_info = registry
        .tool_types
        .get(tool.tool_name())
        .ok_or_else(|| Error::MissingTypeInfo(tool.tool_name().to_string()))?;

    // Try Deserializable first
    if type_info.is_deserializable() {
        let deserialized = type_info.deserialize_from(raw_args)?;
        // Update tool with deserialized args
        return Ok(());
    }

    // Try serde (Pydantic-like)
    if let Ok(deserialized) = serde_json::from_value(raw_args.clone()) {
        return Ok(());
    }

    // Fallback to raw JSON
    Ok(())
}
```

### XML Tool Parsing

```rust
use quick_xml::Reader;

pub struct XmlToolParser {
    tool_types: HashMap<String, TypeInfo>,
}

impl XmlToolParser {
    pub fn parse(&self, xml: &str) -> Result<Vec<Box<dyn Tool>>> {
        let mut reader = Reader::from_str(xml);
        let mut tools = Vec::new();

        loop {
            match reader.read_event()? {
                Event::Start(e) => {
                    let tag_name = String::from_utf8(e.name().as_ref().to_vec())?;

                    if let Some(type_info) = self.tool_types.get(&tag_name) {
                        let tool = self.parse_tool_element(&mut reader, type_info)?;
                        tools.push(tool);
                    }
                }
                Event::Eof => break,
                _ => {}
            }
        }

        Ok(tools)
    }

    fn parse_tool_element(
        &self,
        reader: &mut Reader<&[u8]>,
        type_info: &TypeInfo,
    ) -> Result<Box<dyn Tool>> {
        let mut fields = HashMap::new();

        loop {
            match reader.read_event()? {
                Event::Start(e) => {
                    let field_name = String::from_utf8(e.name().as_ref().to_vec())?;
                    let field_value = reader.read_text(e.name())?;
                    fields.insert(field_name, field_value.to_string());
                }
                Event::End(_) => break,
                _ => {}
            }
        }

        // Construct tool from fields
        type_info.construct_tool(fields)
    }
}
```

### Native Tool Parsing

```rust
use json_repair::repair_json;

pub struct NativeToolParser {
    tool_types: HashMap<String, TypeInfo>,
}

impl NativeToolParser {
    pub fn parse_tool(
        &self,
        provider_format: serde_json::Value,
        provider: Provider,
    ) -> Result<Box<dyn Tool>> {
        let tool_call = ToolCall::create(provider_format, provider)?;
        let tool_name = &tool_call.tool_name;

        let type_info = self.tool_types
            .get(tool_name)
            .ok_or_else(|| Error::UnknownTool(tool_name.clone()))?;

        // Try to deserialize with repair
        let repaired_json = repair_json(&tool_call.args.to_string())?;
        type_info.construct_tool_from_json(&repaired_json)
    }

    pub fn parse_partial_tool(
        &self,
        partial_json: &str,
        tool_name: &str,
    ) -> Result<Box<dyn Tool>> {
        let type_info = self.tool_types
            .get(tool_name)
            .ok_or_else(|| Error::UnknownTool(tool_name.to_string()))?;

        // Use Deserializable::from_partial for streaming
        let repaired = repair_json(partial_json)?;
        let partial_value: serde_json::Value = serde_json::from_str(&repaired)?;
        type_info.construct_partial_tool(partial_value)
    }
}
```

---

## Streaming Architecture

### Server-Sent Events (SSE) Parsing

```rust
use futures::stream::{Stream, StreamExt};
use bytes::Bytes;

pub fn parse_sse_stream(
    response: reqwest::Response,
) -> impl Stream<Item = Result<StreamChunk>> {
    let byte_stream = response.bytes_stream();

    byte_stream
        .map(|chunk_result| {
            chunk_result.map_err(|e| Error::Network(e))
        })
        .flat_map(|chunk_result| {
            futures::stream::iter(parse_sse_chunk(chunk_result))
        })
}

fn parse_sse_chunk(chunk: Result<Bytes>) -> Vec<Result<StreamChunk>> {
    let bytes = match chunk {
        Ok(b) => b,
        Err(e) => return vec![Err(e)],
    };

    let text = match std::str::from_utf8(&bytes) {
        Ok(t) => t,
        Err(e) => return vec![Err(Error::Utf8(e))],
    };

    text.lines()
        .filter(|line| line.starts_with("data: "))
        .filter_map(|line| {
            let data = &line[6..]; // Skip "data: "
            if data == "[DONE]" {
                return None;
            }

            match serde_json::from_str::<serde_json::Value>(data) {
                Ok(json) => Some(Ok(parse_chunk_json(json))),
                Err(e) => Some(Err(Error::Json(e))),
            }
        })
        .collect()
}

fn parse_chunk_json(json: serde_json::Value) -> StreamChunk {
    // Provider-specific parsing logic
    match json["type"].as_str() {
        Some("content_block_delta") => {
            StreamChunk::Content(json["delta"]["text"].as_str().unwrap().to_string())
        }
        Some("thinking_delta") => {
            StreamChunk::Reasoning(json["delta"]["thinking"].as_str().unwrap().to_string())
        }
        // ... more cases
        _ => StreamChunk::Content("".to_string()),
    }
}
```

### Progressive Tool Call Accumulation

```rust
pub struct ToolCallAccumulator {
    current_calls: HashMap<usize, PartialToolCall>,
}

#[derive(Clone)]
struct PartialToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self {
            current_calls: HashMap::new(),
        }
    }

    pub fn feed_delta(&mut self, delta: serde_json::Value) -> Option<StreamChunk> {
        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            for tc in tool_calls {
                let index = tc["index"].as_u64()? as usize;

                let call = self.current_calls.entry(index).or_insert_with(|| {
                    PartialToolCall {
                        id: tc["id"].as_str().unwrap_or_default().to_string(),
                        name: tc["function"]["name"].as_str().unwrap_or_default().to_string(),
                        arguments: String::new(),
                    }
                });

                if let Some(args_delta) = tc["function"]["arguments"].as_str() {
                    call.arguments.push_str(args_delta);
                }

                // Return partial tool call
                return Some(StreamChunk::ToolCallPartial(json!({
                    "id": call.id,
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments
                    }
                })));
            }
        }
        None
    }

    pub fn finalize(self) -> Vec<ToolCall> {
        self.current_calls
            .into_iter()
            .map(|(_, call)| ToolCall {
                tool_use_id: call.id,
                tool_name: call.name,
                args: serde_json::from_str(&call.arguments).unwrap_or_default(),
                raw_arguments: Some(call.arguments),
                signature: None,
                tool_obj: None,
            })
            .collect()
    }
}
```

### UTF-8 Boundary Handling

```rust
pub struct Utf8Buffer {
    buffer: String,
}

impl Utf8Buffer {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    pub fn push(&mut self, chunk: &str) {
        self.buffer.push_str(chunk);
    }

    pub fn try_parse<T>(&mut self, parser: impl Fn(&str) -> Result<T>) -> Result<T> {
        // Try to parse with current buffer
        match parser(&self.buffer) {
            Ok(result) => {
                self.buffer.clear();
                Ok(result)
            }
            Err(e) if e.is_incomplete() => {
                // Keep buffering
                Err(e)
            }
            Err(e) => {
                // Real error, try to recover by truncating to valid UTF-8
                self.buffer = truncate_to_valid_utf8(&self.buffer);
                Err(e)
            }
        }
    }
}

pub fn truncate_to_valid_utf8(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut end = bytes.len();

    while end > 0 {
        match std::str::from_utf8(&bytes[..end]) {
            Ok(valid) => return valid.to_string(),
            Err(_) => end -= 1,
        }
    }

    String::new()
}
```

---

## Storage Backend Design

### Store Trait

```rust
#[async_trait]
pub trait Store: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<serde_json::Value>>;
    async fn set(&self, key: &str, value: serde_json::Value) -> Result<()>;
    async fn delete(&self, key: &str) -> Result<()>;
    async fn keys(&self) -> Result<Vec<String>>;
    async fn clear(&self) -> Result<()>;

    // Pub/sub for mailbox system
    async fn publish_to_mailbox(&self, mailbox_id: &str, value: serde_json::Value) -> Result<()>;
    async fn get_message(&self, mailbox_id: &str, timeout: Option<Duration>) -> Result<Option<serde_json::Value>>;

    // Namespacing
    fn with_prefix(&self, prefix: &str) -> PrefixedStore;
    fn with_suffix(&self, suffix: &str) -> SuffixedStore;
}
```

### Memory Store

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct MemoryStore {
    data: Arc<RwLock<HashMap<String, serde_json::Value>>>,
    mailboxes: Arc<RwLock<HashMap<String, tokio::sync::mpsc::UnboundedSender<serde_json::Value>>>>,
}

#[async_trait]
impl Store for MemoryStore {
    async fn get(&self, key: &str) -> Result<Option<serde_json::Value>> {
        let data = self.data.read().await;
        Ok(data.get(key).cloned())
    }

    async fn set(&self, key: &str, value: serde_json::Value) -> Result<()> {
        let mut data = self.data.write().await;
        data.insert(key.to_string(), value);
        Ok(())
    }

    async fn publish_to_mailbox(&self, mailbox_id: &str, value: serde_json::Value) -> Result<()> {
        let mailboxes = self.mailboxes.read().await;
        if let Some(sender) = mailboxes.get(mailbox_id) {
            sender.send(value).map_err(|_| Error::MailboxClosed)?;
        }
        Ok(())
    }

    async fn get_message(&self, mailbox_id: &str, timeout: Option<Duration>) -> Result<Option<serde_json::Value>> {
        let mut mailboxes = self.mailboxes.write().await;
        let receiver = mailboxes
            .entry(mailbox_id.to_string())
            .or_insert_with(|| tokio::sync::mpsc::unbounded_channel().0);

        // Create receiver
        let mut rx = receiver.subscribe();
        drop(mailboxes); // Release lock

        match timeout {
            Some(duration) => {
                tokio::time::timeout(duration, rx.recv())
                    .await
                    .ok()
                    .flatten()
                    .ok_or(Error::Timeout)
            }
            None => rx.recv().await.ok_or(Error::MailboxClosed),
        }
    }
}
```

### Redis Store

```rust
use redis::aio::ConnectionManager;

pub struct RedisStore {
    client: ConnectionManager,
    prefix: String,
}

#[async_trait]
impl Store for RedisStore {
    async fn get(&self, key: &str) -> Result<Option<serde_json::Value>> {
        let full_key = format!("{}:{}", self.prefix, key);
        let value: Option<String> = redis::cmd("GET")
            .arg(&full_key)
            .query_async(&mut self.client.clone())
            .await?;

        match value {
            Some(v) => Ok(Some(serde_json::from_str(&v)?)),
            None => Ok(None),
        }
    }

    async fn set(&self, key: &str, value: serde_json::Value) -> Result<()> {
        let full_key = format!("{}:{}", self.prefix, key);
        let serialized = serde_json::to_string(&value)?;

        redis::cmd("SET")
            .arg(&full_key)
            .arg(&serialized)
            .query_async(&mut self.client.clone())
            .await?;

        Ok(())
    }

    async fn publish_to_mailbox(&self, mailbox_id: &str, value: serde_json::Value) -> Result<()> {
        let channel = format!("mailbox:{}", mailbox_id);
        let serialized = serde_json::to_string(&value)?;

        redis::cmd("PUBLISH")
            .arg(&channel)
            .arg(&serialized)
            .query_async(&mut self.client.clone())
            .await?;

        Ok(())
    }

    async fn get_message(&self, mailbox_id: &str, timeout: Option<Duration>) -> Result<Option<serde_json::Value>> {
        let channel = format!("mailbox:{}", mailbox_id);
        let mut pubsub = self.client.clone().into_pubsub();
        pubsub.subscribe(&channel).await?;

        let msg = match timeout {
            Some(duration) => {
                tokio::time::timeout(duration, pubsub.on_message().next())
                    .await?
                    .ok_or(Error::Timeout)?
            }
            None => pubsub.on_message().next().await.ok_or(Error::MailboxClosed)?,
        };

        let payload: String = msg.get_payload()?;
        Ok(Some(serde_json::from_str(&payload)?))
    }
}
```

### PostgreSQL Store

```rust
use sqlx::{PgPool, Row};

pub struct PostgresStore {
    pool: PgPool,
    table: String,
    column: String,
}

impl PostgresStore {
    pub async fn new(dsn: &str, table: &str, column: &str) -> Result<Self> {
        let pool = PgPool::connect(dsn).await?;

        // Create table if not exists
        sqlx::query(&format!(
            "CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                {} JSONB NOT NULL DEFAULT '{{}}'::jsonb
            )",
            table, column
        ))
        .execute(&pool)
        .await?;

        // Ensure at least one row exists
        sqlx::query(&format!(
            "INSERT INTO {} ({}) SELECT '{{}}'::jsonb WHERE NOT EXISTS (SELECT 1 FROM {})",
            table, column, table
        ))
        .execute(&pool)
        .await?;

        Ok(Self {
            pool,
            table: table.to_string(),
            column: column.to_string(),
        })
    }
}

#[async_trait]
impl Store for PostgresStore {
    async fn get(&self, key: &str) -> Result<Option<serde_json::Value>> {
        let row = sqlx::query(&format!(
            "SELECT {}->$1 AS value FROM {} LIMIT 1",
            self.column, self.table
        ))
        .bind(key)
        .fetch_one(&self.pool)
        .await?;

        Ok(row.try_get("value").ok())
    }

    async fn set(&self, key: &str, value: serde_json::Value) -> Result<()> {
        let value_str = serde_json::to_string(&value)?;

        sqlx::query(&format!(
            "UPDATE {} SET {} = jsonb_set({}, $1, $2, true)",
            self.table, self.column, self.column
        ))
        .bind(vec![key])
        .bind(&value_str)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn keys(&self) -> Result<Vec<String>> {
        let rows = sqlx::query(&format!(
            "SELECT jsonb_object_keys({}) AS key FROM {}",
            self.column, self.table
        ))
        .fetch_all(&self.pool)
        .await?;

        Ok(rows.into_iter().map(|row| row.get("key")).collect())
    }

    // Note: Mailbox not implemented for Postgres (use Redis or Memory for that)
}
```

---

## Error Handling Strategy

### Error Type Hierarchy

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Context length exceeded: {0}")]
    ContextLengthExceeded(String),

    #[error("Retries exceeded")]
    RetriesExceeded,

    #[error("Non-retryable error: {0}")]
    NonRetryable(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),

    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl Error {
    pub fn is_retryable(&self) -> bool {
        matches!(self,
            Error::Inference(_) |
            Error::Network(_) |
            Error::Io(_)
        )
    }

    pub fn is_retries_exceeded(&self) -> bool {
        matches!(self, Error::RetriesExceeded)
    }

    pub fn is_incomplete(&self) -> bool {
        matches!(self, Error::Parse(msg) if msg.contains("incomplete"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;
```

### Backoff Strategy

```rust
use backoff::{ExponentialBackoff, backoff::Backoff};

pub async fn retry_with_backoff<F, Fut, T>(f: F) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<T>>,
{
    let mut backoff = ExponentialBackoff {
        max_elapsed_time: Some(Duration::from_secs(60)),
        ..Default::default()
    };

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(e) if !e.is_retryable() => return Err(e),
            Err(_) => {
                match backoff.next_backoff() {
                    Some(duration) => tokio::time::sleep(duration).await,
                    None => return Err(Error::RetriesExceeded),
                }
            }
        }
    }
}

// Usage in provider implementations
#[async_trait]
impl InferenceClient for AnthropicClient {
    async fn get_generation(
        &self,
        messages: Vec<ChatMessage>,
        config: GenerationConfig,
    ) -> Result<GenerationResponse> {
        retry_with_backoff(|| async {
            self.get_generation_impl(messages.clone(), config.clone()).await
        }).await
    }
}
```

---

## Async Runtime Design

### Runtime Type

```rust
use tokio::sync::RwLock;

pub struct Runtime {
    // Configuration
    model: String,
    client: Arc<dyn InferenceClient>,
    store: Arc<dyn Store>,
    parser: Arc<dyn OutputParser>,

    // Tool registry
    tools: Arc<RwLock<ToolRegistry>>,

    // State tracking
    messages: Arc<RwLock<Vec<ChatMessage>>>,
    raw_response_accumulator: Arc<RwLock<Vec<String>>>,
    reasoning_accumulator: Arc<RwLock<Vec<String>>>,
    reasoning_segments: Arc<RwLock<Vec<ReasoningSegment>>>,
    current_reasoning_segment: Arc<RwLock<Option<ReasoningSegment>>>,

    // Return type info
    return_type: TypeInfo,

    // State tracking
    used: Arc<RwLock<bool>>,
}

impl Runtime {
    pub fn new(
        model: String,
        client: Arc<dyn InferenceClient>,
        store: Arc<dyn Store>,
        return_type: TypeInfo,
    ) -> Self {
        Self {
            model,
            client,
            store,
            parser: Arc::new(TypeParser::new()),
            tools: Arc::new(RwLock::new(ToolRegistry::new())),
            messages: Arc::new(RwLock::new(Vec::new())),
            raw_response_accumulator: Arc::new(RwLock::new(Vec::new())),
            reasoning_accumulator: Arc::new(RwLock::new(Vec::new())),
            reasoning_segments: Arc::new(RwLock::new(Vec::new())),
            current_reasoning_segment: Arc::new(RwLock::new(None)),
            return_type,
            used: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn tool<T: Tool + 'static>(&self, tool: T) {
        let mut registry = self.tools.write().await;
        registry.register(tool, None).unwrap();
    }

    pub async fn run<T: Deserializable>(&self, prompt: Option<&str>) -> Result<T> {
        // Mark as used
        *self.used.write().await = true;

        // Build messages
        let mut messages = self.messages.write().await;
        if let Some(prompt) = prompt {
            messages.push(ChatMessage {
                role: ChatRole::User,
                content: prompt.to_string(),
                cache_marker: None,
                model_families: None,
                signature: None,
            });
        }

        // Generate schemas
        let registry = self.tools.read().await;
        let schemas = registry.generate_schemas(self.client.schema_generator());

        // Call LLM
        let config = GenerationConfig {
            model: self.model.clone(),
            tools: if schemas.is_empty() { None } else { Some(schemas) },
            ..Default::default()
        };

        let response = self.client.get_generation(messages.clone(), config).await?;

        // Parse output
        self.parser.parse(&response.content, &self.return_type)
    }

    pub async fn run_stream<T: Deserializable>(
        &self,
        prompt: Option<&str>,
    ) -> Result<impl Stream<Item = Result<StreamEvent<T>>>> {
        // Similar to run() but returns stream
        // ...
    }

    pub async fn is_tool_call<T>(&self, result: &T) -> bool {
        // Check if result is a tool call
        // ...
    }

    pub async fn execute_tool<T: Tool>(&self, tool: T) -> Result<String> {
        let registry = self.tools.read().await;
        execute_tool(&registry, &tool, self).await
    }

    // Property accessors
    pub async fn raw_response(&self) -> String {
        self.raw_response_accumulator.read().await.join("")
    }

    pub async fn reasoning(&self) -> String {
        self.reasoning_accumulator.read().await.join("")
    }

    pub async fn reasoning_segments(&self) -> Vec<ReasoningSegment> {
        self.reasoning_segments.read().await.clone()
    }

    pub fn context(&self) -> ContextApi {
        ContextApi {
            store: self.store.clone(),
        }
    }
}

pub struct ContextApi {
    store: Arc<dyn Store>,
}

impl ContextApi {
    pub async fn get<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Result<Option<T>> {
        match self.store.get(key).await? {
            Some(value) => Ok(Some(serde_json::from_value(value)?)),
            None => Ok(None),
        }
    }

    pub async fn set<T: Serialize>(&self, key: &str, value: T) -> Result<()> {
        let json_value = serde_json::to_value(value)?;
        self.store.set(key, json_value).await
    }
}
```

---

## Macro System for Ergonomics

### `#[agentic]` Macro

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn agentic(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let fn_vis = &input_fn.sig.vis;
    let fn_inputs = &input_fn.sig.inputs;
    let fn_output = &input_fn.sig.output;
    let fn_block = &input_fn.block;

    // Parse attributes (model, store_cfg, etc.)
    let attrs = parse_agentic_attrs(attr);

    // Extract runtime parameter
    let runtime_param = find_runtime_param(fn_inputs);

    // Extract return type
    let return_type = extract_return_type(fn_output);

    // Generate wrapper function
    let expanded = quote! {
        #fn_vis async fn #fn_name(#fn_inputs) #fn_output {
            // Initialize runtime
            let runtime = ::reson::Runtime::builder()
                .model(#(attrs.model))
                .store(#(attrs.store))
                .return_type::<#return_type>()
                .build()
                .await?;

            // Auto-bind tools (if autobind=true)
            #(attrs.autobind_code)

            // Call user function
            let result = {
                #fn_block
            };

            // Validate runtime was used
            if !runtime.was_used().await {
                return Err(::reson::Error::RuntimeNotUsed);
            }

            result
        }
    };

    TokenStream::from(expanded)
}

// Example usage:
#[agentic(model = "anthropic:claude-3-opus-20240229")]
async fn extract_people(text: String, runtime: Runtime) -> Result<Vec<Person>> {
    runtime.run(Some(&format!("Extract people from: {}", text))).await
}
```

### `#[tool]` Derive Macro

```rust
#[proc_macro_derive(Tool, attributes(tool))]
pub fn derive_tool(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    // Generate Tool implementation
    let expanded = quote! {
        #[async_trait]
        impl ::reson::Tool for #name {
            fn tool_name(&self) -> &str {
                stringify!(#name)
            }

            fn description(&self) -> &str {
                // Extract from doc comments
                ""
            }

            fn tool_use_id(&self) -> &str {
                &self.tool_use_id
            }

            async fn execute(&self, runtime: &::reson::Runtime) -> ::reson::Result<String> {
                self.execute_impl(runtime).await
            }

            fn schema(&self, generator: &dyn ::reson::SchemaGenerator) -> serde_json::Value {
                generator.generate_schema::<Self>()
            }
        }
    };

    TokenStream::from(expanded)
}
```

### `#[deserializable]` Derive Macro

```rust
#[proc_macro_derive(Deserializable)]
pub fn derive_deserializable(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let expanded = quote! {
        impl ::reson::Deserializable for #name {
            fn from_partial(partial: serde_json::Value) -> ::reson::Result<Self> {
                // Generate partial construction logic
                // Handle missing fields with defaults
                Ok(serde_json::from_value(partial)?)
            }

            fn validate_complete(&self) -> ::reson::Result<()> {
                // Check all required fields are present
                Ok(())
            }

            fn field_descriptions() -> Vec<::reson::FieldDescription> {
                // Extract from field doc comments
                vec![]
            }
        }
    };

    TokenStream::from(expanded)
}
```

---

## Dependencies

### Core Dependencies

```toml
[dependencies]
# Async runtime
tokio = { version = "1.40", features = ["full"] }
async-trait = "0.1"
futures = "0.3"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# HTTP client
reqwest = { version = "0.12", features = ["json", "stream"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Retry logic
backoff = { version = "0.4", features = ["tokio"] }

# Logging
log = "0.4"
tracing = "0.1"
tracing-subscriber = "0.3"

# OpenTelemetry
opentelemetry = "0.24"
opentelemetry-sdk = "0.24"
opentelemetry-otlp = "0.17"

# AWS SDK (for Bedrock)
aws-config = "1.5"
aws-sdk-bedrockruntime = "1.46"

# Google SDKs
google-cloud-aiplatform = "0.1" # Vertex AI
# Note: May need to use tonic + protobuf for Google APIs

# Storage
redis = { version = "0.26", features = ["tokio-comp", "connection-manager"] }
sqlx = { version = "0.8", features = ["postgres", "runtime-tokio", "json"] }

# Templating
tera = "1.20"

# XML parsing
quick-xml = "0.36"

# JSON repair
json-repair = "0.1"

# UUID generation
uuid = { version = "1.10", features = ["v4"] }

# Utilities
bytes = "1.7"
parking_lot = "0.12" # Fast locks
```

### Proc Macro Dependencies

```toml
[dependencies]
proc-macro2 = "1.0"
quote = "1.0"
syn = { version = "2.0", features = ["full"] }
```

---

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Rust project structure (workspace with main crate + macro crate)
- [ ] Define core types (ChatMessage, ToolResult, ToolCall, ReasoningSegment)
- [ ] Implement Error types with thiserror
- [ ] Create InferenceClient trait
- [ ] Implement basic Runtime struct (no macro yet)
- [ ] Write unit tests for core types

### Phase 2: Provider Clients (Weeks 3-4)
- [ ] Implement AnthropicClient with get_generation()
- [ ] Implement OpenAIClient with get_generation()
- [ ] Add streaming support (connect_and_listen) for both
- [ ] Implement SSE parsing utilities
- [ ] Add backoff/retry logic
- [ ] Integration tests with real APIs (skip if no key)

### Phase 3: Storage Backends (Week 5)
- [ ] Implement MemoryStore
- [ ] Implement RedisStore
- [ ] Implement PostgresStore with JSONB
- [ ] Add namespacing (with_prefix/with_suffix)
- [ ] Add mailbox system (pub/sub)
- [ ] Tests for all storage backends

### Phase 4: Tool System (Weeks 6-7)
- [ ] Define Tool trait
- [ ] Implement ToolRegistry
- [ ] Create argument marshalling logic
- [ ] Implement XML tool parsing
- [ ] Implement native tool parsing
- [ ] Add tool signature validation
- [ ] Tests for tool execution

### Phase 5: Schema Generation (Week 8)
- [ ] Define SchemaGenerator trait
- [ ] Implement type introspection helpers
- [ ] Create AnthropicSchemaGenerator
- [ ] Create OpenAISchemaGenerator
- [ ] Create GoogleSchemaGenerator
- [ ] Tests for schema generation

### Phase 6: Parser System (Weeks 9-10)
- [ ] Define OutputParser trait
- [ ] Implement TypeParser (serde-based)
- [ ] Add streaming parser with UTF-8 handling
- [ ] Implement NativeToolParser
- [ ] Add prompt enhancement (schema injection)
- [ ] Tests for parsing (including partial objects)

### Phase 7: Macro System (Weeks 11-12)
- [ ] Create #[agentic] proc macro
- [ ] Add auto-binding logic
- [ ] Create #[tool] derive macro
- [ ] Create #[deserializable] derive macro
- [ ] Add compile-time validation
- [ ] Tests for macro expansion

### Phase 8: Additional Providers (Week 13)
- [ ] Implement BedrockClient
- [ ] Implement GoogleGenAIClient
- [ ] Implement GoogleAnthropicClient (Vertex)
- [ ] Implement OpenRouterClient
- [ ] Add TracingInferenceClient wrapper
- [ ] Tests for all providers

### Phase 9: Advanced Features (Week 14)
- [ ] Add reasoning segment tracking
- [ ] Implement prompt caching support
- [ ] Add cost tracking
- [ ] Implement fallback client pattern
- [ ] Add OpenTelemetry integration
- [ ] Tests for advanced features

### Phase 10: Templating (Week 15)
- [ ] Integrate Tera for Jinja-like templating
- [ ] Add {{return_type}} interpolation
- [ ] Support call context variables
- [ ] Tests for templating

### Phase 11: Documentation & Examples (Week 16)
- [ ] Write comprehensive API documentation
- [ ] Create migration guide from Python
- [ ] Write examples (basic, streaming, tools, etc.)
- [ ] Create README with quickstart
- [ ] Add inline code examples

### Phase 12: Polish & Release (Week 17)
- [ ] Performance benchmarking
- [ ] Memory leak detection (Valgrind/MIRI)
- [ ] Security audit (cargo-audit)
- [ ] CI/CD setup (GitHub Actions)
- [ ] Publish to crates.io

---

## Testing Strategy

### Unit Tests
- Core types serialization/deserialization
- Error type conversions
- Schema generation for various types
- Parser logic (XML, JSON, streaming)
- Tool signature validation
- Storage backend operations

### Integration Tests
- Real API calls (with skip if no credentials)
- End-to-end workflows (user prompt → LLM → parsed output)
- Streaming with progressive parsing
- Tool calling (XML and native)
- Storage backends with concurrency
- Fallback client behavior

### Property Tests (with `proptest`)
- Schema generation for arbitrary types
- Parser handles malformed input gracefully
- UTF-8 boundary handling never panics

### Benchmarks (with `criterion`)
- Parsing performance (XML vs native tools)
- Streaming throughput
- Schema generation time
- Storage backend latency

---

## Performance Targets

### Baseline (Python)
[THOUGHT: Again we can ditch the xml parser]
- XML parsing: ~50ms for 1KB output
- Streaming latency: ~100ms to first token
- Memory: ~50MB per Runtime instance

### Rust Goals
[THOUGHT: Again we can ditch the xml parser]
- **XML parsing**: <5ms for 1KB output (10x faster)
- **Streaming latency**: <10ms to first token (10x faster)
- **Memory**: <5MB per Runtime instance (10x smaller)
- **Throughput**: 1000+ requests/sec (single server)

---

## API Design Examples

### Basic Usage

```rust
use reson::{agentic, Runtime, Deserializable};

#[derive(Deserializable, Serialize, Deserialize)]
struct Person {
    name: String,
    age: u32,
    skills: Vec<String>,
}

#[agentic(model = "anthropic:claude-3-opus-20240229")]
async fn extract_people(text: String, runtime: Runtime) -> Result<Vec<Person>> {
    runtime.run(Some(&format!("Extract people from: {}", text))).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let people = extract_people("John is 30 and knows Python.".to_string()).await?;
    for person in people {
        println!("{}, {}: {:?}", person.name, person.age, person.skills);
    }
    Ok(())
}
```

### Streaming

```rust
#[agentic(model = "anthropic:claude-3-opus-20240229")]
async fn write_story(topic: String, runtime: Runtime) -> Result<String> {
    let mut result = String::new();
    let mut stream = runtime.run_stream::<String>(Some(&format!("Write a story about {}", topic))).await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Content(chunk) => {
                print!("{}", chunk);
                result.push_str(&chunk);
            }
            StreamEvent::Reasoning(reasoning) => {
                println!("[Reasoning: {}]", reasoning);
            }
            _ => {}
        }
    }

    Ok(result)
}
```

### Tools

```rust
#[derive(Tool, Deserializable)]
struct GetWeatherTool {
    #[tool(id)]
    tool_use_id: String,
    city: String,
}

#[async_trait]
impl ToolExecutor for GetWeatherTool {
    async fn execute_impl(&self, _runtime: &Runtime) -> Result<String> {
        // Fetch weather...
        Ok(format!("Sunny and 72°F in {}", self.city))
    }
}

#[agentic(model = "openrouter:openai/gpt-4o", native_tools = true)]
async fn plan_trip(destination: String, runtime: Runtime) -> Result<String> {
    runtime.tool(GetWeatherTool::descriptor()).await;

    let result = runtime.run::<String>(Some(&format!("Plan a trip to {}", destination))).await?;

    if runtime.is_tool_call(&result).await {
        let tool_result = runtime.execute_tool(result).await?;
        return Ok(format!("Tool executed: {}", tool_result));
    }

    Ok(result)
}
```

### Storage

```rust
#[agentic(
    model = "anthropic:claude-3-opus-20240229",
    store = PostgresStore::new("postgresql://...", "agent_data", "data")
)]
async fn conversational_agent(message: String, runtime: Runtime) -> Result<String> {
    let history: Vec<String> = runtime.context().get("conversation").await?.unwrap_or_default();
    history.push(format!("User: {}", message));

    let prompt = format!("{}\nAssistant:", history.join("\n"));
    let response = runtime.run::<String>(Some(&prompt)).await?;

    history.push(format!("Assistant: {}", response));
    runtime.context().set("conversation", history).await?;

    Ok(response)
}
```

---

## Implementation Phases

### Milestone 1: Core Infrastructure (End of Week 5)
**Deliverables:**
- Core types (ChatMessage, ToolResult, etc.)
- 2 provider implementations (Anthropic, OpenAI)
- 3 storage backends (Memory, Redis, Postgres)
- Basic Runtime (no macros)
- Error handling framework

**Success Criteria:**
- Can make API calls to Anthropic/OpenAI
- Can store/retrieve data from all backends
- All unit tests passing

### Milestone 2: Tool System (End of Week 8)
**Deliverables:**
- Tool trait and registry
- XML and native tool parsing
- Schema generation for all providers
- Tool signature validation

**Success Criteria:**
- Can register and execute tools
- XML and native tool calling work end-to-end
- Schema generation passes property tests

### Milestone 3: Parser & Streaming (End of Week 10)
**Deliverables:**
- Complete parser system (TypeParser, NativeToolParser)
- Streaming support with progressive parsing
- UTF-8 boundary handling
- Prompt enhancement

**Success Criteria:**
- Can parse complex nested types
- Streaming works with partial objects
- No panics on malformed input

### Milestone 4: Ergonomic API (End of Week 12)
**Deliverables:**
- #[agentic] macro
- #[tool] macro
- #[deserializable] macro
- Auto-binding logic

**Success Criteria:**
- API feels like Python version
- Macros provide helpful compile errors
- Examples compile and run

### Milestone 5: Feature Complete (End of Week 15)
**Deliverables:**
- All 8 providers implemented
- Reasoning support
- Fallback clients
- Cost tracking
- OpenTelemetry integration
- Templating

**Success Criteria:**
- Feature parity with Python version
- All integration tests passing
- Performance targets met

### Milestone 6: Production Ready (End of Week 17)
**Deliverables:**
- Comprehensive documentation
- Migration guide
- Examples
- CI/CD
- Published to crates.io

**Success Criteria:**
- API documentation at 100% coverage
- All benchmarks pass
- No clippy warnings
- Security audit clean

---

## Open Questions

### 1. Type Introspection
**Question:** How do we replicate Python's runtime introspection for function signatures?

**Options:**
- A) Use proc macros to capture type info at compile time
- B) Require manual schema definitions
- C) Use trait-based approach with associated types

**Recommendation:** A - Proc macros can capture full type info during expansion

### 2. Dynamic Tool Registration
**Question:** Python allows registering any callable at runtime. How do we support this in Rust?

**Options:**
- A) Use trait objects (`Box<dyn Tool>`)
- B) Require compile-time tool registration
- C) Use enum dispatch with macro-generated variants

**Recommendation:** A - Trait objects provide flexibility, minor perf cost acceptable

### 3. Deserializable Partial Construction
**Question:** gasp supports partial object construction for streaming. How do we replicate?

**Options:**
- A) Custom derive macro that generates `from_partial()` impl
- B) Use `serde(default)` and handle missing fields manually
- C) Use `Option<T>` for all fields, validate later

**Recommendation:** A + C - Derive macro generates validation, all fields are `Option<T>` internally

### 4. Jinja2 Templating
**Question:** Python uses Jinja2. What's the Rust equivalent?

**Options:**
- A) Tera (most similar to Jinja2)
- B) Handlebars (simpler, less features)
- C) Custom minimal template engine
- D) minijinja (Jinja2 reimplementation in Rust)

**Recommendation:** D (minijinja) - Closest to Jinja2, better compatibility

### 5. Generator Support
**Question:** Python's `@agentic_generator` returns `AsyncGenerator`. How in Rust?

**Options:**
- A) Return `impl Stream<Item = T>`
- B) Use `async_stream` crate for async generators
- C) Manual `Stream` implementation

**Recommendation:** A + B - `async_stream::stream!` macro provides generator syntax

### 6. Error Handling in Macros
**Question:** How do we handle errors in generated code?

**Options:**
- A) All functions return `Result<T, Error>`
- B) Use `?` operator throughout
- C) Provide `try_*` variants for fallible operations

**Recommendation:** A + B - Idiomatic Rust, clear error propagation

### 7. OpenTelemetry Integration
**Question:** How deep should OTEL integration go?

**Options:**
- A) Basic spans around LLM calls
- B) Detailed spans for parsing, tool execution, etc.
- C) Custom metrics for tokens, cost
- D) All of the above

**Recommendation:** D - Comprehensive observability is key selling point

### 8. Python Interop
**Question:** Should we provide Python bindings?

**Options:**
- A) Yes, via pyo3 (post v1)
- B) No, full rewrite only
- C) Provide REST API wrapper instead

**Recommendation:** A (post v1) - Enables gradual migration, but defer to v2

---

## Conclusion

This specification provides a comprehensive roadmap for rewriting reson in Rust. The architecture preserves all functionality while leveraging Rust's strengths:

- **Type safety**: Compile-time guarantees prevent runtime errors
- **Performance**: Zero-copy parsing, efficient async runtime
- **Safety**: No panics, exhaustive error handling
- **Concurrency**: Fearless async without GIL limitations

The 17-week timeline is aggressive but achievable with focused effort. Key risks:

1. **Provider API complexity** - Each provider has quirks, will need extensive testing
2. **Macro complexity** - Proc macros can be finicky, need good error messages
3. **Type system gaps** - Some Python dynamism won't translate directly

Mitigation strategies are built into the phased approach, with early validation of risky components.

**Next Steps:**
1. Review and approve this spec
2. Initialize Rust project structure
3. Begin Phase 1 implementation
4. Weekly progress reviews against milestones

