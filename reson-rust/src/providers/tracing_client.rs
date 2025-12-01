//! TracingInferenceClient - Wraps providers with fallback, cost tracking, and OTEL spans
//!
//! Features:
//! - Automatic fallback switching on RetriesExceeded
//! - Cost tracking with per-model pricing
//! - OpenTelemetry span instrumentation (via `tracing` crate)
//! - Request/response tracing
//! - Switches back to primary after 5 minutes
//!
//! OTEL Integration:
//! This client uses the `tracing` crate for instrumentation. To export spans to
//! OpenTelemetry, configure a `tracing-opentelemetry` layer in your application.
//! Spans include:
//! - `inference.usage.input_tokens`
//! - `inference.usage.output_tokens`
//! - `inference.usage.cached_tokens`

use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{Duration, Instant};

use crate::error::{Error, Result};
use crate::providers::{GenerationConfig, GenerationResponse, InferenceClient, StreamChunk};
use crate::storage::{MemoryKVStore, Store};
use crate::types::{Provider, TokenUsage};
use crate::utils::ConversationMessage;

const MILLION: f64 = 1_000_000.0;
const FALLBACK_DURATION: Duration = Duration::from_secs(5 * 60); // 5 minutes

/// Tracks when fallback was activated
#[derive(Clone)]
struct FallbackState {
    activated_at: Option<Instant>,
}

impl FallbackState {
    fn new() -> Self {
        Self { activated_at: None }
    }

    fn activate(&mut self) {
        self.activated_at = Some(Instant::now());
    }

    fn should_use_fallback(&self) -> bool {
        if let Some(activated) = self.activated_at {
            activated.elapsed() < FALLBACK_DURATION
        } else {
            false
        }
    }

    fn try_reset(&mut self) -> bool {
        if let Some(activated) = self.activated_at {
            if activated.elapsed() >= FALLBACK_DURATION {
                self.activated_at = None;
                return true;
            }
        }
        false
    }
}

/// TracingInferenceClient wraps a primary and optional fallback client
///
/// Automatically switches to fallback on RetriesExceeded, tracks costs,
/// and traces requests/responses for debugging.
pub struct TracingInferenceClient {
    primary_client: Box<dyn InferenceClient>,
    fallback_client: Option<Box<dyn InferenceClient>>,
    fallback_state: Arc<RwLock<FallbackState>>,
    store: Option<Arc<MemoryKVStore>>,
}

impl TracingInferenceClient {
    /// Create a new tracing client with primary client only
    pub fn new(primary: Box<dyn InferenceClient>) -> Self {
        Self {
            primary_client: primary,
            fallback_client: None,
            fallback_state: Arc::new(RwLock::new(FallbackState::new())),
            store: None,
        }
    }

    /// Create with both primary and fallback clients
    pub fn with_fallback(
        primary: Box<dyn InferenceClient>,
        fallback: Box<dyn InferenceClient>,
    ) -> Self {
        Self {
            primary_client: primary,
            fallback_client: Some(fallback),
            fallback_state: Arc::new(RwLock::new(FallbackState::new())),
            store: None,
        }
    }

    /// Attach storage for cost tracking
    pub fn with_store(mut self, store: Arc<MemoryKVStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Get the active client (primary or fallback)
    async fn get_active_client(&self) -> &dyn InferenceClient {
        let mut state = self.fallback_state.write().await;

        // Try to switch back to primary
        if state.try_reset() {
            log::info!("Switching back to primary client after timeout");
            return self.primary_client.as_ref();
        }

        // Check if we should use fallback
        if state.should_use_fallback() {
            if let Some(ref fallback) = self.fallback_client {
                return fallback.as_ref();
            }
        }

        self.primary_client.as_ref()
    }

    /// Calculate cost based on model and token usage
    fn calculate_cost(&self, model: &str, usage: &TokenUsage) -> f64 {
        let mut cost = 0.0;

        // Note: TokenUsage.cached_tokens represents cache hits (reads)
        // Python separates cache_read and cache_write tokens

        if model.contains("claude") || model.contains("Anthropic") {
            if model.contains("haiku") {
                // Haiku pricing
                cost += (usage.input_tokens as f64) * (80.0 / MILLION);
                cost += (usage.output_tokens as f64) * (400.0 / MILLION);
                cost += (usage.cached_tokens as f64) * (8.0 / MILLION); // cache read
            } else if model.contains("opus") {
                // Opus pricing
                cost += (usage.input_tokens as f64) * (1500.0 / MILLION);
                cost += (usage.output_tokens as f64) * (7500.0 / MILLION);
                cost += (usage.cached_tokens as f64) * (150.0 / MILLION); // cache read
            } else {
                // Default to Sonnet pricing
                cost += (usage.input_tokens as f64) * (300.0 / MILLION);
                cost += (usage.output_tokens as f64) * (1500.0 / MILLION);
                cost += (usage.cached_tokens as f64) * (30.0 / MILLION); // cache read
            }
        } else if model.contains("gpt-4o") || model.contains("o4-mini") {
            if model.contains("mini") {
                // GPT-4o-mini / o4-mini
                cost += (usage.input_tokens as f64) * (110.0 / MILLION);
                cost += (usage.output_tokens as f64) * (440.0 / MILLION);
                cost += (usage.cached_tokens as f64) * (27.5 / MILLION); // cache read
            } else {
                // GPT-4o
                cost += (usage.input_tokens as f64) * (500.0 / MILLION);
                cost += (usage.output_tokens as f64) * (1500.0 / MILLION);
                cost += (usage.cached_tokens as f64) * (125.0 / MILLION); // cache read
            }
        } else if model.contains("o3") {
            // o3 pricing
            cost += (usage.input_tokens as f64) * (1000.0 / MILLION);
            cost += (usage.output_tokens as f64) * (4000.0 / MILLION);
            cost += (usage.cached_tokens as f64) * (250.0 / MILLION); // cache read
        } else {
            log::warn!("No cost information for model: {}", model);
        }

        cost
    }

    /// Track cost to storage (in cents)
    async fn track_cost(&self, model: &str, usage: &TokenUsage) -> Result<()> {
        if let Some(ref store) = self.store {
            let cost_dollars = self.calculate_cost(model, usage);
            let cost_cents = (cost_dollars * 100.0).ceil() as u64;

            // Get current usage
            let current: u64 = store.get("credits_used").await?.unwrap_or(0);

            // Add new cost
            store.set("credits_used", &(current + cost_cents)).await?;

            log::debug!(
                "Cost tracking: {} cents for {} tokens (model: {})",
                cost_cents,
                usage.input_tokens + usage.output_tokens,
                model
            );
        }
        Ok(())
    }

    /// Trace request to storage if configured
    async fn trace_request(&self, _messages: &[ConversationMessage], _id: u64) -> Result<()> {
        // TODO: Implement trace_output equivalent
        // Would write to RESON_TRACE directory or S3 bucket
        Ok(())
    }

    /// Trace response to storage if configured
    async fn trace_response(&self, _response: &str, _id: u64) -> Result<()> {
        // TODO: Implement trace_output equivalent
        Ok(())
    }
}

#[async_trait]
impl InferenceClient for TracingInferenceClient {
    #[tracing::instrument(
        name = "TracingInferenceClient.get_generation",
        skip(self, messages, config),
        fields(
            provider,
            model = %config.model,
            inference.usage.input_tokens,
            inference.usage.output_tokens,
            inference.usage.cached_tokens
        )
    )]
    async fn get_generation(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<GenerationResponse> {
        let client = self.get_active_client().await;
        let provider = client.provider();

        // Record provider on span
        tracing::Span::current().record("provider", format!("{:?}", provider).as_str());

        // Try primary/current client
        let result = client.get_generation(messages, config).await;

        // Handle fallback on retries exceeded
        if let Err(Error::RetriesExceeded) = result {
            if self.fallback_client.is_some() {
                log::info!("Switching to fallback client after retries exceeded");
                let mut state = self.fallback_state.write().await;
                state.activate();
                drop(state);

                // Retry with fallback
                let fallback = self.fallback_client.as_ref().unwrap();
                let response = fallback.get_generation(messages, config).await?;

                // Record token usage on span
                let span = tracing::Span::current();
                span.record("inference.usage.input_tokens", response.usage.input_tokens);
                span.record("inference.usage.output_tokens", response.usage.output_tokens);
                span.record("inference.usage.cached_tokens", response.usage.cached_tokens);

                // Track cost and trace
                let fallback_provider = fallback.provider();
                let model = format!("{:?}", fallback_provider);
                self.track_cost(&model, &response.usage).await?;

                return Ok(response);
            } else {
                return Err(Error::RetriesExceeded);
            }
        }

        let response = result?;

        // Record token usage on span
        let span = tracing::Span::current();
        span.record("inference.usage.input_tokens", response.usage.input_tokens);
        span.record("inference.usage.output_tokens", response.usage.output_tokens);
        span.record("inference.usage.cached_tokens", response.usage.cached_tokens);

        // Track cost for successful call
        let model = format!("{:?}", provider);
        self.track_cost(&model, &response.usage).await?;

        Ok(response)
    }

    #[tracing::instrument(
        name = "TracingInferenceClient.connect_and_listen",
        skip(self, messages, config),
        fields(
            provider,
            model = %config.model
        )
    )]
    async fn connect_and_listen(
        &self,
        messages: &[ConversationMessage],
        config: &GenerationConfig,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let client = self.get_active_client().await;

        // Record provider on span
        tracing::Span::current().record("provider", format!("{:?}", client.provider()).as_str());

        // Try primary/current client
        let result = client.connect_and_listen(messages, config).await;

        // Handle fallback on retries exceeded
        if let Err(Error::RetriesExceeded) = result {
            if self.fallback_client.is_some() {
                log::info!("Switching to fallback client for streaming after retries exceeded");
                let mut state = self.fallback_state.write().await;
                state.activate();
                drop(state);

                // Retry with fallback
                let fallback = self.fallback_client.as_ref().unwrap();
                return fallback.connect_and_listen(messages, config).await;
            } else {
                return Err(Error::RetriesExceeded);
            }
        }

        result
    }

    fn provider(&self) -> Provider {
        // Return primary provider (fallback is transparent)
        self.primary_client.provider()
    }

    fn supports_native_tools(&self) -> bool {
        self.primary_client.supports_native_tools()
    }

    fn set_trace_callback(&mut self, _callback: crate::providers::TraceCallback) {
        // TracingClient handles its own tracing
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::MemoryStore;

    struct MockClient {
        should_fail: bool,
        provider: Provider,
    }

    #[async_trait]
    impl InferenceClient for MockClient {
        async fn get_generation(
            &self,
            _messages: &[ConversationMessage],
            _config: &GenerationConfig,
        ) -> Result<GenerationResponse> {
            if self.should_fail {
                Err(Error::RetriesExceeded)
            } else {
                Ok(GenerationResponse {
                    content: "test".to_string(),
                    reasoning: None,
                    tool_calls: Vec::new(),
                    reasoning_segments: Vec::new(),
                    usage: TokenUsage {
                        input_tokens: 100,
                        output_tokens: 50,
                        cached_tokens: 0,
                    },
                    raw: None,
                })
            }
        }

        async fn connect_and_listen(
            &self,
            _messages: &[ConversationMessage],
            _config: &GenerationConfig,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
            unimplemented!()
        }

        fn provider(&self) -> Provider {
            self.provider
        }

        fn supports_native_tools(&self) -> bool {
            true
        }

        fn set_trace_callback(&mut self, _callback: crate::providers::TraceCallback) {}
    }

    #[tokio::test]
    async fn test_primary_success() {
        let primary = Box::new(MockClient {
            should_fail: false,
            provider: Provider::Anthropic,
        });

        let client = TracingInferenceClient::new(primary);
        let config = GenerationConfig::default();
        let result = client.get_generation(&[], &config).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_fallback_on_retries_exceeded() {
        let primary = Box::new(MockClient {
            should_fail: true,
            provider: Provider::Anthropic,
        });
        let fallback = Box::new(MockClient {
            should_fail: false,
            provider: Provider::OpenAI,
        });

        let client = TracingInferenceClient::with_fallback(primary, fallback);
        let config = GenerationConfig::default();
        let result = client.get_generation(&[], &config).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_cost_tracking() {
        let primary = Box::new(MockClient {
            should_fail: false,
            provider: Provider::Anthropic,
        });

        let store = Arc::new(MemoryKVStore::new());
        let client = TracingInferenceClient::new(primary).with_store(store.clone());

        let config = GenerationConfig::default();
        let _ = client.get_generation(&[], &config).await.unwrap();

        // Check cost was tracked
        let credits: Option<u64> = store.get("credits_used").await.unwrap();
        assert!(credits.is_some(), "No credits tracked");
        let credits_value = credits.unwrap();
        assert!(credits_value > 0, "Credits tracked but zero: {}", credits_value);
    }
}
