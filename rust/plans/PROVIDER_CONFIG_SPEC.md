# Provider Config Spec

## Problem

Chevalier is accumulating provider-specific request knobs in ad hoc places:

- `GenerationConfig.prompt_cache_retention` for OpenAI
- `GenerationConfig.provider_config` for Anthropic automatic caching
- model-string parsing for some provider-only behavior
- runtime-level setters for other provider-only behavior

This is heading toward interface sprawl.

The library needs one typed seam for provider-specific request shaping without:

- widening `RunParams` every time a provider adds a knob
- stuffing unrelated fields directly onto `GenerationConfig`
- hiding policy inside inference defaults

## Goals

- Keep the public API narrow.
- Make provider-specific behavior explicit and typed.
- Preserve the existing builder style.
- Keep policy ownership with the caller.
- Let providers ignore config that does not apply to them.

## Non-Goals

- This does not redesign model-string parsing in general.
- This does not introduce automatic provider policy defaults.
- This does not force all provider behavior behind one trait immediately.
- This does not change message-level cache markers. Those stay on messages.

## Design

### 1. Typed Provider Config Enum

Provider-specific request options live under a single typed enum:

```rust
pub enum ProviderConfig {
    Anthropic(AnthropicConfig),
    OpenAI(OpenAIConfig),
    Google(GoogleConfig),
    Bedrock(BedrockConfig),
    OpenRouter(OpenRouterConfig),
}
```

Each config struct should contain only request-shaping options that are genuinely provider-specific.

Examples:

```rust
pub struct AnthropicConfig {
    pub automatic_prompt_caching: Option<CacheMarker>,
}

pub struct OpenAIConfig {
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    pub prompt_cache_key: Option<String>,
}
```

The enum is the stable public seam. Per-provider structs can grow without turning the top-level API into a grab bag.

### 2. Builder-Style Attachment

The API should remain builder-oriented:

```rust
let config = GenerationConfig::new("claude-sonnet")
    .with_provider_config(ProviderConfig::Anthropic(
        AnthropicConfig {
            automatic_prompt_caching: Some(CacheMarker::Ephemeral),
        }
    ));
```

At the runtime layer:

```rust
runtime.set_provider_config(Some(
    ProviderConfig::Anthropic(AnthropicConfig {
        automatic_prompt_caching: Some(CacheMarker::Ephemeral),
    })
)).await;
```

This preserves the current ergonomic shape:

- `GenerationConfig` for direct provider calls
- `Runtime` setter for repeated runs using one runtime instance

### 3. Message Metadata Stays on Messages

Cache markers for prompt blocks remain message-level metadata:

- `history: Vec<ConversationMessage>`
- `system_messages: Vec<ChatMessage>`

Provider config must not duplicate message metadata.

Specifically:

- no prompt-side cache marker field on `RunParams`
- no automatic “mark the first user message” library behavior
- no hidden cache placement in inference orchestration

The caller decides where breakpoints go. Chevalier serializes them.

### 4. Provider Serialization Boundary

`runtime/inference.rs` may pass `ProviderConfig` through, but it should not contain provider policy.

The provider modules own serialization:

- Anthropic emits top-level request `cache_control`
- Anthropic-family providers serialize message/system `cache_control`
- OpenAI emits `prompt_cache_retention`
- future Google explicit cache support emits `cached_content`

The runtime and inference layers should forward typed config, not interpret it beyond basic routing.

## Migration Plan

### Phase 1

Normalize the current mixed surface:

- Keep `with_provider_config(...)`
- Keep `Runtime::set_provider_config(...)`
- Remove one-off request-shaping fields that belong inside provider structs

Immediate target:

- move `prompt_cache_retention` out of top-level `GenerationConfig`
- put it under `ProviderConfig::OpenAI(...)`

### Phase 2

Reduce model-string special cases for request shaping.

Model strings can still be convenience syntax, but they should hydrate typed provider config instead of adding more loose top-level fields.

Example:

- `openai:gpt-5@cache=24h`

should parse into:

- provider = `openai`
- provider config = `ProviderConfig::OpenAI(OpenAIConfig { prompt_cache_retention: Some(H24), ... })`

### Phase 3

Add more provider structs only when there is real request-shaping value.

Do not precreate large empty config types for every provider just for symmetry.

## Constraints

- Backwards compatibility matters. Do not break the common `Runtime` path unless there is a strong reason.
- The interface should not require callers to understand every provider to use one provider.
- Providers must ignore irrelevant config safely rather than failing on enum growth in unrelated code paths.

## Acceptance Criteria

- There is one obvious typed place for provider-specific request knobs.
- `RunParams` does not grow provider-specific fields.
- Cache placement remains caller-controlled through message metadata.
- OpenAI caching knobs and Anthropic automatic caching both fit the same top-level shape.
- Adding the next provider-specific knob does not require inventing another top-level field on `GenerationConfig` or `RunParams`.

## Immediate Follow-Up

1. Move OpenAI cache retention under `ProviderConfig::OpenAI`.
2. Keep Anthropic automatic caching under `ProviderConfig::Anthropic`.
3. Remove stale top-level request-shaping fields once the new enum path is complete.
4. Add live integration coverage for each provider-specific config path that is publicly exposed.
