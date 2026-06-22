# RFC 0003: Codex Subscription Provider

Status: Implemented 2026-06-19 for first-party Rust provider, TypeScript `codexSubscription` config, SSE/WebSocket transports, and OpenBracket consumption. Live provider smoke requires a real ChatGPT/Codex subscription token.

## Summary

Add a first-party Chevalier provider for ChatGPT/Codex subscription-backed coding models. The provider lives in Chevalier's Rust core, is exposed through the Chevalier TypeScript package, and is consumed by OpenBracket as a normal `Runtime` model integration.

The provider id is:

```text
openai-codex-responses
```

Model strings use:

```text
openai-codex-responses:<codex-model-id>
```

This provider is intentionally distinct from the normal OpenAI API provider. It represents a user subscription transport, not OpenAI API billing.

## Background

OpenBracket already uses Chevalier as the runtime boundary for:

- streaming model events,
- tool schema registration,
- host-dispatched tool calls,
- MCP-backed tools,
- raw provider responses,
- TypeScript consumption through the `chevalier` package.

Ian wants OpenBracket to support Codex subscription models in the same model-selection path as local Qwen and other hosted models. The right ownership boundary is Chevalier, not an OpenBracket-only transport. The Pi `openai-codex-responses` implementation provides a useful reference for request shape, SSE/WebSocket fallback, session caching, usage-limit parsing, and browser/Vite-safe imports.

## Goals

- Implement Codex subscription execution as a first-party Chevalier Rust provider.
- Surface typed provider config through Chevalier TypeScript bindings.
- Preserve the current `Runtime` contract used by OpenBracket.
- Support streaming text, reasoning, tool-call partials, completed tool calls, usage events, and raw response diagnostics.
- Support WebSocket transport with safe SSE fallback before first emitted event.
- Keep subscription credentials out of model strings, logs, raw responses, stream events, and frontend-visible data.
- Let OpenBracket consume the provider without forking its agent loop.

## Non-goals

- Replacing the normal OpenAI API provider.
- Making Codex subscription models the default in OpenBracket.
- Implementing OpenBracket credential vault UI in Chevalier.
- Enforcing OpenBracket command approval, guardian, artifact, planning, or review policies inside Chevalier.
- Persisting provider continuation state across process restarts in the first implementation.

## Design

### Rust Provider

Add provider modules:

```text
rust/src/providers/openai_codex_responses.rs
rust/src/providers/openai_codex_responses_streaming.rs
```

Wire them through:

```text
rust/src/providers/mod.rs
rust/src/runtime/mod.rs
rust/src/runtime/inference.rs
```

Exact runtime factory paths can follow the existing provider-routing code, but the end result must be that:

```rust
Runtime::with_config(
    Some("openai-codex-responses:gpt-5.1-codex".to_string()),
    None,
)
```

can construct a runtime once a Codex subscription provider config is supplied.

### Provider Config

Add a typed provider config instead of overloading the generic API key:

```rust
pub struct CodexSubscriptionProviderConfig {
    pub token: String,
    pub account_id: Option<String>,
    pub base_url: Option<String>,
    pub transport: Option<CodexSubscriptionTransport>,
    pub sse_header_timeout: Option<std::time::Duration>,
    pub websocket_connect_timeout: Option<std::time::Duration>,
}

pub enum CodexSubscriptionTransport {
    Auto,
    WebSocket,
    Sse,
}
```

Extend `ProviderConfig`:

```rust
pub enum ProviderConfig {
    Anthropic(AnthropicProviderConfig),
    CodexSubscription(CodexSubscriptionProviderConfig),
}
```

If multiple provider configs need to coexist, split `ProviderConfig` into a struct with optional per-provider fields rather than a single enum. The TypeScript surface should not require callers to discard Anthropic config when they set Codex config.

### TypeScript Surface

Expose the config through `ts/src/runtime.rs`, `ts/index.ts`, and `ts/index.d.ts`.

Preferred TypeScript shape:

```ts
const rt = new Runtime({
  model: "openai-codex-responses:gpt-5.1-codex",
  providerConfig: {
    codexSubscription: {
      token,
      accountId,
      baseUrl,
      transport: "auto",
      sseHeaderTimeoutMs: 20_000,
      websocketConnectTimeoutMs: 15_000,
    },
  },
});
```

If constructor options become too broad, `setProviderConfig` may be used instead:

```ts
const rt = new Runtime({ model: "openai-codex-responses:gpt-5.1-codex" });
await rt.setProviderConfig({
  codexSubscription: { token, transport: "auto" },
});
```

Do not encode token, account id, base URL, or transport flags into model-string suffixes.

### Request Construction

The provider builds a Responses-style request:

- `model`: model id after the `openai-codex-responses:` prefix.
- `store`: false unless a later RFC explicitly changes continuation semantics.
- `stream`: true.
- `instructions`: Chevalier system prompt.
- `input`: converted Chevalier conversation messages.
- `tools`: converted Chevalier tool schemas.
- `tool_choice`: `"auto"` when tools are present.
- `parallel_tool_calls`: true.
- `text.verbosity`: default low unless the caller supplies a typed option later.
- `include`: reasoning encrypted content only if needed by provider behavior.
- `prompt_cache_key`: bounded stable session key when available.

Tool conversion should follow the existing OpenAI Responses provider where possible. The Codex provider should share conversion helpers instead of duplicating schema logic if the schemas are compatible.

### Stream Mapping

Map provider events to Chevalier stream events:

- output text deltas -> content events,
- reasoning summary/content -> reasoning events,
- function-call argument deltas -> tool partial events,
- completed function calls -> tool-call events,
- usage deltas/final usage -> usage events,
- response completed/done -> stream completion,
- provider failed/incomplete/error -> typed Chevalier error.

The mapping must preserve tool-call ids so downstream loops can correlate partials, approvals, tool results, and task events.

### Transport

Support:

- `auto`: try WebSocket, then SSE fallback only if no event has been emitted,
- `websocket`: WebSocket only,
- `sse`: SSE only.

Rules:

- If WebSocket fails before the first provider event, fallback to SSE.
- If WebSocket fails after any provider event, fail the run instead of replaying and risking duplicate tool calls.
- Bound WebSocket connect timeout.
- Bound SSE response-header timeout.
- Respect per-run timeout and abort signals.
- Keep WebSocket continuation cache in memory, keyed by runtime/session id.
- Drop continuation state when request bodies no longer match safely.
- Clear cached sessions on runtime cleanup.

### Error Handling

Add typed Chevalier errors for:

- missing subscription credential,
- invalid token/account id,
- usage limit reached,
- provider rate limit,
- provider auth failure,
- SSE invalid JSON,
- WebSocket close before completion,
- transport timeout,
- provider `response.failed`.

Every error path must redact:

- authorization headers,
- token,
- account id,
- raw credential-bearing request metadata.

### OpenBracket Consumption Contract

OpenBracket will:

- read `OPENBRACKET_CODEX_SUBSCRIPTION_TOKEN` and related env vars,
- pass typed provider config into Chevalier,
- expose configured Codex models through `GET /models`,
- keep local Qwen as the default,
- treat subscription usage as token usage with zero API dollar cost unless a later billing model says otherwise,
- keep approvals, guardians, MCP tool execution, artifacts, planning, and review behavior outside the provider.

Chevalier must not depend on OpenBracket packages or OpenBracket-specific task models.

## Compatibility

- Existing providers and model strings continue to work.
- Normal OpenAI API models remain on the existing `openai:` and `openai-responses` paths.
- TypeScript callers without Codex config see a clear missing-credential error when selecting `openai-codex-responses:*`.
- No frontend or Vite bundle should import Node-only provider code. The transport lives behind the native Chevalier runtime.

## Validation

Rust tests:

- Provider selection from `openai-codex-responses:<model>`.
- Request conversion for system prompt, durable history, tool calls, and tool results.
- Tool schema conversion.
- Stream mapping for content, reasoning, function-call partials, completed tool calls, usage, and completion.
- WebSocket fallback only before the first emitted event.
- Timeout behavior for SSE headers and WebSocket connect.
- Error parsing and redaction.

TypeScript tests:

- `Runtime` accepts typed Codex provider config.
- `setProviderConfig` or constructor config passes through to native bindings.
- Missing credential produces a typed `ChevalierError`.
- Stream events preserve the same shape as existing providers.

Live smoke:

- Gated by a local subscription credential env var.
- One no-tool streaming prompt.
- One prompt with a schema-only host-dispatched tool.
- One prompt with a handler-backed tool executed through Chevalier.

## Open Questions

- What exact Codex model ids should be exposed in the default OpenBracket catalog?
- Should provider config live on `RuntimeOptions`, `ProviderConfigInput`, or both?
- Should Chevalier expose an explicit `closeProviderSessions()` method for long-running hosts?
- Should subscription usage limits include structured reset timestamps when the provider returns them?
