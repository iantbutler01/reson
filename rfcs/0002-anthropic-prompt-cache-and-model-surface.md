# RFC 0002: Anthropic Prompt-Cache Correctness and Modern Model Surface

## Summary
Three related changes to the Anthropic provider path, all driven by empirical findings from a production agent loop (OtherYou runtime, measured via per-turn usage telemetry):

1. **Automatic prompt caching stamps an explicit `cache_control` marker on the second-to-last message** instead of emitting the top-level `cache_control` request field.
2. **Tool definitions serialize in registration order** instead of alphabetical order.
3. **Adaptive thinking and `output_config` support** (`effort`, structured-output `format`), replacing the deprecated top-level `output_format` and closing the gap where non-numeric `@reasoning=` values were silently dropped.

## Background
Anthropic prompt caching is a strict prefix match over `tools → system → messages`. Cache writes cost 1.25×–2× input price; reads cost 0.1×. Three defects made chevalier-driven agent loops pay the write premium on nearly every call:

- **The top-level `cache_control` field produced unreadable history entries.** In a multi-turn loop, every turn re-wrote its byte-identical message history (`cache_write` ≈ 27–43K tokens/turn) while `cache_read` stayed pinned at the system prefix. Per-segment byte fingerprinting located the cause: `Runtime::run` appends `prompt`/`default_prompt` as the final user message of every request. That message is not part of durable history — its bytes vanish from the next request — so any cache entry keyed through the end of the request can never be matched again.
- **Alphabetical tool serialization resorts the prefix root.** Tool definitions are the first bytes of the cacheable prefix. Sorting by name means a tool added mid-session (agentic tool re-selection) splices into the middle of the tools array and invalidates every cached prefix from the insertion point onward. Registration order makes additions append-only.
- **The model surface predates adaptive thinking.** The only thinking control was `@reasoning=<number>` → `thinking: {enabled, budget_tokens}`, which returns 400 on Opus 4.7+/Fable. Non-numeric values failed `parse::<u32>()` and were **silently ignored** — no thinking, no error. There was no way to express `thinking: {"type": "adaptive"}` or `output_config.effort`, and structured outputs used the deprecated top-level `output_format` parameter.

## Goals
- Multi-turn loops re-read their full durable prefix every turn; per-turn cache writes shrink to the new delta.
- Mid-session tool registration extends the cached prefix instead of invalidating it.
- Deterministic serialization is preserved (same inputs → same bytes), including across processes.
- Adaptive thinking, effort levels, and `output_config.format` are expressible through the existing model-string syntax with loud failures for invalid values.

## Non-goals
- Model-aware validation (e.g. rejecting `budget_tokens` for Opus 4.7+ client-side). Callers opt into combinations per model string; the API's own 400s are the contract.
- Porting these behaviors to the `google_anthropic` and `bedrock` providers (they have separate builders and no automatic-caching support today).
- Cache-TTL selection policy — callers choose markers (`Ephemeral` vs `Ephemeral1h`) via `AnthropicProviderConfig` as before.

## Design

### 1) Automatic caching → explicit second-to-last-message marker
`AnthropicProviderConfig.automatic_prompt_caching` now stamps `cache_control` on the last content block of the **second-to-last** message (falling back to the only message). This places the breakpoint at the end of the *shared* prefix — the documented placement — and lets the volatile trailing message (the appended run prompt) ride after the breakpoint, uncached at ~100 tokens/turn.

Verified effect on a live loop (per-turn, turns ≥ 2):

| | cache_read | cache_write |
|---|---|---|
| before | ~15.7K (system prefix only, flat) | 27–35K every turn |
| after | 41K+, growing each turn | 113–534 (new delta only) |

The marker is skipped when the target block already carries `cache_control`, so explicit per-message markers (`ChatMessage::with_cache_marker`) still take precedence.

### 2) Registration-order tool serialization
`Runtime` records tool registration order (`tool_order: Vec<String>`, maintained by all three registration paths and `unregister_tool`). `generate_tool_schemas` emits schemas in that order. Names present in the tool map but missing from the order record (defensive; should not occur) are appended in sorted order so output stays deterministic.

Callers that want cross-session prefix sharing for identical tool sets should canonicalize (sort) their *initial* registration batch and append subsequent additions — registration order is the contract, ordering policy stays with the caller.

**Empirical limit:** the API treats *any* change to the tools array — including a pure
append at the end — as a full cache invalidation (tools/system/messages all rebuild;
observed live: a 23→28 append-only reselection produced `cache_read = 0` and a full-prompt
re-write on that turn, with full reads resuming the very next turn). So registration order
does NOT make mid-session additions cache-cheap; its value is determinism and cross-session
canonicalization, plus keeping the diff append-shaped for any future provider behavior that
can exploit it. The real mitigation for mid-session additions is making them rare
(selection pinning at the caller).

### 3) Adaptive thinking, effort, `output_config`
`@reasoning=` for `anthropic:` models now maps:

| value | request |
|---|---|
| `<number>` | `thinking: {enabled, budget_tokens}` + legacy max_tokens/temperature adjustments (pre-4.7 models) |
| `adaptive` | `thinking: {"type": "adaptive"}` — no sampling params, no max_tokens adjustment |
| `low\|medium\|high\|xhigh\|max` | adaptive thinking + `output_config: {"effort": <level>}` |
| anything else | hard error (previously: silently dropped) |

`AnthropicClient` gains `with_adaptive_thinking()` and `with_effort(level)`. Structured outputs move from top-level `output_format` to `output_config: {"format": {...}}`; `effort` and `format` share one `output_config` object. Sampling parameters are suppressed whenever either thinking mode is active.

## Compatibility
- The wire change in (1) alters marker placement only; prefix bytes are unchanged, so existing cache entries are unaffected (markers are not part of the hashed prefix).
- (2) changes tool-array byte order for any caller with non-sorted registration order — a one-time cache rebuild on upgrade, then strictly better behavior.
- (3) is additive; numeric `@reasoning=` behavior is unchanged. The only breaking change is that invalid `@reasoning=` strings now error instead of silently doing nothing — intentional.

## Validation
- `test_build_request_with_automatic_prompt_caching` / `test_automatic_prompt_caching_skips_trailing_ephemeral_prompt` pin marker placement.
- `test_generate_tool_schemas_uses_registration_order` / `..._appends_unordered_names_deterministically` pin ordering.
- `test_build_request_adaptive_thinking` / `..._effort_lands_in_output_config` / `..._output_schema_uses_output_config_format` pin the model-surface mapping.
- End-to-end verified against the Anthropic API via the OtherYou runtime with per-turn `cache_read`/`cache_write` telemetry.
