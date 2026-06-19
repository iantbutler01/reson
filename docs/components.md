# Component Map

This repo is intentionally split. The top-level package names are not interchangeable.

## Agent Runtime

`rust/` is the source of truth for the agent runtime:

- provider clients for Anthropic, OpenAI, Google Gemini, OpenRouter, Bedrock, and Vertex/Google Anthropic paths
- typed tool schemas through `#[derive(Tool)]`
- agent ergonomics through `#[agentic]`
- streaming events, reasoning segments, tool calls, tool results, usage, retries, and cost accounting
- optional features for MCP, VFS, and sandbox integration

`ts/` and `py/` expose the parts of that runtime that are useful from those languages. They are bindings, not parallel implementations.

## MCP

`mcp/` is a Rust MCP library. It supports:

- client connections over stdio, HTTP, and WebSocket
- server tool registration
- optional MCP Apps UI resources

The TypeScript package exposes MCP client/server helpers through its native binding.

## Sandbox

`sandbox/` owns the sandbox abstraction:

- `crates/sandbox`: public Rust facade
- `vmd`: local/hosted sandbox daemon
- `portproxy`: guest/host port bridge
- OpenComputer adapter behind the same facade

The default Rust sandbox feature is `local`, which includes both client and local host pieces. Consumers that only talk to a remote provider should depend on `default-features = false` and enable `client`.

`ts-sandbox/` is a separate Node package for the same reason: most TS users should not install the sandbox client unless they actually need it.

## VFS

`vfs/` owns generic VFS mechanics:

- manifest/index interfaces
- packed object layout and zstd slot reads
- batch metadata and batch read paths
- changed-only writes
- atomic multi-write planning
- subtree prefetch and small-file warming
- gateway client mode
- default Postgres manifest/index implementation

It is not where product ownership, account policy, billing, or audit semantics belong.

## Durable

`durable/` is intentionally small. It defines the vocabulary for durable execution:

```text
Run
Step
State
Effect
Wait
Event
```

Use domain-specific `kind`, key, version, metadata, and payload references on those concepts rather than growing a separate durable concept for every product workflow.

