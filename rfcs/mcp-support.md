# RFC: MCP Support for Reson

> **Note**: This RFC should be saved to `reson/rfcs/mcp-support.md` after approval

## Summary

Add MCP (Model Context Protocol) support to reson via a new `reson-mcp` crate that:
1. Uses `rmcp` for core MCP protocol (client + server)
2. Implements **MCP Apps extension (SEP-1865)** in Rust - first Rust implementation
3. Integrates with reson's Runtime for seamless tool bridging

## Motivation

MCP is becoming the standard for connecting AI agents to external tools and data. The MCP Apps extension (SEP-1865, Status: Stable 2026-01-26) enables interactive HTML UIs embedded in conversations. Supporting MCP allows reson agents to:
- **As Client**: Consume any MCP server (filesystem, databases, GitHub, etc.)
- **As Server**: Expose reson agents to MCP hosts (Claude Desktop, ChatGPT, VS Code)
- **With Apps**: Return interactive UIs from tool calls

## Completed Work

### Phase 1: Client (DONE)
- `McpClient` with HTTP, WebSocket, stdio transports
- WebSocket transport implemented ourselves (rmcp doesn't provide it)
- 6 integration tests, all passing

### Phase 3: Server (DONE)
- `McpServer` with `ServerTransport` enum and single `serve()` method
- Dynamic tool registration via `McpServerBuilder::with_tool()`
- 5 integration tests, all passing

### Phase 4: MCP Apps Extension (IN PROGRESS)

**Spec reference**: https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx
**Status**: Stable (2026-01-26), supported by Claude, Claude Desktop, VS Code Insiders, Goose, Postman

#### Completed:
- `apps/types.rs` - Core types: UiToolMeta, UiResourceCsp, UiPermissions, UiResourceMeta, DisplayMode, Visibility, UiResource, ui_uri(). All match stable spec.
- `apps/bridge.rs` - Bridge protocol message types and method constants.
- `apps/resource.rs` - UiResourceRegistry with conversion to rmcp Resource/ReadResourceResult types. Uses Arc for cheap cloning when McpServer clones per-connection.
- `apps/mod.rs` - Exports updated, compiles clean with `--features apps`

#### Architecture decision (per spec):
The spec requires tools to be pre-associated with UI resources via `_meta.ui.resourceUri`, and hosts fetch HTML via a separate `resources/read` call. This means the server must store the HTML content to serve it when requested. The `UiResourceRegistry` exists as a parallel to rmcp's `ToolRouter` - it's composed into `McpServer` and handles `resources/list` + `resources/read`. The spec explicitly designed it this way for caching, prefetching, security review, and auditability.

#### Spec audit gaps to fix in bridge.rs:

**`UiInitializeParams`:**
- Field should be `app_capabilities` (not `capabilities`) per spec's `McpUiAppCapabilities`
- Spec typed fields: `experimental: Option<Value>`, `tools: Option<Value>`, `available_display_modes: Option<Vec<DisplayMode>>`

**`HostContext` missing fields:**
- `user_agent: Option<String>` - host app identifier
- `platform: Option<String>` - "web" | "desktop" | "mobile"
- `available_display_modes: Option<Vec<DisplayMode>>`
- `device_capabilities: Option<Value>` - {touch, hover}
- `safe_area_insets: Option<Value>` - {top, right, bottom, left}

**`ToolCancelledParams`:**
- Spec says `reason: string` (required), we have `Option<String>` — keep as Option, spec text says "which can optionally be specified"

**Missing result type:**
- `RequestDisplayModeResult { mode: DisplayMode }` - response to ui/request-display-mode includes actual mode set

**Missing sandbox proxy methods:**
- `SANDBOX_PROXY_READY: &str = "ui/notifications/sandbox-proxy-ready"`
- `SANDBOX_RESOURCE_READY: &str = "ui/notifications/sandbox-resource-ready"`

**Missing sandbox proxy params type:**
- `SandboxResourceReadyParams { html: String, sandbox: Option<String>, csp: Option<UiResourceCsp>, permissions: Option<UiPermissions> }`

#### Remaining Phase 4 work:
1. Fix bridge.rs gaps listed above
2. Integrate apps with McpServer:
   - Add `with_ui_tool()` to `McpServerBuilder` - registers tool with `_meta.ui` AND stores HTML resource
   - Implement `ServerHandler::list_resources()` - delegates to registry
   - Implement `ServerHandler::read_resource()` - delegates to registry
   - Enable `resources` capability in `ServerCapabilities` when apps feature active
3. Write tests
4. Build and verify

### Phase 2: Reson Integration (DEFERRED)
After all MCP phases complete.

### Phase 5: Python Bindings (DEFERRED)

## Key Files

```
reson-mcp/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── error.rs
│   ├── transport/
│   │   ├── mod.rs
│   │   └── websocket.rs      # Our WebSocket transport (rmcp doesn't provide)
│   ├── client/
│   │   ├── mod.rs
│   │   ├── connection.rs
│   │   └── tools.rs
│   ├── server/
│   │   ├── mod.rs
│   │   └── handler.rs        # McpServer, McpServerBuilder, ServerTransport enum
│   └── apps/
│       ├── mod.rs
│       ├── types.rs           # SEP-1865 extension types
│       ├── resource.rs        # UiResourceRegistry + rmcp conversion
│       └── bridge.rs          # postMessage JSON-RPC bridge types
└── tests/
    ├── client_tests.rs        # 6 tests
    └── server_tests.rs        # 5 tests
```

## Decisions Made

- **Structure**: Sibling directory to reson-rust (like reson-py)
- **Priority**: Client-first implementation
- **License**: Apache-2.0 (consistent with reson)
- **Dependency**: Re-export rmcp types, don't vendor
- **WebSocket**: Implement ourselves using `tokio-tungstenite` (rmcp doesn't provide it)
- **Server API**: Single `serve(ServerTransport)` method, not individual serve_stdio/serve_http/etc
- **Transport module**: Shared at `src/transport/`, not nested under client or server
- **Apps registry**: Separate `UiResourceRegistry` type paralleling rmcp's `ToolRouter`, composed into McpServer. Per spec, resources are pre-declared and served via `resources/read` (not inline in tool results).
- **Spec version**: Using stable 2026-01-26, NOT the draft
- **User preference**: Feature → tests → red/green → next feature workflow. No dead code. No copying types in tests — import real ones.
