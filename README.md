# Chevalier

Chevalier is the runtime toolkit behind agentic products that need more than a chat completion call. It is Rust-first, with TypeScript and Python bindings where they make sense.

It currently covers four product-grade surfaces:

- model/provider runtime and tool calling
- MCP clients, servers, and app resources
- sandboxed computers backed by local `vmd` or OpenComputer
- optimized VFS storage with local, object-store, packed, Postgres-indexed, and gateway modes

## Packages

| Path | Package | What lives there |
| --- | --- | --- |
| [`rust/`](rust/README.md) | `chevalier` | Rust LLM runtime, provider clients, tool schemas, streaming, MCP/VFS/sandbox feature hooks |
| [`ts/`](ts/README.md) | `chevalier` | Node/Bun bindings for runtime, MCP, and VFS client APIs |
| [`ts-sandbox/`](ts-sandbox/README.md) | `chevalier-sandbox` | Separate Node sandbox binding so normal TS users do not pull in the heavier sandbox client |
| [`py/`](py/README.md) | `chevalier` | Python bindings via PyO3/maturin |
| [`mcp/`](mcp/README.md) | `chevalier-mcp` | Rust MCP client/server/apps library |
| [`sandbox/`](sandbox/README.md) | `chevalier-sandbox`, `vmd`, `portproxy` | Sandbox facade, local host daemon, OpenComputer adapter, control-plane plumbing |
| [`vfs/`](vfs/README.md) | `chevalier-vfs` | Optimized VFS primitives: manifests, packed storage, batch reads/writes, gateway client/server contracts |
| [`durable/`](durable/README.md) | `chevalier-durable` | Small durable execution vocabulary: run, step, state, effect, wait, event |

More detail:

- [Component map](docs/components.md)
- [Sandbox and VFS integration](docs/sandbox-vfs.md)
- [Contributing and verification expectations](CONTRIBUTING.md)

## How The Pieces Fit

Chevalier keeps product code above the infrastructure boundary:

```text
Product runtime
  -> chevalier / TS / Python runtime APIs
  -> MCP, sandbox, VFS, durable primitives
  -> provider APIs, vmd/OpenComputer, object stores, databases
```

The important rule is that product policy stays in the product. Chevalier provides reusable runtime machinery: tool loops, provider clients, sandbox sessions, VFS storage interfaces, command mounts, packed file storage, and gateway clients. Ownership, authorization, billing, audit, and domain-specific path rules wrap these packages.

## Development

This repo uses Rust 2024 and the local `.tool-versions` `rust stable` toolchain. Run checks from the package you changed:

```bash
# Rust agent runtime
cd rust
cargo test --all-features
cargo clippy --all-features -- -D warnings

# Sandbox facade, vmd, and portproxy
cd sandbox
make verify

# VFS
cd vfs
cargo test --all-features
cargo clippy --all-features -- -D warnings

# TypeScript bindings
cd ts
npm test
npm run build

# TypeScript sandbox binding
cd ts-sandbox
npm run build

# Python bindings
cd py
maturin develop
```

Some gates require live providers or real sandbox machinery. Do not treat a narrow unit test as proof that a provider, VFS gateway, or sandbox backend works end-to-end.

## License

Apache-2.0
