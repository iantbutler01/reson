# chevalier-sandbox

Sandbox runtime for Chevalier. It provides one facade over local/self-hosted `vmd` and managed OpenComputer sandboxes.

## What Is Here

| Path | Purpose |
| --- | --- |
| `crates/sandbox/` | Public Rust facade: connect, session, attach, exec, file operations, fork, mounts |
| `vmd/` | Host daemon for local or self-hosted sandbox execution |
| `portproxy/` | Guest/host port proxy support |
| `proto/` | gRPC contracts for `vmd` and portproxy |
| `scripts/` | Verification, integration, DR, security, and rollout drills |

## Providers

The facade is stable across providers:

- `chevalier` / `local` / `vmd`: Chevalier's own host daemon
- `opencomputer`: OpenComputer backend through their API

Default crate features include `local`. Remote-only consumers can avoid host/process dependencies:

```toml
chevalier-sandbox = { version = "0.1", default-features = false, features = ["client"] }
```

Use `distributed-control` for etcd/NATS-backed routing and HA control paths. Use `vfs-server` when serving VFS gateway routes from the sandbox crate.

## VFS Mounts

Shared mounts carry Chevalier VFS metadata through the same session API. With OpenComputer, command mounts require the guest template to include `chevalier-vfs-fuse`; see [Sandbox and VFS](../docs/sandbox-vfs.md).

## Verification

Common gates:

```bash
make verify
make verify-strict
make verify-e2e
make verify-strict-real PROFILE=local-dev
```

The Makefile also exposes targeted gates for fork, API facade, storage, admission, DR, security, SLO, ownership fence, partition handling, and control-gateway failover.

Real gates need real runtime dependencies. Keep mock/unit checks for fast iteration, but do not use them as proof of product readiness.
