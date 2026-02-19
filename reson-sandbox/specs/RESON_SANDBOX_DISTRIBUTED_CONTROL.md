# Reson Sandbox Distributed Control (etcd + MQ)

This document defines the distributed backing layer while keeping the public Rust facade unchanged.

## Goals

- Keep `Sandbox` / `Session` API stable for users.
- Support multi-node daemon routing with durable session attachment.
- Use etcd for authoritative control state.
- Use NATS for control-plane event fanout (not stdout/stderr streaming).

## User-Facing Contract

- No API changes:
  - `Sandbox::new(config)`
  - `Sandbox::connect(endpoint, config)`
  - `Sandbox::session(opts)`
  - `Sandbox::attach_session(session_id)`
  - `Sandbox::list_sessions()`
  - `Session::fork/exec/shell/discard`
- Distributed mode is enabled by setting `SandboxConfig.distributed_control`.
- Local/direct mode remains default.

## etcd Keyspace

Prefix default: `/reson-sandbox`

- Node registry:
  - `/reson-sandbox/nodes/<node_id>`
  - Value: JSON with at least `{ "endpoint": "http://node-host:8052" }`
- Session routing:
  - `/reson-sandbox/sessions/<session_id>`
  - Value: JSON:
    - `session_id`
    - `vm_id`
    - `endpoint`
    - optional `node_id`
    - optional `fork_id`

## Node Selection

- Deterministic hash on `session_id` over currently registered nodes.
- Existing session binding in etcd wins.
- If binding is stale, the facade falls back to scanning known nodes.

## Node Self-Registration

- `vmd` can self-register into etcd with lease-backed heartbeats.
- Enable by setting `RESON_SANDBOX_ETCD_ENDPOINTS` (comma-separated list).
- Optional env overrides:
  - `RESON_SANDBOX_ETCD_PREFIX` (default `/reson-sandbox`)
  - `RESON_SANDBOX_NODE_ID` (default hostname/uuid fallback)
  - `RESON_SANDBOX_NODE_ENDPOINT` (default `http://<listen>`)
  - `RESON_SANDBOX_NODE_TTL_SECS` (default `15`)
- Equivalent CLI overrides on `vmd`:
  - `--registry-etcd-endpoints`
  - `--registry-prefix`
  - `--node-id`
  - `--advertise-endpoint`
  - `--registry-ttl-secs`
  - `--disable-node-registry`

## MQ (NATS) Usage

Subject prefix default: `reson.sandbox.control`

Published lifecycle events:

- `reson.sandbox.control.session.bound`
- `reson.sandbox.control.session.discarded`

Payloads are JSON envelopes with `event`, `timestamp_unix_ms`, and `payload`.

## Streaming Boundary

- Interactive process I/O (`exec`/`shell` streams) remains direct gRPC to the resolved node.
- etcd/NATS are control-plane only.
- This keeps ordering and latency properties for runtime streams.
