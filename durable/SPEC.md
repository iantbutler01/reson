# Chevalier Durable Execution Spec

## Purpose

`chevalier-durable` defines a small durable execution substrate for agentic runtimes. It is not a general workflow engine and it does not try to capture arbitrary language continuations. It gives host applications a stable vocabulary for resumable execution where program state, side effects, and waits are explicit.

The durable vocabulary is exactly:

```text
Run
Step
State
Effect
Wait
Event
```

Everything else is represented as `kind`, `version`, key, metadata, or a payload reference on one of those six concepts.

## Concepts

### Run

A `Run` is one durable execution instance. It owns the current status, current step, current state reference, run kind/version, and optimistic state version.

Runs are the unit a host scheduler claims and advances. Scheduler jobs are only wakeups; the run row is the source of truth.

### Step

A `Step` is a registered code unit that advances a run from one state to another. Step dispatch is by `(run_kind, run_version, step_key)` so new tasks can use newly registered steps while old in-flight runs keep executing against the version they were created with.

Existing step behavior must remain compatible for the versions that reference it. Breaking behavior or schema changes require a new run version, a new step key, or an explicit state migration owned by the host.

### State

`State` is serialized resumable program data. The crate only stores a reference and content hash; the host owns the backing store.

For Nym-shaped hosts, this is expected to be VFS/NymFS/GCS-backed payloads with relational control-plane rows pointing at exact refs and hashes.

### Effect

An `Effect` is a named idempotent side effect. A host must enforce stable keys:

```text
same run + same effect key + same input hash => recorded output
same run + same effect key + different input hash => hard error
```

Effects cover external actions such as model calls, tool calls, file writes, spawned runs, VM/browser operations, emails, or any other operation that should not be repeated accidentally after a retry.

### Wait

A `Wait` is a durable suspension point. A run in `Waiting` is alive but not runnable until an event resolves the wait.

Waits cover permission gates, user replies, child run completion, external callbacks, timers, or any host-defined blocking condition.

### Event

An `Event` is external input recorded durably. Events may resolve waits or wake runs. Like state payloads, event payloads are referenced by ref and hash; the host owns storage.

## Host Responsibilities

The crate is intentionally not opinionated about persistence. A production host must provide:

- transactional control-plane storage for runs, steps, states, effects, waits, and events
- artifact storage for state/effect/event payloads
- content-hash verification when loading payload refs
- optimistic concurrency or leases for claiming runnable runs
- idempotency checks for effects
- wait resolution and wakeup enqueueing
- versioned step registration and dispatch

The recommended durability rule is:

```text
write payload to backing store
verify or compute hash
commit relational row that points at ref + hash
advance run pointer in the same transaction
```

If payload write succeeds and the DB transaction fails, the payload is garbage. If the DB transaction succeeds and the payload cannot be read or hash-verified later, that is a durability invariant violation.

## Execution Loop

The host execution loop should be structurally simple:

```text
claim runnable Run
load current State by ref/hash
dispatch Step by run kind/version + step key
Step records Effects, Waits, Events, and States through DurableContext
commit one transition
mark Run runnable, waiting, finished, or failed
```

Pods and workers are disposable. After restart, the host recovers expired `Running` leases and reclaims runs. Because effects are keyed and idempotent, retrying a step returns recorded effect outputs rather than duplicating external work.

## Versioning

New runs bind to the current active `(kind, version)` at creation time. Existing runs remain pinned to their original version.

New deployments may register newer steps and versions. They must not silently change the semantics of old step registrations that old runs still need.

Migration is allowed, but it is a host-owned explicit transition from one state schema/version to another. Automatic migration during resume should be treated as a risky operation and avoided until the host can surface and audit it clearly.

## Non-Goals

`chevalier-durable` does not provide:

- a workflow DSL
- arbitrary Rust continuation capture
- a generic DAG engine
- a scheduler implementation
- database schema or storage adapters
- VM/browser/tool-specific concepts
- anthropomorphic agent phase names

Those belong in host runtimes or higher-level libraries built on top of these six primitives.

