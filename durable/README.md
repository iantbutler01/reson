# chevalier-durable

Durable execution primitives for Chevalier agent runtimes.

This crate intentionally keeps the durable vocabulary small:

```text
Run
Step
State
Effect
Wait
Event
```

Everything else is represented as a `kind`, key, version, payload reference, or metadata field on one of those six concepts.

