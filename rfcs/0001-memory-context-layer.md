# RFC 0001: Context Layer (Turn + Context)

## Summary
Define the context layer around two primitives: **Turn** and **Context**. A Context is an ordered history of Turns plus a configurable pipeline for projecting a context view into the model. Storage backends (Postgres, Redis, vector DBs) are implementation details, not primitives. Retrieval, compression, and codecs are pipeline operations, not first-class types.

## Background
We already have:
- A storage layer with multiple backends.
- Production usage of Redis as an operational bus (OpenBracket).
- Working patterns for retrieval, clustering, and summarization (OpenBracket context slices/topics, FAISS).

We want to formalize the smallest set of primitives while keeping storage flexible and pipelines composable.

## Goals
- Keep primitives minimal and composable.
- Preserve temporal ordering as the core invariant.
- Let users choose raw or compressed Turn representations.
- Provide pipeline hooks for retrieval, clustering, compression, and codecs.
- Keep storage backends out of the primitive model.

## Non-goals
- Managing external infra (vector DBs, Redis, Postgres).
- A prescriptive "mode" system.
- Enforcing a single retrieval or compression strategy.

## Primitives
### 1) Turn (atomic record)
The atomic unit of agent interaction. A Turn captures the full cycle of what happened:
- Inputs (user, tool results, other agent messages).
- Model output.
- Tool calls and tool results.
- Any metadata the user chooses to store.

A Turn is stored as a single record (raw or compressed at user choice).

### 2) Context (managed history)
A Context is an ordered collection of Turns within a boundary (session, thread, agent, etc.). Context is the ordered Turn history. Operational state (coordination, caching, intermediate results) is a separate concern handled by storage backends like Redis, not part of the Context primitive.

Context is a managed object with:
- Turn history (ordered).
- A projection pipeline that builds the model view.

## Invariants
Minimal set:
1) **Temporality is preserved.** Turns keep a stable ordering.
2) **Turn atomicity.** A Turn is complete or it does not exist.
3) **Context boundary.** Each Turn belongs to exactly one Context boundary.

## Operations (verbs, not primitives)
These are pipeline operations applied to a Context's Turn history:
- **Retrieve**: select Turns or artifacts (recency, similarity, clustering).
- **Compress**: summarize or transform selected items.
- **Encode**: apply schema/codec compression for domain-specific compaction.
- **Assemble**: build a context view for the next model call.

All of these are configurable and optional.

## Pipeline (Context projection)
Context projection builds the model-visible view from Turns. Example config:

```json
{
  "retrievers": ["recency", "vector", "cluster"],
  "compressors": ["dynamic_summary", "taxonomy_labeler"],
  "codecs": ["prefs_v1", "tasks_v1"],
  "budgets": { "total_tokens": 8000, "per_stage": { "retrieve": 4000, "summary": 2000 } },
  "policies": { "evidence_pointers": true, "drop_rules": ["low_similarity"] }
}
```

## Storage Backends (implementation detail)
Storage is intentionally below the primitive layer:
- Redis, Postgres, and vector DBs are interchangeable backends.
- Backends can be used for operational state, durable history, or both.
- The primitive model does not assume any backend is authoritative.

## Compatibility Notes
- Existing Storage trait stores Turns (Context history).
- Existing MemoryKVStore remains a backend choice.
- OpenBracket patterns map to pipeline operations (clustering, summaries, vector retrieval).

## Evaluation
Metrics to validate:
- Task success rate in multi-turn agent workflows.
- Token usage (raw vs compressed).
- Context precision/recall (facts surfaced vs needed).
- Latency impact of retrieval + compression.

## Open Questions
- Which codecs should be provided first (prefs, tasks, tool results)?
- What metadata is minimally required on a Turn (timestamps, ids)?
- How should pipeline config be exposed in Python without API churn?
