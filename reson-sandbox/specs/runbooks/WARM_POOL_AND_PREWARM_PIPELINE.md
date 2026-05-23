<!-- @dive-file: Runbook for architecture-aware warm pools and prewarmed image pipeline semantics. -->
<!-- @dive-rel: Enforced by scripts/verify_warm_pool_pipeline.sh for checklist item 12.35. -->
<!-- @dive-rel: Implemented by warm-pool prewarm paths in crates/reson-sandbox/src/lib.rs. -->
# Warm Pool And Prewarm Pipeline Runbook

Status: Locked  
Scope: Section 12 item `12.35` architecture-aware warm pools + prewarmed image pipeline.

## 1) Objective

Ensure session startup avoids request-path image prep by prewarming architecture-scoped image profiles ahead of first use.

## 2) Pipeline Contract

On sandbox startup when `prewarm_on_start=true`:

1. Resolve warm-pool profiles from explicit config (`warm_pool_profiles`) or default image/host architecture fallback.
2. For each healthy endpoint and profile, call VMD `PreDownloadVmImage`.
3. Cache ready profile keys as `(endpoint, image, architecture)`.
4. Session creation classifies warm-pool hit vs cold path and emits SLO observations.
5. Cold path triggers async refill for the requested profile.

## 3) Verification

Run:

```bash
./scripts/verify_warm_pool_pipeline.sh --strict
```

The gate executes:

- `warm_pool_prewarms_profiles_by_architecture`

## 4) Pass Criteria

- Gate exits `0`.
- Startup prewarm requests are issued for configured architectures.
- Warm-pool keying includes endpoint + image + architecture.
- Cold path refill hook exists.
