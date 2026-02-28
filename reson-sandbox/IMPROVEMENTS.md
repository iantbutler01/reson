# Sandbox Improvements

Issues encountered while integrating reson-sandbox with OtherYou.

## `inspect_image_platforms` is fragile

**File:** `vmd/src/virt/mod.rs:302`

**Problem:** Uses `docker manifest inspect` to get platform info, which:
- Fails on plain v2 manifests (single-arch images pushed with `docker push` — no manifest list)
- Requires `--insecure` flag for HTTP registries even when they're in `insecure-registries`
- Needs Docker CLI experimental features enabled in some configurations
- Returns "no such manifest" for images that `docker pull` can resolve fine

**Impact:** Any single-arch image pushed without `docker buildx` fails with "docker manifest inspect did not return platform information" even though the image is perfectly valid and pullable.

**Suggestion:** Fall back gracefully when manifest inspect doesn't return a manifest list:
1. Try `docker manifest inspect` first (current behavior)
2. If it fails or returns a plain v2 manifest, fall back to `docker image inspect` after pulling — the local inspect always has `Architecture` and `Os` fields
3. Or query the registry HTTP API directly: `GET /v2/{name}/manifests/{tag}` with `Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.docker.distribution.manifest.v2+json` — parse the `config.digest` and fetch the config blob which has platform info
4. Could also use `skopeo inspect` which handles all manifest types cleanly

Option 3 (direct registry API) would remove the Docker CLI dependency entirely for this check.
