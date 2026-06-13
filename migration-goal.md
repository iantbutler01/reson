• Completion Goal

  Finish the Reson/OtherYou VFS+VMD migration to a product-ready state where OtherYou consumes the generic Reson VM/VFS gateway boundary, the GCS-backed VFS path is verified end-
  to-end, local validation is comprehensive but cost-controlled, and the branch pair is ready for your morning verification before any prod deploy.

  Notably the migration is creating a shared higher levelish interface for these in reson as we do intend shortly to have multiple VM providers.

  Prereq: Local Secrets

  Use the gitignored local env overlay for the local tmux path:

  /Users/crow/SoftwareProjects/OtherYou/infra/env/api.env.example.local

  It is sourced into the generated API env by ops/dev/run_local_tmux.sh.

  Required/likely entries:

  ANTHROPIC_API_KEY=...
  KERNEL_API_KEY=...
  GOOGLE_GEMINI_API_KEY=...
  NYM__RUNTIME__GCS__SERVICE_ACCOUNT_KEY_PATH=/path/to/gcs-key.json
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/gcs-key.json

  Do not edit .run/local-tmux/api.env directly; it is generated.

  Acceptance Criteria

  1. Branches are coherent and clean:
      - reson: wip/reson-vfs-vmd-migration
      - OtherYou: ian/vmd-vfs-reson-migration
      - both clean, committed, rebased, and OtherYou compiles against sibling Reson.

  2. Fix current compile blocker:
      - Resolve the TokenUsage.cache_write_input_tokens mismatch.
      - cargo check --manifest-path api/Cargo.toml passes.

  3. Reson gates pass:
      - cargo test -p reson-sandbox --features vfs-server vfs::
      - rustup run 1.90-aarch64-apple-darwin cargo test -p vmd fuse::
      - shared mount/facade tests pass.
      - Reson VFS/FUSE code remains product-neutral: no OtherYou/Nym semantics in generic Reson modules.

  4. OtherYou adapter gates pass:
      - internal VFS route tests pass.
      - computer mount planning tests pass.
      - legacy /internal/nymfs/{nym_id} alias still works.
      - new /internal/reson/vfs/{owner_id} route works.
      - new VM mounts use /v1/internal/reson/vfs/{owner_id}.

  5. Local runtime validation uses the correct ops path:
      - start via ops/dev/run_local_tmux.sh up --no-attach or attached equivalent.
      - remote corvidae VMD/control plane is ready.
      - local API health passes.
      - desktop UI loads.
      - generated API env uses infra/env/api.env.example.local.
      - local heartbeat schedules are paused during validation and restored on shutdown.

  6. GCS-backed VFS path is proven clean:
      - use NYM_LOCAL_DEV_NYMFS_BACKEND=gcs.
      - VM mounts include gcs-vfs-fuse.
      - write through /nym/vm/mounts/task.
      - stat/read through /v1/internal/reson/vfs/{owner_id}.
      - verify persisted content is actually in canonical VFS/GCS-backed storage, not only VM-local state.
      - range read works.
      - mkdir/write/rename/delete behavior works with product write leases.
      - resource-key mismatch is rejected before mutation.
      - current-task alias resolves correctly before and after pause/resume.

  7. VM/product scenarios are covered without wasteful LLM spend:
      - Prefer computer_repro, direct API calls, and MCP/internal tool calls over open-ended LLM tasks.
      - Use at most one short Nym/chat/agent run only if needed to verify product integration.
      - Avoid long heartbeats, browsing loops, or repeated model-driven validation.
      - Do not leave any test heartbeat enabled.

  8. Runtime lifecycle is verified:
      - cold connect creates or attaches surface.
      - warm attach is fast and does not rebuild state unnecessarily.
      - exec works.
      - pause/resume works.
      - VM-side file written before pause is still visible after resume.
      - VMD logs show no FUSE/gateway/auth/lease errors during the scenario.

  9. Cleanup is mandatory:
      - stop local tmux session when done.
      - restore any paused schedules if the script did not.
      - disable/delete test heartbeats.
      - clean temporary test Nyms/tasks/files only where safe.
      - leave both repos with only intentional code changes.
      - commit final work.

  10. Prod readiness, but no prod deploy before your verification:

  - capture exact commands for deploy/rollback.
  - identify required Reson + OtherYou image/build steps.
  - record known smoke checks for after deploy.
  - do not mutate prod until you explicitly approve after morning verification.

  Definition Of Done

  The migration is complete when the branch pair compiles, Reson and OtherYou tests pass, the GCS-backed generic VFS route is proven through a real corvidae VM lifecycle, cleanup
  is complete, and the repo is committed with a concise validation note you can review before approving prod deploy. Additionally you must go through local end to end tests using the browser MCP to test the flows both on the local version but also the version with GCS enabled + the vm deployment method as specified in our OVH ops scripts in other you but through the local corvidae path.
  You may need to wait on GCS live path depending on whether gcloud auth has expired if this is the case use the local VFS path first for testing and stop there until I can get involved in the morning.

  Ensure we have a good higherish level interface that multiple VM/sandbox providers can be put behind they will have roughly the same semantics are our own solution.

  Refer to the exisiting RFC for this migration to make sure you are on the up and up with what the direction was.
