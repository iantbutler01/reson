<!-- @dive-file: Module-level metadata for supporting runtime files across facade, vmd, and portproxy layers. -->
<!-- @dive-rel: Captures helper/runtime boundaries not already expanded in .dive/modules/ha_contract.md. -->
<!-- @dive-rel: Tracks where utility modules participate in lifecycle, continuity, and operational behavior. -->
# Runtime Support Module
Maps auxiliary runtime modules that underpin VM lifecycle execution, image handling, partition policy, and guest-side process services.

## Files
- `crates/reson-sandbox/build.rs` - Facade protobuf generation wiring for vmd/portproxy contracts.
- `crates/reson-sandbox/src/slo.rs` - SLO budget model and evaluator used by rollout/gate policy checks.
- `vmd/src/assets/mod.rs` - Asset module root exporting bootstrap payload helpers.
- `vmd/src/image.rs` - Base image resolution and download pipeline for VM creation.
- `vmd/src/partition.rs` - Partition fail-closed monitor and mutation gating policy.
- `vmd/src/state/mod.rs` - State module root exporting manager/metadata/runtime/types.
- `vmd/src/state/metadata.rs` - Persistent metadata load/save helpers for `vm.json`.
- `vmd/src/state/runtime.rs` - In-memory runtime process state for active VMs.
- `vmd/src/state/types.rs` - Core VM state/source/snapshot type definitions.
- `vmd/src/virt/mod.rs` - QEMU runtime helpers, monitor control, and conversion support.
- `vmd/src/bin/vmdctl.rs` - CLI operations client used by users and verifier harnesses.
- `portproxy/build.rs` - Guest-side protobuf generation for portproxy services.
- `portproxy/src/child_tracker.rs` - Child PID tracking for cleanup/signal handling.
- `portproxy/src/cli.rs` - Command-line mode and argument validation for portproxy.
- `portproxy/src/main.rs` - Portproxy service entrypoint and service wiring.
- `portproxy/src/port_forward.rs` - TCP forwarding server/client implementation.

## Relationships
- `crates/reson-sandbox/build.rs` -> `crates/reson-sandbox/src/lib.rs`: generated protobuf types feed facade API transport calls.
- `crates/reson-sandbox/src/slo.rs` -> `scripts/verify_slo_profile.sh`: SLO evaluator results drive gate and rollout policy checks.
- `vmd/src/image.rs` -> `vmd/src/state/manager.rs`: image download/selection supports VM create/start flows.
- `vmd/src/partition.rs` -> `vmd/src/app.rs`: partition monitor feeds fail-closed policy decisions during control-plane outages.
- `vmd/src/state/mod.rs` -> `vmd/src/app.rs`: exports lifecycle/state APIs consumed by daemon RPC handlers.
- `vmd/src/state/metadata.rs` -> `vmd/src/state/manager.rs`: metadata persistence anchors VM lineage and snapshot durability.
- `vmd/src/state/runtime.rs` -> `vmd/src/state/manager.rs`: transient runtime fields track active qemu process ownership.
- `vmd/src/state/types.rs` -> `vmd/src/state/metadata.rs`: typed structs back serialized metadata and runtime/state conversions.
- `vmd/src/virt/mod.rs` -> `vmd/src/state/manager.rs`: monitor/process helpers execute snapshot, start, stop, and conversion operations.
- `vmd/src/bin/vmdctl.rs` -> `scripts/verify_*.sh`: CLI is used as black-box control surface in integration gates.
- `portproxy/build.rs` -> `portproxy/src/main.rs`: generated proto modules power gRPC service endpoints.
- `portproxy/src/child_tracker.rs` -> `portproxy/src/main.rs`: PID set supports signal forwarding and zombie cleanup loops.
- `portproxy/src/cli.rs` -> `portproxy/src/main.rs`: validated mode/arguments select server vs forwarding behavior.
- `portproxy/src/main.rs` -> `portproxy/src/port_forward.rs`: server/client runtime delegates TCP forwarding operations.
