# Reson Sandbox Deep Dive Review

Date: 2026-04-24
Scope: `reson-sandbox` plus the `OtherYou` API and production manifests that integrate with it.
Status: Review artifact with follow-up fix status appended.

Paths are relative to `/Users/crow/SoftwareProjects/reson` unless prefixed with `../OtherYou`.

## Ranking Rubric

- **Security impact**: compromise of sandbox isolation, control-plane tokens, guest execution, egress policy, image provenance, or user credentials.
- **Correctness impact**: lifecycle, HA/failover, idempotency, data integrity, filesystem semantics, or contract mismatch.
- **Speed impact**: latency, throughput, memory pressure, O(n) or N+1 behavior, unbounded growth, or cloud/backend cost.
- **Severity order**: Critical, High, Medium, Low. The table is ranked by overall blast radius first, then likelihood.

## Executive Ranking

| Rank | ID | Severity | Primary Impact | Security | Correctness | Speed | Finding |
| ---: | --- | --- | --- | --- | --- | --- | --- |
| 1 | RS-001 | Critical | Security | Critical | High | Medium | Guest execution and file RPC are exposed on a pod-wide bind without auth. |
| 2 | RS-002 | Critical | Security | Critical | Medium | Low | Envoy egress filtering is vulnerable to DNS rebinding and SSRF-style policy bypass. |
| 3 | RS-003 | Critical | Security | Critical | Medium | Low | Direct high-port UDP egress bypasses domain policy entirely. |
| 4 | RS-004 | High | Security | High | Medium | Low | vmd/API/NymFS bearer tokens move over plaintext internal HTTP. |
| 5 | RS-005 | High | Security | High | High | Low | NATS and etcd are unauthenticated/plaintext in prod. |
| 6 | RS-006 | High | Security | High | Medium | Low | Public ingress forwards user `Authorization` and cookies toward guest services. |
| 7 | RS-007 | High | Security | High | Medium | Low | Public VM proxy can tunnel arbitrary declared guest ports. |
| 8 | RS-008 | High | Security | High | Medium | Low | Explicit empty domain allowlist normalizes to allow-all egress. |
| 9 | RS-009 | High | Security | High | High | Low | Internal NymFS write routes can bypass lease/fence coordination. |
| 10 | RS-010 | High | Security | High | Medium | Low | Root recursive `chown` follows symlinks before dropping privileges. |
| 11 | RS-011 | High | Security | High | High | Low | Per-VM cgroup guardrails can fail open or affect the wrong process set. |
| 12 | RS-012 | High | Security | High | High | Medium | Prebuilt VM base images are trusted without digest/signature verification. |
| 13 | RS-013 | High | Security | High | Medium | Low | Privileged d2vm conversion uses the Docker socket and mutable image tags. |
| 14 | RS-014 | High | Correctness | Medium | High | Low | Network policy persistence and runtime apply are non-atomic. |
| 15 | RS-015 | High | Correctness | Low | High | Low | Distributed streaming exec ignores timeout and detach semantics. |
| 16 | RS-016 | High | Correctness | Medium | High | Medium | Exec stream messages are ACKed before validated delivery and do not validate stream identity. |
| 17 | RS-017 | High | Correctness | Medium | High | Low | Command/stdin replay is not idempotent across redelivery. |
| 18 | RS-018 | High | Correctness | Medium | High | Low | Session-route ownership fences and control-bus ownership fences are separate keyspaces. |
| 19 | RS-019 | High | Correctness | Medium | High | Medium | Reconciliation writes incomplete legacy session routes. |
| 20 | RS-020 | High | Correctness | Low | High | Medium | Admission and capacity checks race with VM/session creation. |
| 21 | RS-021 | High | Correctness | Low | High | Low | Snapshot restore and fork can corrupt or strand VM lifecycle state on partial failure. |
| 22 | RS-022 | High | Correctness | Low | High | Medium | GCS NymFS hashes packed files by hashing the whole pack object. |
| 23 | RS-023 | High | Correctness | Low | High | Medium | NymFS FUSE writes are last-writer-wins whole-file flushes. |
| 24 | RS-024 | Medium/High | Correctness | Low | High | Low | NymFS FUSE ignores `O_TRUNC` and truncate semantics. |
| 25 | RS-025 | Medium/High | Correctness | Low | High | Medium | Paused VMs cannot be stopped/deleted through the normal lifecycle path. |
| 26 | RS-026 | High | Correctness | Medium | High | Medium | Prod NATS/etcd are single-replica despite HA contracts. |
| 27 | RS-027 | High | Correctness | Low | High | Low | Partition fail-closed detection is weak and can be disabled by missing endpoints. |
| 28 | RS-028 | High | Speed | Medium | Medium | High | VM resource requests have no upper bounds before QEMU launch. |
| 29 | RS-029 | Medium | Security | Medium | Medium | Low | Protected ingress launch URLs issue query tokens that vmd ingress does not accept. |
| 30 | RS-030 | Medium | Speed | Medium | Medium | High | Public ingress buffers whole requests/responses and scans all VMs per request. |
| 31 | RS-031 | Medium | Speed | Medium | Medium | High | `exec.run` buffers stdout/stderr fully in memory. |
| 32 | RS-032 | Medium | Speed | Medium | Medium | High | Portproxy file APIs are unbounded full-payload operations. |
| 33 | RS-033 | Medium | Speed | Low | High | High | FUSE writes buffer and flush entire files. |
| 34 | RS-034 | Medium | Correctness | Low | High | Medium | NymFS FUSE writes likely fail above Axum's default 2 MiB body limit. |
| 35 | RS-035 | Medium | Speed | Low | Medium | High | NymFS directory listing hashes each file by reading object bytes. |
| 36 | RS-036 | Medium | Speed | Low | Medium | High | GCS range reads for packed files fetch/decompress whole slots. |
| 37 | RS-037 | Medium | Speed | Low | Medium | High | Placement/admission does full-prefix scans over session routes. |
| 38 | RS-038 | Medium | Speed | Low | Medium | High | Node and port lease heartbeats churn etcd clients and leases. |
| 39 | RS-039 | Medium | Speed | Low | Medium | High | Control-plane keyspaces grow without TTL or compaction boundaries. |
| 40 | RS-040 | Medium | Speed | Low | Medium | High | FUSE directory and inode state grow without eviction. |
| 41 | RS-041 | Medium | Speed | Medium | Medium | High | Base-image downloads have no per-image lock and race on `.part` files. |
| 42 | RS-042 | Medium | Speed | Low | Medium | High | Envoy access-log accounting is an unbounded hot path. |
| 43 | RS-043 | Medium | Correctness | Medium | Medium | Medium | Portproxy timeouts do not kill process trees. |
| 44 | RS-044 | Medium | Correctness | Low | Medium | Low | `cargo check --workspace` fails on stable due unstable Rust syntax and no pinned toolchain. |
| 45 | RS-045 | Low/Medium | Security | Medium | Low | Low | d2vm bakes a root password of `root` into converted VM images. |

## Detailed Findings

### RS-001: Guest execution and file RPC are exposed without auth

- **Impact**: Security Critical, Correctness High, Speed Medium.
- **Evidence**: Prod sets `RESON_SANDBOX_PORT_FORWARD_BIND_ADDRESS=0.0.0.0` in `../OtherYou/ops/gke/nym-prod/vmd-statefulset.yaml:137`. QEMU host forwarding uses that bind address in `reson-sandbox/vmd/src/state/manager.rs:2332`. The guest `portproxy` gRPC server binds its address without auth in `reson-sandbox/portproxy/src/main.rs:90`, and exposes shell/file services in `reson-sandbox/portproxy/src/services.rs`.
- **Risk**: Pod-network reachability becomes guest command execution and guest file access.
- **Fix direction**: Bind forwarded guest RPC to loopback/private sidecar only, require mTLS or per-session tokens, and put shell/file APIs behind explicit authorization.

### RS-002: Envoy egress filtering is DNS-rebinding/SSRF bypassable

- **Impact**: Security Critical, Correctness Medium, Speed Low.
- **Evidence**: Allow/block decisions are authority-regex based in `reson-sandbox/vmd/src/network/mod.rs:611` and `reson-sandbox/vmd/src/network/mod.rs:765`. Envoy dynamic forward proxy resolves hosts via configured DNS in `reson-sandbox/vmd/src/network/mod.rs:553`. Private IP regex blocks are applied to hostname authority strings in `reson-sandbox/vmd/src/network/mod.rs:827`, not to resolved upstream IPs.
- **Risk**: A permitted hostname can resolve to private or cluster IPs after policy checks, bypassing the sandbox egress boundary.
- **Fix direction**: Enforce resolved-IP denies at connect time, pin DNS answers per decision, and block private/link-local/cluster CIDRs below Envoy as well as in route policy.

### RS-003: Direct high-port UDP egress bypasses domain policy

- **Impact**: Security Critical, Correctness Medium, Speed Low.
- **Evidence**: Firewall rules allow private-range UDP high ports before dropping private CIDRs in `reson-sandbox/vmd/src/network/firewall.rs:159`, allow WebRTC UDP ports in `reson-sandbox/vmd/src/network/firewall.rs:236`, and finally allow UDP ephemeral range in `reson-sandbox/vmd/src/network/firewall.rs:273`.
- **Risk**: A guest can use UDP 32768:65535 directly without Envoy, so domain allowlists and blocklists are irrelevant for that traffic.
- **Fix direction**: Scope UDP bypasses to explicit STUN/TURN destinations or authenticated relay flows, and deny arbitrary guest UDP except policy-approved endpoints.

### RS-004: Internal bearer tokens move over plaintext HTTP

- **Impact**: Security High, Correctness Medium, Speed Low.
- **Evidence**: vmd TLS only enables when TLS config exists in `reson-sandbox/vmd/src/app.rs:143` and `reson-sandbox/vmd/src/config.rs:1060`. Prod starts and advertises `http://...:8052` in `../OtherYou/ops/gke/nym-prod/vmd-statefulset.yaml:121` and `../OtherYou/ops/gke/nym-prod/vmd-statefulset.yaml:135`. The API connects over HTTP in `../OtherYou/ops/gke/nym-api/configmap.yaml:15` while attaching bearer metadata in `../OtherYou/api/src/services/computer/reson.rs:170`. Internal NymFS auth also uses bearer headers in `reson-sandbox/vmd/src/fuse/client.rs:56` and `../OtherYou/api/src/middleware/internal_auth.rs:23`.
- **Risk**: Internal network visibility is enough to steal and replay vmd or NymFS service tokens.
- **Fix direction**: Require TLS/mTLS for vmd and internal NymFS in prod, rotate leaked static tokens, and consider per-pod scoped credentials.

### RS-005: NATS and etcd are unauthenticated/plaintext

- **Impact**: Security High, Correctness High, Speed Low.
- **Evidence**: Prod config uses `http://` etcd and `nats://` NATS in `../OtherYou/ops/gke/nym-prod/vmd-configmap.yaml:12` and `../OtherYou/ops/gke/nym-prod/vmd-configmap.yaml:13`. etcd listens on HTTP in `../OtherYou/ops/gke/nym-prod/etcd.yaml:83`. NATS now requires a token from `reson-vmd-auth/internal-service-token`, but transport is still plaintext and etcd is still unauthenticated/plaintext.
- **Risk**: Any actor on the cluster network can read/write authoritative routing state or publish control commands/events.
- **Fix direction**: Enable etcd client auth/TLS and NATS auth/TLS, restrict NetworkPolicies, and use separate identities for readers, writers, and command consumers.

### RS-006: Public ingress forwards user credentials to guest services

- **Impact**: Security High, Correctness Medium, Speed Low.
- **Evidence**: vmd public ingress authorizes using bearer/cookie headers in `reson-sandbox/vmd/src/public_ingress.rs:156`, then copies all request headers except connection/transfer/content-length to the guest in `reson-sandbox/vmd/src/public_ingress.rs:304`. The safer API ingress path strips `Authorization` and sanitizes cookies in `../OtherYou/api/src/routes/vm_ingress.rs:73`.
- **Risk**: Protected guest apps receive user bearer tokens or cookies and can exfiltrate them.
- **Fix direction**: Strip `Authorization`, `Cookie`, `Proxy-Authorization`, and sensitive auth headers before guest proxying; inject only explicit forwarded identity headers if needed.

### RS-007: Public VM proxy tunnels arbitrary guest ports

- **Impact**: Security High, Correctness Medium, Speed Low.
- **Evidence**: VM startup maps the public proxy port to guest proxy port `13337` in `reson-sandbox/vmd/src/state/manager.rs:1307`. Public ingress writes the requested guest port preface to that proxy in `reson-sandbox/vmd/src/public_ingress.rs:273`. API-generated ingress records carry arbitrary `guest_port` values into VM metadata in `../OtherYou/api/src/services/computer/reson.rs:1375`.
- **Risk**: Any exposed ingress can reach any declared guest port, including unintended admin or local-only guest services.
- **Fix direction**: Enforce per-exposure port allowlists at vmd, validate against the API's intended exposure record, and reject localhost/admin ports by policy.

### RS-008: Empty domain allowlist means allow all

- **Impact**: Security High, Correctness Medium, Speed Low.
- **Evidence**: The DTO allows optional domain allowlists in `../OtherYou/api/src/dto/nym_network_policy_dto.rs:8`. Normalization stores optional/empty policy in `../OtherYou/api/src/services/nym_network_policy_service.rs:257` and `../OtherYou/api/src/repo/nym_network_policy_repo.rs:48`. Envoy fallback returns `[^:]+` when the allowlist is absent or empty in `reson-sandbox/vmd/src/network/mod.rs:765`.
- **Risk**: A user or API path intending "allow none" gets "allow all."
- **Fix direction**: Distinguish `None` from `Some([])`, require explicit open-egress mode, and test the generated Envoy regex for empty allowlists.

### RS-009: Internal NymFS writes can bypass lease/fence coordination

- **Impact**: Security High, Correctness High, Speed Low.
- **Evidence**: Internal write routes make owner token optional in `../OtherYou/api/src/routes/runtime_nymfs_internal.rs:556`, only set `coordination_resource_key` when owner token exists in `../OtherYou/api/src/routes/runtime_nymfs_internal.rs:308`, and `NymFsService` accepts missing fence fields in `../OtherYou/api/src/services/runtime/nym_fs_service.rs:747` and `../OtherYou/api/src/services/runtime/nym_fs_service.rs:767`.
- **Risk**: Any internal-token caller can mutate protected NymFS state outside the coordination lease contract.
- **Fix direction**: Require lock owner and resource key for all mutating internal NymFS routes, except narrowly defined bootstrap/system paths.

### RS-010: Recursive root `chown` follows symlinks

- **Impact**: Security High, Correctness Medium, Speed Low.
- **Evidence**: VM launch recursively `chown`s before setuid in `reson-sandbox/vmd/src/state/manager.rs:2830`. Traversal calls `chown_single_path` on directories and entries in `reson-sandbox/vmd/src/state/manager.rs:2886` and `reson-sandbox/vmd/src/state/manager.rs:2902`; `chown_single_path` uses `libc::chown` in `reson-sandbox/vmd/src/state/manager.rs:2919`.
- **Risk**: A symlink inside a VM-owned tree can cause host files outside the intended tree to be chowned by root.
- **Fix direction**: Use `lchown`, `openat`/`fstatat` with no-follow semantics, and refuse symlink traversal under writable VM directories.

### RS-011: Per-VM cgroup guardrails can fail open

- **Impact**: Security High, Correctness High, Speed Low.
- **Evidence**: cgroup path extraction is best-effort in `reson-sandbox/vmd/src/network/firewall.rs:590`. Block rules use cgroup path matching in `reson-sandbox/vmd/src/network/firewall.rs:377`. Manager logs and continues on registration failure in `reson-sandbox/vmd/src/state/manager.rs:1540`, and the enforcer skips missing pid data in `reson-sandbox/vmd/src/network/mod.rs:899`.
- **Risk**: Per-VM network caps may not apply, or can apply to the wrong process set if cgroup identification is stale.
- **Fix direction**: Make guardrail registration fail closed for protected policies, verify cgroup membership after QEMU start, and expose health for skipped enforcement.

### RS-012: Prebuilt VM base images are trusted without integrity verification

- **Impact**: Security High, Correctness High, Speed Medium.
- **Evidence**: Default prebuilt image registry/reference is configured in `reson-sandbox/vmd/src/image.rs:24` and `reson-sandbox/vmd/src/image.rs:51`. Downloads stream raw bytes into a target file in `reson-sandbox/vmd/src/image.rs:207` and publish with rename in `reson-sandbox/vmd/src/image.rs:251`. Cache trust is based on local existence/size in `reson-sandbox/vmd/src/state/manager.rs:1802`.
- **Risk**: A compromised registry, tag, cache, or partial artifact can become the trusted VM base.
- **Fix direction**: Require digest-pinned references and verify sha256/signatures before publish; record provenance in VM metadata.

### RS-013: Privileged d2vm conversion uses Docker socket and mutable images

- **Impact**: Security High, Correctness Medium, Speed Low.
- **Evidence**: Docker fallback runs d2vm with `--privileged` and `/var/run/docker.sock` mounted in `reson-sandbox/vmd/src/virt/mod.rs:140`. Prod sets `RESON_SANDBOX_D2VM_IMAGE` to `:latest` in `../OtherYou/ops/gke/nym-prod/vmd-configmap.yaml:7`; the vmd app image is tag-pinned but not digest-pinned in `../OtherYou/ops/gke/nym-prod/vmd-statefulset.yaml:93`.
- **Risk**: Mutable or compromised converter images run privileged with host Docker control.
- **Fix direction**: Digest-pin d2vm/vmd, remove Docker socket dependency where possible, and isolate conversion in a dedicated locked-down worker pool.

### RS-014: Network policy update is non-atomic

- **Impact**: Security Medium, Correctness High, Speed Low.
- **Evidence**: The API now snapshots the previous policy, persists the requested policy, pushes runtime state, and rolls back the row if the runtime apply fails and no newer policy update has replaced it. vmd stages policy maps and validates Envoy config before restart, but the API still does not track a generation acknowledged by the VM runtime.
- **Risk**: DB/runtime divergence from a simple failed push is mitigated. A crash after runtime apply but before response/audit, or a missing runtime apply-status generation, can still leave operators without an exact committed generation record.
- **Fix direction**: Stage policy with generation numbers, start and verify new proxy before swapping, and report apply status back to the API.

### RS-015: Streaming exec ignores timeout and detach

- **Impact**: Security Low, Correctness High, Speed Low.
- **Evidence**: The facade sends timeout/detach fields in `reson-sandbox/crates/reson-sandbox/src/lib.rs:1013` and `reson-sandbox/crates/reson-sandbox/src/lib.rs:1038`. vmd explicitly ignores them in `reson-sandbox/vmd/src/control_bus.rs:1055`.
- **Risk**: Client-visible exec contracts are not enforced in distributed mode.
- **Fix direction**: Enforce timeout/detach in the producer process lifecycle and publish terminal events consistently.

### RS-016: Exec stream ACK and validation order is unsafe

- **Impact**: Security Medium, Correctness High, Speed Medium.
- **Evidence**: Subscriber decodes then ACKs before event channel delivery is proven in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1020` and `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1021`; the bounded channel can fail in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1068`. Event decode fills missing stream ids rather than validating against the subscribed stream in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1736`.
- **Risk**: Events can be lost after ACK, and wrong-stream events can be accepted.
- **Fix direction**: Validate stream identity before ACK, ACK only after successful enqueue/delivery, and NAK/term invalid payloads.

### RS-017: Command/stdin replay is not idempotent

- **Impact**: Security Medium, Correctness High, Speed Low.
- **Evidence**: Dedupe can fall back to local-only or continue without short-circuit in `reson-sandbox/vmd/src/control_bus.rs:722`. Stdin input carries sequence metadata but does not enforce monotonic application in `reson-sandbox/vmd/src/control_bus.rs:1774`.
- **Risk**: Redelivered commands or stdin frames can run twice or reorder.
- **Fix direction**: Persist command and stream-input application state keyed by session/stream/sequence and fence terminal operations.

### RS-018: Ownership fences use mismatched keyspaces

- **Impact**: Security Medium, Correctness High, Speed Low.
- **Evidence**: Facade session-route writes use route `expected_fence` in `reson-sandbox/crates/reson-sandbox/src/lib.rs:2115` and `reson-sandbox/crates/reson-sandbox/src/lib.rs:2128`. vmd control-bus commands check a separate `/command-dedupe/ownership-fences` keyspace in `reson-sandbox/vmd/src/control_bus.rs:231`.
- **Risk**: Route ownership and command ownership can diverge, so a command can pass a fence that does not represent the authoritative session route.
- **Fix direction**: Use one authoritative fence source for route binding and command execution, or transactionally mirror them.

### RS-019: Reconciliation writes incomplete legacy routes

- **Impact**: Security Medium, Correctness High, Speed Medium.
- **Evidence**: Normal route writes use sharded route/fence records and delete legacy keys in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:316` and `reson-sandbox/crates/reson-sandbox/src/distributed.rs:357`. Reconciliation writes legacy `/sessions/{session_id}` values in `reson-sandbox/vmd/src/reconcile.rs:216`, while route decoding defaults missing fields in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1421`.
- **Risk**: A reconcile repair can downgrade rich route metadata, weakening placement, storage-profile, and fence expectations.
- **Fix direction**: Reconcile only the canonical sharded schema, include all required route fields, and delete legacy compatibility paths after migration.

### RS-020: Admission and capacity checks race

- **Impact**: Security Low, Correctness High, Speed Medium.
- **Evidence**: Distributed node selection counts current routes before create in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:506`; the session route is bound after VM creation in `reson-sandbox/crates/reson-sandbox/src/lib.rs:2248` and `reson-sandbox/crates/reson-sandbox/src/lib.rs:2267`. Local manager checks capacity before insert in `reson-sandbox/vmd/src/state/manager.rs:394` and inserts later in `reson-sandbox/vmd/src/state/manager.rs:529`.
- **Risk**: Concurrent creates can over-admit sessions or exceed active VM limits.
- **Fix direction**: Reserve capacity transactionally before VM creation and release reservation on failure.

### RS-021: Snapshot restore and fork partial failures can corrupt lifecycle state

- **Impact**: Security Low, Correctness High, Speed Low.
- **Evidence**: Restore reverts disk before checking the RAM file exists in `reson-sandbox/vmd/src/state/manager.rs:1727` and `reson-sandbox/vmd/src/state/manager.rs:1731`. Fork pauses the parent before snapshot and can error before resume/cleanup in `reson-sandbox/vmd/src/state/manager.rs:623` and `reson-sandbox/vmd/src/state/manager.rs:636`.
- **Risk**: A failed restore can leave disk/RAM mismatched; a failed fork can leave the parent paused or force-stopped.
- **Fix direction**: Preflight all artifacts before destructive steps and add compensating resume/rollback guards around fork.

### RS-022: GCS NymFS hashes packed files incorrectly

- **Impact**: Security Low, Correctness High, Speed Medium.
- **Evidence**: Packed metadata stores slot offset/length in `../OtherYou/api/src/services/runtime/gcs_nym_fs_adapter.rs:370`. `hash_file_if_present` hashes `object_state.pack_key` bytes without slicing the slot in `../OtherYou/api/src/services/runtime/gcs_nym_fs_adapter.rs:1528`; the async version does the same in `../OtherYou/api/src/services/runtime/gcs_nym_fs_adapter.rs:1889`.
- **Risk**: File hashes can represent the pack object rather than the file, breaking freshness, cache invalidation, and projection repair.
- **Fix direction**: Hash the decoded slot content or reuse stored per-file content hashes from the manifest/object sync table.

### RS-023: NymFS FUSE writes are last-writer-wins

- **Impact**: Security Low, Correctness High, Speed Medium.
- **Evidence**: FUSE lazily loads whole-file handles, writes into an in-memory buffer, and flushes the whole buffer through `reson-sandbox/vmd/src/fuse/fs.rs`. The client PUT has no server-side previous-hash/version precondition in `reson-sandbox/vmd/src/fuse/client.rs`.
- **Risk**: Concurrent writers could overwrite each other even if they changed disjoint ranges.
- **Fix status**: Mitigated in the FUSE path by recording the content hash observed at open and returning `EBUSY` when the remote file changed before flush. Future range-write UX can still improve merge behavior and large-file write cost.

### RS-024: NymFS FUSE ignores truncation semantics

- **Impact**: Security Low, Correctness High, Speed Low.
- **Evidence**: `open` ignores `_flags` in `reson-sandbox/vmd/src/fuse/fs.rs:555`, lazy loading reads existing content in `reson-sandbox/vmd/src/fuse/fs.rs:363`, writes overlay bytes in `reson-sandbox/vmd/src/fuse/fs.rs:625`, and flush writes the resulting whole buffer in `reson-sandbox/vmd/src/fuse/fs.rs:309`.
- **Risk**: `O_TRUNC` and truncate/write workflows can preserve stale tail bytes.
- **Fix direction**: Handle truncate flags and implement `setattr` size changes explicitly.

### RS-025: Paused VMs cannot be stopped/deleted normally

- **Impact**: Security Low, Correctness High, Speed Medium.
- **Evidence**: `stop_vm` returns metadata immediately for `VmState::Paused` in `reson-sandbox/vmd/src/state/manager.rs:1552`; `delete_vm` rejects paused VMs in `reson-sandbox/vmd/src/state/manager.rs:582`. OtherYou's stop helper no-ops unless state is running in `../OtherYou/api/src/services/computer/reson.rs:1008`.
- **Risk**: Paused VMs can remain allocated and undeletable through normal API cleanup paths.
- **Fix direction**: Treat paused as stoppable, or make delete use force-stop/resume-stop as part of cleanup.

### RS-026: Prod NATS/etcd are single-replica despite HA contracts

- **Impact**: Security Medium, Correctness High, Speed Medium.
- **Evidence**: NATS is `replicas: 1` in `../OtherYou/ops/gke/nym-prod/nats.yaml:52`; etcd is `replicas: 1` in `../OtherYou/ops/gke/nym-prod/etcd.yaml:46`; JetStream stream replicas default to 1 in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1355`.
- **Risk**: A single pod/node/storage failure can remove routing state or command delivery, violating distributed HA expectations.
- **Fix direction**: Run quorum etcd and clustered NATS/JetStream with stream replicas aligned to the failure domain.

### RS-027: Partition fail-closed detection is weak

- **Impact**: Security Low, Correctness High, Speed Low.
- **Evidence**: HA mode now requires node registry/control bus configuration, starts the node registry before the partition monitor, and treats an etcd probe with no visible node-registry key as a partition failure. Defaults remain a 2 second probe and threshold 3 in `reson-sandbox/vmd/src/partition.rs:14`.
- **Risk**: Split-brain and missing-registry visibility are mitigated for HA startup/runtime. The remaining risk is broader quorum/topology design rather than the local gate silently disabling itself.
- **Fix direction**: Require partition monitor in distributed prod, use quorum/lease-aware checks, and gate mutating paths on fresh authoritative state.

### RS-028: VM resource requests lack upper bounds

- **Impact**: Security Medium, Correctness Medium, Speed High.
- **Evidence**: Facade resource options are optional in `reson-sandbox/crates/reson-sandbox/src/lib.rs:250`; vmd clamps only to non-negative values in `reson-sandbox/vmd/src/app.rs:503`; manager applies minimum defaults in `reson-sandbox/vmd/src/state/manager.rs:385`; QEMU uses the resulting values in `reson-sandbox/vmd/src/state/manager.rs:2347`.
- **Risk**: A caller can request excessive CPU, memory, or disk and degrade the host/node.
- **Fix direction**: Enforce per-tenant and per-node maxes at API, facade, and vmd admission layers.

### RS-029: Protected ingress launch URL token is not accepted by vmd ingress

- **Impact**: Security Medium, Correctness Medium, Speed Low.
- **Evidence**: API launch URLs issue `?nym_ingress_token=...` in `../OtherYou/api/src/services/nym_vm_ingress_service.rs:161`. The API middleware path accepts that token in `../OtherYou/api/src/routes/vm_ingress.rs:40`. The vmd public ingress path only accepts bearer/cookie and forwards those to internal auth in `reson-sandbox/vmd/src/public_ingress.rs:156` and `reson-sandbox/vmd/src/public_ingress.rs:220`. The generated Kubernetes ingress points at the vmd service/port in `../OtherYou/api/src/services/nym_vm_ingress_service.rs:412`.
- **Risk**: Protected launch URLs fail when routed to vmd, or auth behavior differs depending on which ingress path handles the request.
- **Fix direction**: Make vmd accept and verify the same `nym_ingress_token`, or route protected ingress through the API middleware that already supports it.

### RS-030: Public ingress buffers whole requests/responses and scans all VMs

- **Impact**: Security Medium, Correctness Medium, Speed High.
- **Evidence**: Route resolution loops through `manager.list()` for every request in `reson-sandbox/vmd/src/public_ingress.rs:174` and `reson-sandbox/vmd/src/state/manager.rs:313`. Request bodies use `to_bytes(body, usize::MAX)` in `reson-sandbox/vmd/src/public_ingress.rs:269`, and responses are collected fully in `reson-sandbox/vmd/src/public_ingress.rs:327`.
- **Risk**: Large uploads/downloads or many VMs can turn public ingress into an O(n) memory hot path.
- **Fix direction**: Maintain a hostname route index and stream request/response bodies with bounded limits.

### RS-031: `exec.run` buffers stdout/stderr fully

- **Impact**: Security Medium, Correctness Medium, Speed High.
- **Evidence**: Control-bus run exec appends stdout/stderr into strings in `reson-sandbox/vmd/src/control_bus.rs:918` through `reson-sandbox/vmd/src/control_bus.rs:938`, then publishes one result payload.
- **Risk**: Large command output can consume daemon memory and delay result delivery.
- **Fix direction**: Stream output frames with size caps and terminal metadata; cap buffered fallback output.

### RS-032: Portproxy file APIs are unbounded

- **Impact**: Security Medium, Correctness Medium, Speed High.
- **Evidence**: `ReadFile` returns one `bytes data` payload in `reson-sandbox/proto/bracket/portproxy/v1/portproxy.proto:99`. Service `read_file` uses `fs::read` in `reson-sandbox/portproxy/src/services.rs:466`, `write_file` accepts the full request body in `reson-sandbox/portproxy/src/services.rs:483`, and `list_directory` accumulates entries in `reson-sandbox/portproxy/src/services.rs:508`.
- **Risk**: Large files or huge directories cause memory spikes, gRPC message failures, or guest-side DoS.
- **Fix direction**: Add range/chunked file APIs, response caps, and paginated directory listing.

### RS-033: FUSE writes buffer and flush whole files

- **Impact**: Security Low, Correctness High, Speed High.
- **Evidence**: Write resizes the in-memory buffer in `reson-sandbox/vmd/src/fuse/fs.rs:625`; flush writes the whole buffer to the remote service in `reson-sandbox/vmd/src/fuse/fs.rs:309`; loading a handle can fetch the whole file in `reson-sandbox/vmd/src/fuse/fs.rs:371`.
- **Risk**: Sparse writes or small edits to large files create large memory and network cost.
- **Fix direction**: Support range writes, chunked flush, and sparse-file semantics.

### RS-034: FUSE writes likely hit Axum's default 2 MiB body limit

- **Impact**: Security Low, Correctness High, Speed Medium.
- **Evidence**: Internal `put_file` takes `Bytes` in `../OtherYou/api/src/routes/runtime_nymfs_internal.rs:279`, and no body-limit override was found under `../OtherYou/api/src`. Local `axum-core` default limit is `2_097_152` in `/Users/crow/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/axum-core-0.5.6/src/ext_traits/request.rs:319`. FUSE flush sends one full PUT body in `reson-sandbox/vmd/src/fuse/fs.rs:309`.
- **Risk**: NymFS FUSE writes larger than 2 MiB can fail even before backend storage is reached.
- **Fix direction**: Set an explicit body limit appropriate for NymFS or move to chunked/ranged upload APIs.

### RS-035: Directory listing hashes each file by reading object bytes

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: FUSE `readdir` calls `list_dir` in `reson-sandbox/vmd/src/fuse/fs.rs:216`. Internal `get_tree` hashes each file in `../OtherYou/api/src/routes/runtime_nymfs_internal.rs:121`; GCS async hashing fetches object bytes in `../OtherYou/api/src/services/runtime/gcs_nym_fs_adapter.rs:1899`.
- **Risk**: Listing a directory with many files becomes N object reads and hashes; packed files can reread the same pack many times.
- **Fix direction**: Return stored content hashes from metadata/projection state and avoid object fetches during listing.

### RS-036: GCS range reads fetch whole packed slots

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: FUSE uses ranged reads for large files in `reson-sandbox/vmd/src/fuse/fs.rs:235`. GCS `read_range_async` fetches the full packed slot and slices locally in `../OtherYou/api/src/services/runtime/gcs_nym_fs_adapter.rs:2097` through `../OtherYou/api/src/services/runtime/gcs_nym_fs_adapter.rs:2126`.
- **Risk**: Random reads of large packed/compressed files can repeatedly download and decompress much more than requested.
- **Fix direction**: Store large files uncompressed or chunked, maintain an index for compressed blocks, or bypass packs for files needing random access.

### RS-037: Placement/admission performs full-prefix scans

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: Node selection calls `session_route_records` in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:506`; that scans `/sessions/` with prefix reads in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1188`; counts are recomputed in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1572`.
- **Risk**: Placement latency and etcd load grow linearly with active sessions.
- **Fix direction**: Maintain per-node/tenant counters transactionally with route writes and use bounded indexes for placement.

### RS-038: Heartbeat and lease paths churn clients/leases

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: Node registry creates/connects and grants a lease each heartbeat in `reson-sandbox/vmd/src/registry.rs:99`. Port lease heartbeat similarly creates clients/leases in `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1095` and `reson-sandbox/crates/reson-sandbox/src/distributed.rs:1856`.
- **Risk**: etcd sees avoidable connection and lease churn under node/session count.
- **Fix direction**: Reuse etcd clients and long-lived leases with keepalive streams.

### RS-039: Control-plane keyspaces grow without TTL

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: etcd dedupe store writes command keys in `reson-sandbox/vmd/src/control_bus.rs:220` without an attached lease. Ownership fence keys live under `/ownership-fences` in `reson-sandbox/vmd/src/control_bus.rs:242`. Reconcile summaries are written under `/reconcile/` in `reson-sandbox/vmd/src/reconcile.rs:245`.
- **Risk**: Dedupe, fence, and reconcile records accumulate indefinitely and slow prefix operations/backups.
- **Fix direction**: Attach TTL leases where safe, compact historical summaries, and define retention per keyspace.

### RS-040: FUSE directory/inode state is unbounded

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: Directory cache inserts without a capacity cap in `reson-sandbox/vmd/src/fuse/cache.rs:96`; inode table ensures new entries without eviction in `reson-sandbox/vmd/src/fuse/fs.rs:85` and `reson-sandbox/vmd/src/fuse/fs.rs:135`.
- **Risk**: Walking large trees or many paths can grow daemon memory until restart.
- **Fix direction**: Bound caches, evict by LRU/TTL, and recycle inode mappings.

### RS-041: Base-image downloads race on shared `.part` files

- **Impact**: Security Medium, Correctness Medium, Speed High.
- **Evidence**: Download temp path is deterministic `{}.part` in `reson-sandbox/vmd/src/image.rs:112`; it resumes and renames without a per-image process lock in `reson-sandbox/vmd/src/image.rs:129` and `reson-sandbox/vmd/src/image.rs:251`. Create/pre-download paths can both invoke download in `reson-sandbox/vmd/src/state/manager.rs:1844` and `reson-sandbox/vmd/src/app.rs:835`.
- **Risk**: Concurrent downloads can corrupt partial state or waste bandwidth.
- **Fix direction**: Add per-reference file locks and verify final digest before publish.

### RS-042: Envoy log accounting is unbounded hot-path work

- **Impact**: Security Low, Correctness Medium, Speed High.
- **Evidence**: Envoy writes access logs in `reson-sandbox/vmd/src/network/mod.rs:700`. The enforcer polls periodically in `reson-sandbox/vmd/src/network/mod.rs:876`. Counters read new log data into a string in `reson-sandbox/vmd/src/network/vm_counters.rs:165` and store counters in an unbounded `by_vm` map in `reson-sandbox/vmd/src/network/vm_counters.rs:62`.
- **Risk**: High traffic or many VM ids can make policy accounting CPU/memory heavy.
- **Fix direction**: Rotate logs aggressively, stream parse bounded chunks, and expire inactive VM counters.

### RS-043: Portproxy timeouts do not kill process trees

- **Impact**: Security Medium, Correctness Medium, Speed Medium.
- **Evidence**: Direct exec starts a child in `reson-sandbox/portproxy/src/services.rs:105`; timeout calls `child.start_kill()` only on the top-level process in `reson-sandbox/portproxy/src/services.rs:207`; daemon commands have no timeout in `reson-sandbox/portproxy/src/daemon.rs:66`; child tracker stores top-level PIDs in `reson-sandbox/portproxy/src/child_tracker.rs:22`.
- **Risk**: Timed-out commands can leave grandchildren running in the guest.
- **Fix direction**: Run commands in a process group/session and kill the group on timeout/daemon teardown.

### RS-044: Workspace check fails on stable Rust

- **Impact**: Security Low, Correctness Medium, Speed Low.
- **Evidence**: Local `cargo check --workspace` in `reson-sandbox` fails with stable rustc 1.86.0 because `if let` chains are used in `reson-sandbox/crates/reson-sandbox/src/lib.rs:2397`; no `rust-toolchain` was present to pin nightly/stable expectations.
- **Risk**: Basic verification is not reproducible for contributors or CI unless the environment happens to match.
- **Fix direction**: Remove unstable syntax or add a pinned toolchain and CI gate that matches it.

### RS-045: d2vm bakes a root password into images

- **Impact**: Security Medium, Correctness Low, Speed Low.
- **Evidence**: Native d2vm args include `--password root` in `reson-sandbox/vmd/src/virt/mod.rs:116`; Docker d2vm args do the same in `reson-sandbox/vmd/src/virt/mod.rs:165`.
- **Risk**: If SSH/console login becomes reachable in a base image, the credential is predictable.
- **Fix direction**: Disable password auth or generate one-time locked credentials; prefer keyless/no-login VM images.

## Highest-Leverage Fix Order

1. Close direct execution surfaces: RS-001, RS-004, RS-005, RS-006.
2. Fix egress policy bypasses: RS-002, RS-003, RS-008, RS-011, RS-014.
3. Repair distributed correctness contracts: RS-015 through RS-021, plus RS-026 and RS-027.
4. Stabilize NymFS correctness: RS-009, RS-022 through RS-025, RS-034.
5. Bound speed and resource risks: RS-028, RS-030 through RS-042.

## Questions and Assumptions

- The Axum body-limit finding assumes no body-limit layer is added outside `../OtherYou/api/src`.
- The protected ingress mismatch assumes the generated Kubernetes ingress is routing to the vmd public ingress path, as rendered in the API manifest generator.
- Some security findings are cluster-internal rather than public-internet exposed, but they still cross the sandbox trust boundary because pod-network access is enough to reach control or guest execution surfaces.

## Fix Pass Status - 2026-04-24

This section records the follow-up patch status after the review. It intentionally distinguishes code fixes from mitigations and deferred architecture/operations work.

### Fixed or Directly Mitigated

- **Security**: RS-001, RS-002, RS-006, RS-007, RS-008, RS-009, RS-010, RS-011, RS-028, RS-029, RS-030, RS-043, RS-045.
- **Correctness**: RS-015, RS-016, RS-017, RS-018, RS-019, RS-020, RS-021, RS-022, RS-023, RS-024, RS-025, RS-034, RS-044.
- **Speed**: RS-031, RS-032, RS-035, RS-036, RS-038, RS-039, RS-040, RS-041, RS-042.

Key details from the final scoped fix pass:

- Public ingress no longer forwards user bearer/cookie credentials to guests, accepts protected launch query tokens, strips that token before upstream proxying, streams request/response bodies, caches host routes briefly, denies sandbox/control/database guest ports including port 0, and treats legacy exposure records with missing `auth_required` as protected by default.
- Guest portproxy shell/file/daemon RPC now requires a per-VM bearer token when bootstrapped by vmd, and facade/control-bus clients attach that token on readiness probes and all guest RPC calls. OtherYou redacts the internal token from persisted/returned provider metadata.
- HA/prod network startup now installs Envoy-process OUTPUT rules that allow the configured CoreDNS resolver but drop resolved loopback, link-local, private, carrier-grade NAT, benchmarking, multicast, and reserved IPv4 ranges. The private-range high-UDP WebRTC carveout is now local/dev only; HA/prod keeps private ranges denied.
- Envoy network-policy changes are now staged in vmd, candidate configs are validated with `envoy --mode validate` before an existing proxy is stopped, and in-memory VM policy maps commit only after the supervised restart path succeeds.
- Internal NymFS mutating routes now require owner/fence metadata, large FUSE flushes have an explicit API body limit, packed-file hashes/range reads use the file slot instead of the whole pack, and FUSE writes detect remote content changes before overwrite.
- VM lifecycle and distributed exec paths now enforce timeout/detach, dedupe command/stdin replay, validate stream identity before ACK, preserve paused VM cleanup paths, and use canonical route/fence records during reconciliation.
- Distributed node registry heartbeats and port allocation leases now grant once and use etcd lease keepalive streams on the normal path, falling back to a new lease only when keepalive fails.
- Prebuilt qcow2 downloads now enforce registry-published `.sha256` sidecars when present and write a computed `.sha256` sidecar next to cached images for provenance without adding network or full-file hashing to the normal cached launch path.
- API network-policy updates now roll back the persisted row on failed VM policy pushes when no newer update has superseded the write, reducing DB/runtime divergence without changing the existing endpoint contract or success path.
- HA partition monitoring now starts after the node registry has written its lease-backed key and fails the probe when no registry key is visible, so mutating paths do not run with an apparently reachable but scheduler-invisible control plane.
- Live frontend testing also found and fixed two UX-correctness issues outside the original sandbox list: the chat error fallback now preserves the Apps navigation menu so developer-mode surfaces are not stranded, and auth restore now dedupes StrictMode refreshes so rotating refresh tokens do not leave authenticated UI state with an unauthenticated API client.
- NATS command-bus clients now support an optional auth token, and prod wires the existing internal service token into NATS, vmd, and the API without changing subjects, streams, or local default behavior.
- Latest live prod smoke: approved test account, developer mode enabled, Apps -> Computer -> Watch reached a 1920x1080 ready video stream with no console/page/request errors and no premature Watching state. Readiness took 32.1s in this run, between the earlier 13.9s and 44.9s smokes, so watch startup latency still needs monitoring.

### Partially Mitigated

- **RS-004, RS-005**: Production Kubernetes NetworkPolicies now restrict vmd, etcd, and NATS ingress to expected pods, and NATS requires the shared internal service token from API/vmd clients. This reduces cluster-internal exposure but does not replace mTLS, separate NATS identities, or etcd auth/TLS.
- **RS-003, RS-014**: Header stripping, public port policy, body streaming, NetworkPolicy changes, HA/prod private high-UDP denial, Envoy resolved-private-IP blocks, atomic CoreDNS/Envoy config publishing, staged vmd policy maps, pre-stop Envoy config validation, and guarded API rollback reduce abuse and partial-policy paths. Arbitrary public high-port UDP remains open for WebRTC media, and generation-acknowledged API DB-to-runtime policy application with zero-gap Envoy hot restart remains a larger sandbox-network change.
- **RS-012, RS-013**: Base-image download locking, prebuilt qcow2 `.sha256` sidecar enforcement when published, root-password removal, and production d2vm/vmd digest pinning landed. Strong signature enforcement for every registry artifact and Docker-socket build hardening remain open.
- **RS-017**: Stream stdin sequencing, duplicate input handling, stream timeout/detach propagation, and completed-command markers landed. A broker redelivery after result publication but before ACK no longer reruns the command.
- **RS-027**: HA guardrail registration is fail-closed during VM startup, and partition monitoring now requires visible node-registry state. Broader quorum topology was not changed.
- **RS-037**: Reconcile writes canonical sharded route keys, but placement/admission still needs transactional per-node counters to remove full-prefix scans.

### Deferred

- **Speed**: RS-033 remains deferred.
- **Correctness/Operations**: RS-026 remains deferred because it requires production topology changes rather than a scoped code patch.

### Validation Notes

- Static validation passed for `reson-sandbox`: `cargo fmt --all -- --check`, `cargo check --workspace`, `cargo test -p vmd --lib`, `cargo test -p reson-sandbox --lib`, `cargo test -p portproxy`, and `git diff --check`.
- Static validation passed for `OtherYou`: `cargo fmt --check` and `rustup run 1.90 cargo check` in `api/`, `npm --prefix desktop run build`, `kubectl kustomize ops/gke/nym-prod`, the focused library test `rustup run 1.90 cargo test --lib build_vm_surface_uses_provider_metadata_for_node_and_owner_fence`, and `git diff --check`. A broader `cargo test build_vm_surface_uses_provider_metadata_for_node_and_owner_fence` attempt still compiles integration tests first and currently hits unrelated fixture drift in `tests/runtime_api.rs` for missing `CreateRuntimeTaskInput.initial_content_override`.
- Live deployed web smoke confirmed the prod test account is approved and developer mode is enabled; it also exposed a deployed-frontend UX issue where `Watching` appeared before video readiness. The final local fixed frontend against the production API passed the same browser flow with 1920x1080 video ready in 13.904s after the prebuilt-image digest sidecar change, no console/page errors, no premature `Watching` state, and no visible navigation regressions while entering the VM through the developer-mode Apps menu. Latest screenshot: `/private/tmp/nym-prod-live-flow-latest.png`.
