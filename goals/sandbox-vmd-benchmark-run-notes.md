# Sandbox VMD Benchmark Run Notes

These notes exist so compaction or a new session does not waste time rediscovering
the same setup.

## Current Status

- Current fix verified: concurrent one-shot exec streams into one warm VM no
  longer show the ~20s tail in the live OtherYou product path when callers use
  `ExecOptions::close_stdin_on_start`.
- Changed files from the current pass:
  - `reson-sandbox/vmd/src/control_bus.rs`
  - `reson-sandbox/crates/reson-sandbox/src/lib.rs`
  - `reson-sandbox/portproxy/src/main.rs`
  - `reson-sandbox/portproxy/src/services.rs`
  - `reson-sandbox/portproxy/bin/portproxy-linux-amd64`
  - `reson-sandbox/portproxy/bin/portproxy-linux-arm64`
- Local checks passed:
  - `cargo fmt -p vmd`
  - `cargo test -p vmd vm_readiness_cache -- --nocapture`
  - `cargo check -p vmd`
  - `cargo fmt -p reson-sandbox -p portproxy`
  - `cargo check -p reson-sandbox`
  - `cargo test -p portproxy -- --nocapture`
- Remote image `reson-vmd:otheryou-migration-test` was rebuilt on `corvidae`
  after rebuilding the linux guest portproxy binaries.
- Live product-style validation hits the perf target after adding explicit
  one-shot stdin closure for product computer exec.
- Output correctness for tiny one-shot commands is fixed: after the portproxy
  drain-before-exit patch, every `printf exec-ok` row reports `stdout_bytes=7`.
- A client-side cached `ShellExecClient`/HTTP2 channel patch was tried and
  removed because it made the c4 tail worse. Do not re-add that without a
  specific lower-level proof.
- Root cause of the ~20s tail: product `Session::exec` kept the client
  request stream open for optional stdin, unlike direct portproxy one-shot exec
  paths that dropped the sender before invoking gRPC. Under c4 load this could
  return a stream handle quickly but delay the first stdout/exit event. The fix
  adds `ExecOptions::close_stdin_on_start` and uses it for OtherYou computer
  exec + the benchmark. Interactive stdin behavior remains available by
  leaving the option false.
- Current cleanup state: remote corvidae VMD nodes were stopped, stale remote
  test VM data was quarantined, and both local + remote integration control
  planes were purged after the final HA run.
- HA route reconciliation now only publishes session routes for VMs whose
  durable state is `running`. Discovery already normalizes crash-stale
  `running`/`paused`/`creating` metadata to `stopped`; stopped/cooled VMs remain
  discoverable by endpoint scan and get a fresh route when reattached, but they
  no longer consume workspace admission slots during reconciliation.

## Corvidae Image Refresh

From `/Users/crow/SoftwareProjects/OtherYou`:

```bash
rsync -az \
  --exclude target \
  --exclude .run \
  --exclude .integration-artifacts \
  --exclude .git \
  ../reson/reson-sandbox/ \
  crow@corvidae:/home/crow/reson-sandbox-migration-test/

ssh crow@corvidae \
  'cd /home/crow/reson-sandbox-migration-test && docker build --build-arg D2VM_SOURCE_IMAGE=reson-d2vm:otheryou-test -f Dockerfile.vmd-gke -t reson-vmd:otheryou-migration-test .'
```

## Exec-Throughput Benchmark

Use an absolute output dir. A relative `NYM_COMPUTER_BENCH_OUTPUT_DIR` breaks the
generated env sourcing path.

Use the newer Rust toolchain explicitly on this Mac. The default `rustc 1.86.0`
is too old for the current dependency set.

Focused single-node check:

```bash
cd /Users/crow/SoftwareProjects/OtherYou

NYM_RUSTUP_TOOLCHAIN=1.90-aarch64-apple-darwin \
NYM_COMPUTER_BENCH_OUTPUT_DIR="/Users/crow/SoftwareProjects/OtherYou/.run/computer-bench/oneshot-stdin-$(date -u +%Y%m%dT%H%M%SZ)" \
ops/dev/bench_computer_modes.sh \
  --modes single \
  --scenarios exec-throughput \
  --exec-concurrency-levels 1,4 \
  --exec-requests-per-level 16 \
  --restart-vmd
```

Focused multi/HA check:

```bash
cd /Users/crow/SoftwareProjects/OtherYou

NYM_RUSTUP_TOOLCHAIN=1.90-aarch64-apple-darwin \
NYM_COMPUTER_BENCH_OUTPUT_DIR="/Users/crow/SoftwareProjects/OtherYou/.run/computer-bench/oneshot-stdin-multi-ha-$(date -u +%Y%m%dT%H%M%SZ)" \
ops/dev/bench_computer_modes.sh \
  --modes multi,ha \
  --scenarios exec-throughput \
  --exec-concurrency-levels 4 \
  --exec-requests-per-level 16 \
  --restart-vmd
```

Do not leave this running unattended. If interrupted, run the cleanup below.

## Cleanup

```bash
cd /Users/crow/SoftwareProjects/OtherYou
ops/computer/remote_vmd_box.sh down --host corvidae
ops/computer/remote_vmd_box.sh status --host corvidae
ps -axo pid,command | rg 'bench_computer|remote_vmd_box|docker logs|vmdctl|target/debug/api|cargo run --bin api'
```

If local bench/API helper processes remain, kill only those matching the command
line above. Do not kill unrelated user work.

For stale remote HA route/admission state, use the scripted reset rather than a
handwritten SSH cleanup:

```bash
cd /Users/crow/SoftwareProjects/OtherYou
ops/computer/remote_vmd_box.sh reset --host corvidae
```

## Fixed Live Result

Output dir:

```text
/Users/crow/SoftwareProjects/OtherYou/.run/computer-bench/oneshot-stdin-20260614T230556Z
```

Observed after `ExecOptions::close_stdin_on_start` was wired into the product
computer exec path and benchmark:

- warm VM connect: about 4.44s.
- c1: 16/16 success, throughput 3.65/s, p95 about 871ms, max about 2.52s.
- c4: 16/16 success, throughput 20.15/s, p95 about 234ms, max about 363ms.
- c4 stream establish p95 about 234ms; first event p95 about 234ms.
- c4 CSV: `.run/computer-bench/oneshot-stdin-20260614T230556Z/computer-exec-single-c4-20260614T230659Z.csv`

Additional focused backend-mode checks:

- Multi-node c4 output dir:
  `/Users/crow/SoftwareProjects/OtherYou/.run/computer-bench/oneshot-stdin-multi-ha-20260614T230948Z`
  - 16/16 success, p95 about 188ms, max about 195ms.
- HA c4 initially failed before connect with
  `workspace admission budget exhausted (used=65, limit=64)`.
  Purging local and remote integration control-plane containers was not enough:
  remote etcd still had 65 `/reson-sandbox/sessions/` routes because stale
  remote VM data was re-registering old sessions on VMD startup.
- Quarantined corvidae test VM data under
  `/home/crow/reson-sandbox-migration-test/.run/quarantine-20260614T231434Z`,
  purged remote etcd/NATS, then reran HA c4.
- HA c4 clean output dir:
  `/Users/crow/SoftwareProjects/OtherYou/.run/computer-bench/oneshot-stdin-ha-clean-20260614T231505Z`
  - warm VM connect about 5.03s.
  - 16/16 success, throughput 12.25/s, p95 about 470ms, max about 575ms.
  - c4 CSV: `.run/computer-bench/oneshot-stdin-ha-clean-20260614T231505Z/computer-exec-ha-c4-20260614T231615Z.csv`

Checks after this patch:

- `cargo check -p reson-sandbox`
- `rustup run 1.90-aarch64-apple-darwin cargo check --bin computer_exec_bench`
- `rustup run 1.90-aarch64-apple-darwin cargo check` in `OtherYou/api`
- `cargo test -p portproxy -- --nocapture` passed unsandboxed. The sandboxed
  run fails only because listener bind is denied by the execution sandbox.

Cleanup after this run:

- `ops/computer/remote_vmd_box.sh status --host corvidae` showed both
  `reson-vmd-corvidae-node1` and `reson-vmd-corvidae-node2` exited.
- Local process check showed no leftover benchmark/API helper processes beyond
  the `ps | rg` check itself.
- After the final HA check, quarantined the new remote HA test VM data under
  `/home/crow/reson-sandbox-migration-test/.run/quarantine-20260614T231745Z`
  and purged both remote and local integration control planes.

## Previous Bad Live Result

Output dir:

```text
/Users/crow/SoftwareProjects/OtherYou/.run/computer-bench/stream-cache-fix-20260614T230002Z
```

Observed after the portproxy output-ordering fix, after removing the failed
client channel-cache patch, and after caching `GuestRpcAccess` in
`reson-sandbox`:

- c1: 16/16 success, p95 about 399ms.
- c4: 16/16 success, p50 about 143ms, p95 about 5.1s, max about 19.8s.
- c4 CSV: `.run/computer-bench/stream-cache-fix-20260614T230002Z/computer-exec-single-c4-20260614T230049Z.csv`
- c4 stream establishment is now healthy: p95 about 174ms. The slow row had
  `stream_establish_ms=108.122` and `first_event_ms=19787.284`, with
  `stdout_bytes=7` and `exit_code=0`.

Direct guest portproxy isolation against one manually-created VM was fast:

- VM `62df0d9c-3461-4a7b-96e7-c496c6bf720e`, RPC port `35257`, direct
  `portproxy_probe` with concurrency 4.
- 16/16 success, p50 about 102ms, p95/max about 605ms.

Current conclusion: the guest portproxy and direct corvidae VM RPC path are not
the ~20s tail source. The tail was above direct portproxy: the
`reson-sandbox` `Session::exec` one-shot/stdin request-stream path kept stdin
open for product one-shot commands. Re-running VMD setup without changing that
layer was wasted time.
