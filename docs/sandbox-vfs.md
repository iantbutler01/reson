# Sandbox And VFS

Chevalier has one sandbox interface and multiple backends. Product code should choose the backend by config, not by branching into separate APIs.

## Backends

| Backend | When to use it | Notes |
| --- | --- | --- |
| Local `vmd` | local dev, self-hosted prod, direct control of VM host | Uses Docker/QEMU and Chevalier's own daemon/control plane |
| Distributed `vmd` | multi-node or HA self-hosted deployments | Adds etcd/NATS control routing without changing the facade API |
| OpenComputer | managed sandbox provider | Uses the same facade, with OpenComputer-specific config under the hood |

The facade keeps the common operations stable:

- create or attach a session
- execute commands
- read and write files
- fork sessions where the backend supports it
- mount shared VFS paths into the guest

## VFS Mounts

For local `vmd`, shared VFS mounts are handled by Chevalier's daemon and guest helpers.

For OpenComputer, Chevalier uses OpenComputer command mounts. The guest image or template must contain:

```text
/usr/local/bin/chevalier-vfs-fuse
```

The command mount runs that binary with a VFS endpoint, scope, mount tag, and mountpoint. The sandbox calls back to the product API for VFS reads and writes, so the product must provide an internal base URL reachable from the sandbox.

For OtherYou, `NYM_OPENCOMPUTER_API_INTERNAL_BASE_URL` is that callback URL. In local testing it is usually a tunnel to the local API server; in prod it should be the API address the OpenComputer VM can reach for VFS gateway requests.

## OtherYou Local Testing

OtherYou can point at OpenComputer with a config swap:

```bash
cd ../OtherYou
cp infra/env/api.env.opencomputer.local.example infra/env/api.env.opencomputer.local
# fill OPENCOMPUTER_API_KEY
# fill OPENCOMPUTER_TEMPLATE_ID for a guest image with chevalier-vfs-fuse
# fill NYM_OPENCOMPUTER_API_INTERNAL_BASE_URL with the sandbox-reachable API callback URL

./ops/dev/run_api_opencomputer.sh --watch
```

That path should exercise the same VFS behavior through the sandbox facade. If it needs special product code beyond config, treat that as a compatibility bug unless there is a clear provider limitation.

## Verification

Use the narrow package gates while iterating:

```bash
cd sandbox
make verify

cd ../vfs
cargo test --all-features
```

Use real product surfaces before deployment. A successful crate test does not prove the full chain of API -> sandbox -> FUSE mount -> VFS gateway -> storage.
