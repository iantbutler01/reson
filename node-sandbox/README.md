# chevalier-sandbox

MicroVM **sandbox client** bindings for Chevalier. This is a thin, typed
[napi-rs](https://napi.rs) wrapper over the Rust sandbox client; it connects to
an external `vmd` daemon over gRPC and **never spawns one**. Kept separate from
the core [`chevalier`](../node) package because it pulls in a gRPC stack.

```bash
npm install chevalier-sandbox
```

## Usage

```ts
import { Sandbox } from "chevalier-sandbox";

const sb = await Sandbox.connect("http://127.0.0.1:8052", { authToken: process.env.VMD_TOKEN });

const sess = await sb.session({ image: "ubuntu:22.04", autoStart: true });
console.log(sess.sessionId, sess.vmId);

// Exec with a bidirectional handle: write stdin, read events.
const ex = await sess.exec("cat", { closeStdinOnStart: false });
await ex.write(Buffer.from("hello\n"));
await ex.eof();
for (;;) {
  const e = await ex.next();
  if (!e) break;
  if (e.type === "stdout") process.stdout.write(e.data);
  else if (e.type === "exit") console.log("exit", e.code);
}

// Files, fork (CoW):
await sess.writeFile("/tmp/x", Buffer.from("data"));
const bytes = await sess.readFile("/tmp/x");
const child = await sess.fork();           // returns the child Session
```

### Reconnecting

```ts
const again = await sb.attachSession(sess.sessionId);
```

## Status

v1 covers `connect`, `session`/`attachSession`, `exec` (bidirectional),
`readFile`/`writeFile`, `fork`, and `sessionId`/`vmId`. Follow-ups: `shell`,
`forwardPort`, `listDir`, and snapshot/restore + daemon-manager (the last two
need facade methods added to the Rust sandbox crate).

Requires a running `vmd` daemon — there is no live test here without one.

## License

Apache-2.0
