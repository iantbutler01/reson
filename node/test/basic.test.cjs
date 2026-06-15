// Offline regression tests (no LLM needed). Live model tests are separate and
// gated on CHEVALIER_TEST_MODEL.
const { test } = require("node:test");
const assert = require("node:assert");
const os = require("node:os");
const path = require("node:path");
const fs = require("node:fs");
const { Runtime, agentic, McpClient, McpServer, VfsStorage, version } = require("../index.js");

test("version()", () => {
  assert.match(version(), /^\d+\.\d+\.\d+/);
});

test("tool handler round-trip + schema introspection", async () => {
  const rt = new Runtime();
  let seen;
  await rt.tool({
    name: "add",
    description: "Add two numbers",
    schema: { type: "object", properties: { a: { type: "number" }, b: { type: "number" } }, required: ["a", "b"] },
    handler: async ({ a, b }) => {
      seen = { a, b };
      return String(a + b);
    },
  });
  assert.strictEqual(await rt.executeToolCall("add", { a: 2, b: 3 }), "5");
  assert.deepStrictEqual(seen, { a: 2, b: 3 });
  const schemas = await rt.getToolSchemas();
  assert.ok(schemas.some((s) => s.name === "add"));
});

test("vfs local round-trip", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "chev-test-"));
  const vfs = VfsStorage.local(root);
  await vfs.write("a.txt", Buffer.from("hi chevalier"));
  assert.strictEqual((await vfs.read("a.txt")).toString(), "hi chevalier");
  const st = await vfs.stat("a.txt");
  assert.strictEqual(st.kind, "File");
});

test("mcp server + client end-to-end", async () => {
  const server = new McpServer("test", { version: "0.0.1" });
  await server.tool(
    "echo",
    "echo back",
    { type: "object", properties: { m: { type: "string" } }, required: ["m"] },
    async ({ m }) => m,
  );
  server.serve("http", "127.0.0.1:38091").catch(() => {});
  await new Promise((r) => setTimeout(r, 1000));
  const client = await McpClient.http("http://127.0.0.1:38091/mcp");
  const tools = await client.listTools();
  assert.ok((tools.tools || []).some((t) => t.name === "echo"));
  const res = await client.callTool("echo", { m: "pong" });
  assert.ok(JSON.stringify(res).includes("pong"));
});

test("agentic() injects a Runtime as the last arg", () => {
  const fn = agentic({ model: "x" }, (a, rt) => (rt instanceof Runtime ? a * 2 : -1));
  assert.strictEqual(fn(21), 42);
});

test("schema-only tool registers (no handler)", async () => {
  const rt = new Runtime();
  await rt.tool({ name: "search", schema: { type: "object", properties: { q: { type: "string" } } } });
  const schemas = await rt.getToolSchemas();
  assert.ok(schemas.some((s) => s.name === "search"));
});
