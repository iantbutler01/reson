// Chevalier — TypeScript ergonomic layer over the native (napi-rs) bindings.
// Adds: Zod-typed structured output, Zod tool schemas, `for await` streaming,
// and the `agentic()` higher-order helper. Raw bindings live in ./native.js.

import * as native from "./native.js";
import { zodToJsonSchema } from "zod-to-json-schema";
import type { ZodType } from "zod";

export type {
  RunResult,
  ToolCallJs,
  ToolSchemaJs,
  StreamEvent,
  Message,
  MediaPartInput,
  GatewayOptions,
  ProviderConfigInput,
  AnthropicCacheConfig,
} from "./native.js";
export { McpClient, McpServer, VfsStorage, version } from "./native.js";

function isZod(s: unknown): s is ZodType {
  return (
    !!s &&
    typeof (s as { parse?: unknown }).parse === "function" &&
    typeof (s as { safeParse?: unknown }).safeParse === "function"
  );
}

function toJsonSchema(s: ZodType | object): unknown {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  return isZod(s) ? (zodToJsonSchema(s as any) as unknown) : s;
}

export interface RuntimeOptions {
  /** Provider model string, e.g. `anthropic:claude-3-5-sonnet` or
   *  `custom-openai:my-model@server_url=http://host:port/v1/chat/completions`. */
  model?: string;
  apiKey?: string;
}

export interface RunArgs<T = unknown> {
  prompt?: string;
  system?: string;
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  model?: string;
  apiKey?: string;
  /** Zod schema (gives `result.value` typed + validated) or a raw JSON Schema. */
  output?: ZodType<T> | object;
  outputType?: string;
  history?: native.Message[];
  timeoutMs?: number;
}

export interface ToolDef {
  name: string;
  description?: string;
  /** Zod schema or raw JSON Schema describing the tool's args. */
  schema: ZodType | object;
  /** Async handler. Omit for a schema-only (host-dispatched) tool. */
  handler?: (args: any) => string | Promise<string>;
}

/** Result of `run`, with an optional decoded `value` when an output schema is given. */
export type TypedRunResult<T> = native.RunResult & { value?: T };

/** The Chevalier agent runtime. */
export class Runtime {
  /** @internal access to the raw napi runtime */
  readonly native: native.Runtime;

  constructor(options?: RuntimeOptions) {
    this.native = new native.Runtime(options);
  }

  /** Non-streaming inference. Pass `output` (Zod) to get a typed, validated `value`. */
  async run<T = unknown>(args: RunArgs<T> = {}): Promise<TypedRunResult<T>> {
    const { output, ...rest } = args;
    const outputSchema = output ? toJsonSchema(output) : undefined;
    const res = (await this.native.run({ ...rest, outputSchema })) as TypedRunResult<T>;
    if (output && isZod(output)) {
      res.value = (output as ZodType<T>).parse(JSON.parse(res.text));
    }
    return res;
  }

  /** Streaming inference as an async iterator: `for await (const ev of rt.runStream(...))`. */
  async *runStream<T = unknown>(args: RunArgs<T> = {}): AsyncGenerator<native.StreamEvent, void, void> {
    const { output, ...rest } = args;
    const outputSchema = output ? toJsonSchema(output) : undefined;
    const handle = await this.native.runStream({ ...rest, outputSchema });
    for (;;) {
      const ev = await handle.next();
      if (ev == null) return;
      yield ev;
    }
  }

  /** Register a tool. With `handler`, the engine runs it on `executeToolCall`;
   *  without, it's schema-only (the model can call it; you dispatch it). */
  async tool(def: ToolDef): Promise<void> {
    const schema = toJsonSchema(def.schema);
    if (def.handler) {
      const h = def.handler;
      await this.native.tool(def.name, def.description ?? "", schema, async (a: any) => String(await h(a)));
    } else {
      await this.native.registerToolSchema(def.name, def.description ?? "", schema);
    }
  }

  executeToolCall(toolName: string, args: unknown): Promise<string> {
    return this.native.executeToolCall(toolName, args);
  }
  getToolSchemas(): Promise<native.ToolSchemaJs[]> {
    return this.native.getToolSchemas();
  }
  setSystemMessages(messages: native.Message[]): Promise<void> {
    return this.native.setSystemMessages(messages);
  }
  setDefaultPrompt(prompt: string): Promise<void> {
    return this.native.setDefaultPrompt(prompt);
  }
  setProviderConfig(config: native.ProviderConfigInput): Promise<void> {
    return this.native.setProviderConfig(config);
  }
  rawResponse(): Promise<string> {
    return this.native.rawResponse();
  }
  reasoning(): Promise<string> {
    return this.native.reasoning();
  }
  reasoningSegments(): Promise<unknown> {
    return this.native.reasoningSegments();
  }
  /** Connect to an MCP server and register its tools (auto-detected transport). */
  mcp(uri: string): Promise<void> {
    return this.native.mcp(uri);
  }
  /** Like `mcp`, but namespaces tools as `{label}_{tool}`. */
  mcpAs(uri: string, label: string): Promise<void> {
    return this.native.mcpAs(uri, label);
  }
}

/** "An agent is just a function." Wraps a function so a fresh `Runtime` is
 *  created per call and passed as the last argument. */
export function agentic<A extends unknown[], R>(
  config: RuntimeOptions,
  fn: (...argsAndRuntime: [...A, Runtime]) => R,
): (...args: A) => R {
  return (...args: A) => fn(...([...args, new Runtime(config)] as [...A, Runtime]));
}
