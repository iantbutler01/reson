import * as native from "./native.js";
import type { ZodType } from "zod";
export type { RunResult, ToolCallJs, ToolSchemaJs, StreamEvent, Message, MediaPartInput, GatewayOptions, ProviderConfigInput, AnthropicCacheConfig, } from "./native.js";
export { McpClient, McpServer, VfsStorage, version } from "./native.js";
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
export type TypedRunResult<T> = native.RunResult & {
    value?: T;
};
/** The Chevalier agent runtime. */
export declare class Runtime {
    /** @internal access to the raw napi runtime */
    readonly native: native.Runtime;
    constructor(options?: RuntimeOptions);
    /** Non-streaming inference. Pass `output` (Zod) to get a typed, validated `value`. */
    run<T = unknown>(args?: RunArgs<T>): Promise<TypedRunResult<T>>;
    /** Streaming inference as an async iterator: `for await (const ev of rt.runStream(...))`. */
    runStream<T = unknown>(args?: RunArgs<T>): AsyncGenerator<native.StreamEvent, void, void>;
    /** Register a tool. With `handler`, the engine runs it on `executeToolCall`;
     *  without, it's schema-only (the model can call it; you dispatch it). */
    tool(def: ToolDef): Promise<void>;
    executeToolCall(toolName: string, args: unknown): Promise<string>;
    getToolSchemas(): Promise<native.ToolSchemaJs[]>;
    setSystemMessages(messages: native.Message[]): Promise<void>;
    setDefaultPrompt(prompt: string): Promise<void>;
    setProviderConfig(config: native.ProviderConfigInput): Promise<void>;
    rawResponse(): Promise<string>;
    reasoning(): Promise<string>;
    reasoningSegments(): Promise<unknown>;
    /** Connect to an MCP server and register its tools (auto-detected transport). */
    mcp(uri: string): Promise<void>;
    /** Like `mcp`, but namespaces tools as `{label}_{tool}`. */
    mcpAs(uri: string, label: string): Promise<void>;
}
/** "An agent is just a function." Wraps a function so a fresh `Runtime` is
 *  created per call and passed as the last argument. */
export declare function agentic<A extends unknown[], R>(config: RuntimeOptions, fn: (...argsAndRuntime: [...A, Runtime]) => R): (...args: A) => R;
