"use strict";
// Chevalier — TypeScript ergonomic layer over the native (napi-rs) bindings.
// Adds: Zod-typed structured output, Zod tool schemas, `for await` streaming,
// and the `agentic()` higher-order helper. Raw bindings live in ./native.js.
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.Runtime = exports.version = exports.VfsStorage = exports.McpServer = exports.McpClient = void 0;
exports.agentic = agentic;
const native = __importStar(require("./native.js"));
const zod_to_json_schema_1 = require("zod-to-json-schema");
var native_js_1 = require("./native.js");
Object.defineProperty(exports, "McpClient", { enumerable: true, get: function () { return native_js_1.McpClient; } });
Object.defineProperty(exports, "McpServer", { enumerable: true, get: function () { return native_js_1.McpServer; } });
Object.defineProperty(exports, "VfsStorage", { enumerable: true, get: function () { return native_js_1.VfsStorage; } });
Object.defineProperty(exports, "version", { enumerable: true, get: function () { return native_js_1.version; } });
function isZod(s) {
    return (!!s &&
        typeof s.parse === "function" &&
        typeof s.safeParse === "function");
}
function toJsonSchema(s) {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return isZod(s) ? (0, zod_to_json_schema_1.zodToJsonSchema)(s) : s;
}
/** The Chevalier agent runtime. */
class Runtime {
    constructor(options) {
        this.native = new native.Runtime(options);
    }
    /** Non-streaming inference. Pass `output` (Zod) to get a typed, validated `value`. */
    async run(args = {}) {
        const { output, ...rest } = args;
        const outputSchema = output ? toJsonSchema(output) : undefined;
        const res = (await this.native.run({ ...rest, outputSchema }));
        if (output && isZod(output)) {
            res.value = output.parse(JSON.parse(res.text));
        }
        return res;
    }
    /** Streaming inference as an async iterator: `for await (const ev of rt.runStream(...))`. */
    async *runStream(args = {}) {
        const { output, ...rest } = args;
        const outputSchema = output ? toJsonSchema(output) : undefined;
        const handle = await this.native.runStream({ ...rest, outputSchema });
        for (;;) {
            const ev = await handle.next();
            if (ev == null)
                return;
            yield ev;
        }
    }
    /** Register a tool. With `handler`, the engine runs it on `executeToolCall`;
     *  without, it's schema-only (the model can call it; you dispatch it). */
    async tool(def) {
        const schema = toJsonSchema(def.schema);
        if (def.handler) {
            const h = def.handler;
            await this.native.tool(def.name, def.description ?? "", schema, async (a) => String(await h(a)));
        }
        else {
            await this.native.registerToolSchema(def.name, def.description ?? "", schema);
        }
    }
    executeToolCall(toolName, args) {
        return this.native.executeToolCall(toolName, args);
    }
    getToolSchemas() {
        return this.native.getToolSchemas();
    }
    setSystemMessages(messages) {
        return this.native.setSystemMessages(messages);
    }
    setDefaultPrompt(prompt) {
        return this.native.setDefaultPrompt(prompt);
    }
    setProviderConfig(config) {
        return this.native.setProviderConfig(config);
    }
    rawResponse() {
        return this.native.rawResponse();
    }
    reasoning() {
        return this.native.reasoning();
    }
    reasoningSegments() {
        return this.native.reasoningSegments();
    }
    /** Connect to an MCP server and register its tools (auto-detected transport). */
    mcp(uri) {
        return this.native.mcp(uri);
    }
    /** Like `mcp`, but namespaces tools as `{label}_{tool}`. */
    mcpAs(uri, label) {
        return this.native.mcpAs(uri, label);
    }
}
exports.Runtime = Runtime;
/** "An agent is just a function." Wraps a function so a fresh `Runtime` is
 *  created per call and passed as the last argument. */
function agentic(config, fn) {
    return (...args) => fn(...[...args, new Runtime(config)]);
}
