// ESM entry point. The implementation (and the native addon loader) is CommonJS;
// this re-exports it so `import { Runtime } from "chevalier"` works in ESM /
// "type": "module" projects without relying on named-export auto-detection.
import cjs from "./index.js";

export const { Runtime, agentic, ChevalierError, McpClient, McpServer, VfsStorage, version, createVfsGatewayServer } = cjs;
export default cjs;
