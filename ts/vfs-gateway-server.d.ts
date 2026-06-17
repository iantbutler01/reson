import type { VfsStorage } from "./native.js";
export interface VfsGatewayServerOptions {
    /** Map a request's `{owner_id}` to the backing store. Typically
     *  `(ownerId) => VfsStorage.local(scopeRootFor(ownerId))`. */
    resolveStore: (ownerId: string) => VfsStorage | Promise<VfsStorage>;
    /** If set, requests must carry `Authorization: Bearer <authToken>`. */
    authToken?: string;
    /** Route prefix the routes live under. Default `/internal/chevalier/vfs`. */
    routePrefix?: string;
}
/** Build a WHATWG `(Request) => Promise<Response>` handler that serves chevalier's
 *  VFS gateway protocol, delegating storage to `resolveStore(ownerId)`. */
export declare function createVfsGatewayServer(opts: VfsGatewayServerOptions): (req: Request) => Promise<Response>;
