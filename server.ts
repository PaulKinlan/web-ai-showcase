import { serveDir } from "jsr:@std/http/file-server";

export const SITE_PREFIX = "/web-ai-showcase";

const PUBLIC_ROOT_FILES = new Set([
  "download-routes.json",
  "home-core.mjs",
  "home.js",
  "index.html",
  "models.json",
  "sw.js",
]);
const PUBLIC_DIRECTORIES = new Set([
  "architecture",
  "explore",
  "image-credits",
  "image-provenance",
  "lib",
  "media",
  "models",
  "public",
  "reports",
  "search",
]);

export const ISOLATION_HEADERS = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Resource-Policy": "same-origin",
  "Origin-Agent-Cluster": "?1",
  "Referrer-Policy": "strict-origin-when-cross-origin",
  "X-Content-Type-Options": "nosniff",
} as const;

function isolated(response: Response): Response {
  const headers = new Headers(response.headers);
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) headers.set(name, value);
  headers.set("Cache-Control", "public, max-age=0, must-revalidate");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function publicPath(pathname: string): boolean {
  const relative = pathname.slice(SITE_PREFIX.length).replace(/^\/+/, "");
  if (!relative) return true;
  const segments = relative.split("/");
  if (segments.some((segment) => !segment || segment.startsWith("."))) return false;
  return segments.length === 1
    ? PUBLIC_ROOT_FILES.has(segments[0])
    : PUBLIC_DIRECTORIES.has(segments[0]);
}

export function createHandler(root = Deno.cwd()) {
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    if (request.method !== "GET" && request.method !== "HEAD") {
      return isolated(
        new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } }),
      );
    }
    if (url.pathname === "/") {
      return isolated(Response.redirect(new URL(`${SITE_PREFIX}/`, url), 308));
    }
    if (!url.pathname.startsWith(`${SITE_PREFIX}/`) || !publicPath(url.pathname)) {
      return isolated(new Response("Not found", { status: 404 }));
    }
    const fileUrl = new URL(request.url);
    fileUrl.pathname = fileUrl.pathname.slice(SITE_PREFIX.length) || "/";
    const fileRequest = new Request(fileUrl, request);
    const response = await serveDir(fileRequest, {
      fsRoot: root,
      quiet: true,
      showDirListing: false,
      showDotfiles: false,
    });
    return isolated(response);
  };
}

export const handler = createHandler();

if (import.meta.main) Deno.serve(handler);
