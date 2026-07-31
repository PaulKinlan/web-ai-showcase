export const SITE_PREFIX = "/web-ai-showcase";
export const CANONICAL_ORIGIN = "https://webai.show";
export const UPSTREAM_ORIGIN = "https://paulkinlan.github.io";

const PUBLIC_ROOT_FILES = new Set([
  "download-routes.json",
  "favicon.ico",
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
  "storage",
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
  headers.delete("set-cookie");
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) headers.set(name, value);
  headers.set("Cache-Control", "public, max-age=0, must-revalidate");
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function publicPath(pathname: string): boolean {
  const relative = pathname.replace(/^\/+|\/+$/g, "");
  if (!relative) return true;
  const segments = relative.split("/");
  if (segments.some((segment) => !segment || segment.startsWith("."))) return false;
  return segments.length === 1
    ? PUBLIC_ROOT_FILES.has(segments[0]) || PUBLIC_DIRECTORIES.has(segments[0])
    : PUBLIC_DIRECTORIES.has(segments[0]);
}

function stripLegacyPrefix(pathname: string): string | null {
  if (pathname === SITE_PREFIX || pathname === `${SITE_PREFIX}/`) return "/";
  if (pathname.startsWith(`${SITE_PREFIX}/`)) return pathname.slice(SITE_PREFIX.length);
  return null;
}

function canonicalUrl(pathname: string): URL {
  return new URL(pathname, CANONICAL_ORIGIN);
}

const LEGACY_DEPLOY_HOSTS = new Set([
  "web-ai-showcase.paulkinlan-ea.deno.net",
  "web-ai-showcase-isolated.paulkinlan-ea.deno.net",
]);

function isLegacyDeployHost(hostname: string): boolean {
  return LEGACY_DEPLOY_HOSTS.has(hostname);
}

function upstreamRequest(request: Request, url: URL): Request {
  const target = new URL(`${SITE_PREFIX}${url.pathname}${url.search}`, UPSTREAM_ORIGIN);
  const headers = new Headers();
  for (const name of ["accept", "if-modified-since", "if-none-match", "range"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Request(target, { method: request.method, headers, redirect: "follow" });
}

function isRewritableContentType(contentType: string): boolean {
  return /(?:text\/(?:html|css|javascript)|application\/(?:javascript|json|manifest\+json)|image\/svg\+xml)/i
    .test(contentType);
}

function rewriteApplicationUrls(source: string): string {
  let text = source;
  for (
    const base of [
      "https://paulkinlan.github.io/web-ai-showcase/",
      "https://web-ai-showcase.paulkinlan-ea.deno.net/web-ai-showcase/",
      "https://web-ai-showcase-isolated.paulkinlan-ea.deno.net/web-ai-showcase/",
    ]
  ) text = text.replaceAll(base, `${CANONICAL_ORIGIN}/`);
  // Rewrite quoted/root-relative application URLs without corrupting repository URLs such as
  // https://github.com/PaulKinlan/web-ai-showcase/… .
  return text.replace(/(["'`(=])\/web-ai-showcase\//g, "$1/");
}

function htmlWithCanonical(source: string, url: URL): string {
  const href = canonicalUrl(url.pathname).href.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
  const withoutOld = source
    .replace(/<link\s+[^>]*rel=["']canonical["'][^>]*>\s*/gi, "")
    .replace(/<meta\s+[^>]*(?:property|name)=["']og:url["'][^>]*>\s*/gi, "");
  const metadata =
    `<link rel="canonical" href="${href}" />\n    <meta property="og:url" content="${href}" />\n    `;
  return withoutOld.replace(/<head(\s[^>]*)?>/i, (head) => `${head}\n    ${metadata}`);
}

async function canonicalized(response: Response, request: Request, url: URL): Promise<Response> {
  const headers = new Headers(response.headers);
  const canonical = canonicalUrl(url.pathname).href;
  const contentType = headers.get("content-type") ?? "";
  const rewritable = isRewritableContentType(contentType);
  headers.set("Link", `<${canonical}>; rel="canonical"`);
  // A rewritten GET is a different representation from GitHub Pages. Strip upstream validators on
  // HEAD too, otherwise a HEAD → conditional GET can incorrectly produce a 304 for upstream bytes.
  if (rewritable) {
    for (const name of ["content-encoding", "content-length", "etag", "last-modified"]) {
      headers.delete(name);
    }
  }
  if (request.method === "HEAD" || !response.body) {
    return isolated(
      new Response(null, { status: response.status, statusText: response.statusText, headers }),
    );
  }
  if (!rewritable) {
    return isolated(
      new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers,
      }),
    );
  }
  let text = rewriteApplicationUrls(await response.text());
  if (/^text\/html/i.test(contentType)) text = htmlWithCanonical(text, url);
  return isolated(
    new Response(text, { status: response.status, statusText: response.statusText, headers }),
  );
}

export function createHandler(fetchUpstream: typeof fetch = fetch) {
  return async (request: Request): Promise<Response> => {
    const url = new URL(request.url);
    if (request.method !== "GET" && request.method !== "HEAD") {
      return isolated(
        new Response("Method not allowed", { status: 405, headers: { Allow: "GET, HEAD" } }),
      );
    }

    const legacyPath = stripLegacyPrefix(url.pathname);
    if (legacyPath !== null) {
      const target = canonicalUrl(legacyPath);
      target.search = url.search;
      return isolated(Response.redirect(target, 308));
    }
    if (isLegacyDeployHost(url.hostname)) {
      const target = canonicalUrl(url.pathname);
      target.search = url.search;
      return isolated(Response.redirect(target, 308));
    }
    if (!publicPath(url.pathname)) return isolated(new Response("Not found", { status: 404 }));

    try {
      const response = await fetchUpstream(upstreamRequest(request, url));
      return await canonicalized(response, request, url);
    } catch {
      return isolated(new Response("Upstream unavailable", { status: 502 }));
    }
  };
}

export const handler = createHandler();

if (import.meta.main) Deno.serve(handler);
