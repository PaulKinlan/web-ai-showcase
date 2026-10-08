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

// Every URL this server emits — a redirect Location, a Link: rel="canonical" header, or an
// injected <link rel="canonical">/og:url — MUST resolve onto CANONICAL_ORIGIN. WHATWG URL
// resolution treats a leading "//" (or "\", a path separator for special schemes) as a
// network-path reference, so new URL("//evil.com", CANONICAL_ORIGIN) resolves to
// https://evil.com/, and new URL("//", CANONICAL_ORIGIN) has an empty host and throws. Validate
// the RESOLVED target's origin — never the raw input string — and refuse anything off-origin or
// unparseable. Redirects refuse with a 404, the same response the publicPath allowlist gives for
// unlisted paths; href sinks emit no canonical at all rather than an off-origin one.
function canonicalTarget(pathname: string): URL | null {
  let target: URL;
  try {
    target = canonicalUrl(pathname);
  } catch {
    return null;
  }
  return target.origin === CANONICAL_ORIGIN ? target : null;
}

function canonicalRedirectTarget(pathname: string, search: string): URL | null {
  const target = canonicalTarget(pathname);
  if (!target) return null;
  target.search = search;
  return target;
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
  // "manual" rather than "follow": the proxy must never blind-follow a 3xx whose destination it has
  // not checked. fetchUpstreamSameOrigin resolves and checks each hop itself.
  return new Request(target, { method: request.method, headers, redirect: "manual" });
}

const UPSTREAM_REDIRECT_STATUSES = new Set([301, 302, 303, 307, 308]);
// Bounded so a redirect cycle cannot hold a connection open indefinitely.
const MAX_UPSTREAM_REDIRECTS = 5;

// A redirect chain left UPSTREAM_ORIGIN. Thrown rather than returned so every caller gets the same
// isolated 502 and no body from the foreign origin is ever read, rewritten, or republished.
class OffOriginUpstreamRedirect extends Error {}

// Every hop of a redirect chain MUST stay on UPSTREAM_ORIGIN. Upstream is GitHub Pages, so a 3xx is
// normally a same-origin canonicalization (a trailing slash, a renamed path) and legitimately
// followed. A 3xx to any other origin is refused BEFORE the request is issued, so the foreign origin
// is never contacted and its bytes can never be republished under CANONICAL_ORIGIN. As with
// canonicalTarget, the RESOLVED target's origin is checked, never the raw Location string, so a
// network-path reference such as "//evil.com/x" is rejected rather than resolved off-origin.
async function fetchUpstreamSameOrigin(
  fetchUpstream: typeof fetch,
  request: Request,
  url: URL,
): Promise<Response> {
  let current = upstreamRequest(request, url);
  for (let hop = 0; hop <= MAX_UPSTREAM_REDIRECTS; hop++) {
    const response = await fetchUpstream(current);
    // Applied to EVERY response before any body is read or republished, whatever its status. The loop
    // above only ever issues requests to UPSTREAM_ORIGIN, but if the fetch implementation followed a
    // redirect on its own the final URL would show it. `response.url` is set by the fetch implementation
    // for every network response, so a non-empty value that is off-origin is a real escape; it is empty
    // only for a locally constructed Response, which is the test seam, where there is no network origin
    // to verify. Checking here rather than only on the non-redirect path also covers the case below where
    // a redirect status carries no Location and is therefore handed straight back unchecked.
    if (response.url) {
      let finalOrigin: string | null = null;
      try {
        finalOrigin = new URL(response.url).origin;
      } catch {
        finalOrigin = null;
      }
      if (finalOrigin !== UPSTREAM_ORIGIN) throw new OffOriginUpstreamRedirect();
    }
    if (!UPSTREAM_REDIRECT_STATUSES.has(response.status)) return response;
    const location = response.headers.get("location");
    // A 3xx with no Location cannot be followed; hand it back rather than guessing at a destination.
    if (!location) return response;
    let next: URL;
    try {
      next = new URL(location, current.url);
    } catch {
      throw new OffOriginUpstreamRedirect();
    }
    if (next.origin !== UPSTREAM_ORIGIN) throw new OffOriginUpstreamRedirect();
    current = new Request(next, {
      method: current.method,
      headers: current.headers,
      redirect: "manual",
    });
  }
  throw new OffOriginUpstreamRedirect();
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

function htmlWithCanonical(source: string, canonical: URL): string {
  const href = canonical.href.replaceAll("&", "&amp;").replaceAll('"', "&quot;");
  const withoutOld = source
    .replace(/<link\s+[^>]*rel=["']canonical["'][^>]*>\s*/gi, "")
    .replace(/<meta\s+[^>]*(?:property|name)=["']og:url["'][^>]*>\s*/gi, "");
  const metadata =
    `<link rel="canonical" href="${href}" />\n    <meta property="og:url" content="${href}" />\n    `;
  return withoutOld.replace(/<head(\s[^>]*)?>/i, (head) => `${head}\n    ${metadata}`);
}

async function canonicalized(response: Response, request: Request, url: URL): Promise<Response> {
  const headers = new Headers(response.headers);
  const canonical = canonicalTarget(url.pathname);
  const contentType = headers.get("content-type") ?? "";
  const rewritable = isRewritableContentType(contentType);
  if (canonical) headers.set("Link", `<${canonical.href}>; rel="canonical"`);
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
  if (canonical && /^text\/html/i.test(contentType)) text = htmlWithCanonical(text, canonical);
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
      const target = canonicalRedirectTarget(legacyPath, url.search);
      if (!target) return isolated(new Response("Not found", { status: 404 }));
      return isolated(Response.redirect(target, 308));
    }
    if (isLegacyDeployHost(url.hostname)) {
      const target = canonicalRedirectTarget(url.pathname, url.search);
      if (!target) return isolated(new Response("Not found", { status: 404 }));
      return isolated(Response.redirect(target, 308));
    }
    if (!publicPath(url.pathname)) return isolated(new Response("Not found", { status: 404 }));

    try {
      const response = await fetchUpstreamSameOrigin(fetchUpstream, request, url);
      return await canonicalized(response, request, url);
    } catch (error) {
      if (error instanceof OffOriginUpstreamRedirect) {
        // Named distinctly from an unreachable upstream: this is a refusal, not an outage, and the
        // message must not suggest the canonical origin can be retried into serving foreign bytes.
        return isolated(new Response("Upstream redirect refused", { status: 502 }));
      }
      return isolated(new Response("Upstream unavailable", { status: 502 }));
    }
  };
}

export const handler = createHandler();

if (import.meta.main) Deno.serve(handler);
