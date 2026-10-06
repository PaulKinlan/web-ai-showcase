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

// ── Phase 0 CSP measurement (web-ai-showcase-ega) — DEV-FLAGGED, LOCAL-ONLY ─────────────────────
// Inert unless CSP_REPORT_ONLY=1 is set at startup. This exists so a local browser harness can
// measure REAL violations (Report-Only, nothing enforced) before the D1-D4 rollout decisions are
// taken; production deployments must not set the flag. The policy below is the DRAFT derived
// from the measured origin inventory on the bead: jsdelivr (transformers/ORT/tasks-vision),
// esm.run (web-llm), huggingface.co + the *.hf.co LFS/xet CDN family (measured live: weight
// resolves 302 to us.aws.cdn.hf.co — cdn-lfs*.huggingface.co alone would block every download),
// storage.googleapis.com (mediapipe .tflite). connect-src carries data: and blob: because
// transformers.js fetches caller-provided data:/blob: image+audio URLs inside the worker
// (MEASURED phase-0 run 4: a connect-src violation with blocked-uri 'data', source-file
// cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5, on yolos-detection — Report-Only only;
// enforcing without data: would break image-input families). 'unsafe-inline' in script-src/style-src
// is the phase-0 measurement stance (1521 inline bootstrap scripts exist); nonce strategies are a
// later decision. frame-ancestors is inert in Report-Only by spec — listed so the enforce flip
// is a one-word change.
export const CSP_PHASE0_POLICY = [
  "default-src 'self'",
  "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net https://esm.run",
  "worker-src 'self' blob:",
  "connect-src 'self' data: blob: https://huggingface.co https://*.hf.co https://cdn.jsdelivr.net https://esm.run https://storage.googleapis.com",
  "img-src 'self' data: blob: https:",
  "media-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "font-src 'self' data:",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'self'",
  "frame-ancestors 'none'",
].join("; ");

export const CSP_REPORT_PATH = "/__csp-report";

export function cspDevEnabled(env: { get(name: string): string | undefined } = Deno.env): boolean {
  try {
    return env.get("CSP_REPORT_ONLY") === "1";
  } catch {
    return false; // no --allow-env (e.g. deno test defaults): the feature is OFF.
  }
}

function stripUrlSecrets(value: string): string {
  // Signed HF/CDN tokens live in the query string — never log them. Guarantee (reviewer P2 #5,
  // precise): query + fragment are stripped from EVERY string, URL-parseable or not; path
  // segments are retained (a secret in a path would survive — accepted residual, Chrome's
  // report schema carries no such field today).
  try {
    const u = new URL(value);
    u.search = "";
    u.hash = "";
    return u.href.slice(0, 300);
  } catch {
    return value.replace(/[?#].*$/, "").slice(0, 300);
  }
}

/** Reduce a CSP report body to flat, length-capped fields with query/fragment secrets stripped.
 *  Accepts the legacy {csp-report:{...}} wrapper, flat CSP3 bodies, and single-entry
 *  Reporting-API arrays [{type,body}] (reviewer P2 #6 — previously degraded to {}). */
export function sanitizeCspReport(raw: unknown): Record<string, unknown> | null {
  let body: unknown = raw;
  if (Array.isArray(body)) {
    if (body.length !== 1) return null; // multi-entry arrays: unrecognized shape, refuse rather than half-log
    body = body[0];
  }
  if (body && typeof body === "object" && "type" in body && "body" in body) {
    body = (body as Record<string, unknown>)["body"]; // Reporting-API envelope
  }
  if (body && typeof body === "object" && "csp-report" in body) {
    body = (body as Record<string, unknown>)["csp-report"];
  }
  if (!body || typeof body !== "object") return null;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(body as Record<string, unknown>)) {
    const key = String(k).slice(0, 64); // reviewer P2 #4: keys are capped too
    if (typeof v === "string") {
      out[key] = stripUrlSecrets(v); // EVERY string, not just URL-ish ones (reviewer P2 #5)
    } else if (typeof v === "number" || typeof v === "boolean") {
      out[key] = v;
    }
    // Everything else (nested objects/arrays) is dropped: a report body must not smuggle blobs.
  }
  return out;
}

/** The sink (POST ingest + GET retrieval) exists only for the LOCAL harness: dev flag AND a
 *  loopback peer. This makes the local-only posture structural, not a comment (reviewer P1/P2). */
export function cspSinkAllowed(
  remoteAddr: { hostname: string } | undefined,
  enabled: boolean = cspDevEnabled(),
): boolean {
  if (!enabled || !remoteAddr) return false;
  const h = remoteAddr.hostname;
  return h === "::1" || h === "[::1]" || h === "localhost" || h === "127.0.0.1" || h.startsWith("127.");
}

// Ephemeral in-process sink (Phase 0): sanitized reports are kept in a bounded in-memory array
// (retrievable via loopback GET while the flag is on — Deno buffers stdout when redirected, so a
// stdout-only sink can lose every line to a SIGTERM) and echoed one-line-JSON to stdout.
// No files, no credentials, nothing retained past the process.
const CSP_REPORT_BUFFER_LIMIT = 1000;
const CSP_REPORT_BODY_LIMIT = 65_536;
const cspReports: Record<string, unknown>[] = [];
let cspReportsDropped = 0;

/** Read at most `limit` bytes; null means over-limit (caller answers 413) — bounds the
 *  ALLOCATION, not just the parsed string (reviewer P1). */
async function readBodyCapped(request: Request, limit: number): Promise<string | null> {
  const declared = Number(request.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > limit) return null;
  if (!request.body) return "";
  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      total += value.byteLength;
      if (total > limit) {
        await reader.cancel().catch(() => {});
        return null;
      }
      chunks.push(value);
    }
  } finally {
    try { reader.releaseLock(); } catch { /* already released */ }
  }
  const all = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.byteLength; }
  return new TextDecoder().decode(all);
}

function sinkResponse(body: BodyInit | null, status: number, headers: Record<string, string> = {}): Response {
  const r = isolated(new Response(body, { status, headers }));
  r.headers.set("Cache-Control", "no-store"); // a JSON dump of report data is never cacheable (reviewer P2 #2)
  return r;
}

async function cspReportSink(
  request: Request,
  remoteAddr: { hostname: string } | undefined,
): Promise<Response> {
  if (!cspSinkAllowed(remoteAddr)) return isolated(new Response("Not found", { status: 404 }));
  if (request.method === "GET") {
    // Local harness retrieval endpoint: the driver reads the buffer back before teardown.
    return sinkResponse(
      JSON.stringify({ count: cspReports.length, dropped: cspReportsDropped, reports: cspReports }),
      200,
      { "content-type": "application/json" },
    );
  }
  if (request.method !== "POST") {
    return sinkResponse("Method not allowed", 405, { Allow: "POST, GET" });
  }
  try {
    const text = await readBodyCapped(request, CSP_REPORT_BODY_LIMIT);
    if (text === null) return sinkResponse("Report too large", 413);
    const clean = sanitizeCspReport(JSON.parse(text));
    if (clean) {
      const entry = { t: Date.now(), ...clean };
      if (cspReports.length < CSP_REPORT_BUFFER_LIMIT) cspReports.push(entry);
      else cspReportsDropped++; // visible on GET, never a silent truncation (reviewer P2 #4)
      console.log(`CSP-REPORT ${JSON.stringify(entry)}`);
    }
    return sinkResponse(null, 204);
  } catch {
    return sinkResponse("Bad report", 400);
  }
}

function isolated(response: Response): Response {
  const headers = new Headers(response.headers);
  headers.delete("set-cookie");
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) headers.set(name, value);
  headers.set("Cache-Control", "public, max-age=0, must-revalidate");
  // Report-Only header (web-ai-showcase-ega): dev-flagged; NEVER enforced from this flag — the
  // enforce flip is a separate, explicit decision (D1-D4). When Phase 1 turns the flag on in the
  // Deno deployment this header is exactly what ships (Report-Only cannot block anything); the
  // report SINK stays loopback-only via cspSinkAllowed, so production reports would need their
  // own decision too.
  if (cspDevEnabled()) {
    headers.set(
      "Content-Security-Policy-Report-Only",
      `${CSP_PHASE0_POLICY}; report-uri ${CSP_REPORT_PATH}`,
    );
  }
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
  return async (
    request: Request,
    info?: { remoteAddr?: { hostname: string } },
  ): Promise<Response> => {
    const url = new URL(request.url);
    // Phase 0 CSP report sink (dev-flagged AND loopback-only; 404s otherwise — checked inside).
    if (url.pathname === CSP_REPORT_PATH) return await cspReportSink(request, info?.remoteAddr);
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
      const response = await fetchUpstream(upstreamRequest(request, url));
      return await canonicalized(response, request, url);
    } catch {
      return isolated(new Response("Upstream unavailable", { status: 502 }));
    }
  };
}

export const handler = createHandler();

if (import.meta.main) Deno.serve(handler);
