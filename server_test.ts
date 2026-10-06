import { assertEquals, assertMatch, assertNotMatch } from "jsr:@std/assert";
import {
  CANONICAL_ORIGIN,
  createHandler,
  sanitizeCspReport,
  ISOLATION_HEADERS,
  SITE_PREFIX,
  UPSTREAM_ORIGIN,
} from "./server.ts";

const requests: Request[] = [];
const handle = createHandler((input) => {
  const request = input instanceof Request ? input : new Request(input);
  requests.push(request);
  const path = new URL(request.url).pathname;
  if (path.endsWith("media-pipeline.js")) {
    return Promise.resolve(
      new Response('import "/web-ai-showcase/lib/helper.js";', {
        headers: { "content-type": "text/javascript", "set-cookie": "not-forwarded=1" },
      }),
    );
  }
  if (path.endsWith("sample.bin")) {
    return Promise.resolve(
      new Response(new Uint8Array([1, 2, 3]), {
        headers: { "content-type": "application/octet-stream" },
      }),
    );
  }
  return Promise.resolve(
    new Response(
      '<!doctype html><html><head><title>Every model</title></head><body><a href="/web-ai-showcase/models/demo/">Demo</a></body></html>',
      { headers: { "content-type": "text/html", etag: "upstream-etag" } },
    ),
  );
});

function assertIsolated(response: Response) {
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) {
    assertEquals(response.headers.get(name), value, name);
  }
}

Deno.test("canonical root proxies the prefixed GitHub Pages root", async () => {
  const response = await handle(new Request(`${CANONICAL_ORIGIN}/?q=audio`));
  assertEquals(response.status, 200);
  const upstream = new URL(requests.at(-1)!.url);
  assertEquals(upstream.origin, UPSTREAM_ORIGIN);
  assertEquals(upstream.pathname, `${SITE_PREFIX}/`);
  assertEquals(upstream.search, "?q=audio");
  const html = await response.text();
  assertMatch(html, /<link rel="canonical" href="https:\/\/webai\.show\/"/);
  assertMatch(html, /<meta property="og:url" content="https:\/\/webai\.show\/"/);
  assertMatch(html, /href="\/models\/demo\/"/);
  assertNotMatch(html, /href="\/web-ai-showcase\//);
  assertEquals(response.headers.get("link"), '<https://webai.show/>; rel="canonical"');
  assertEquals(response.headers.get("etag"), null);
  assertIsolated(response);
});

Deno.test("canonical favicon proxies the root asset", async () => {
  const response = await handle(new Request(`${CANONICAL_ORIGIN}/favicon.ico`));
  assertEquals(response.status, 200);
  assertEquals(new URL(requests.at(-1)!.url).pathname, `${SITE_PREFIX}/favicon.ico`);
  assertIsolated(response);
});

Deno.test("legacy prefixed paths redirect to the matching canonical path and query", async () => {
  const response = await handle(
    new Request(`${CANONICAL_ORIGIN}${SITE_PREFIX}/models/demo/?mode=inside`),
  );
  assertEquals(response.status, 308);
  assertEquals(response.headers.get("location"), `${CANONICAL_ORIGIN}/models/demo/?mode=inside`);
  assertIsolated(response);
});

Deno.test("legacy prefixed redirects never leave the canonical origin", async () => {
  // A remainder of "//evil.com" is a WHATWG network-path reference: resolving it against the
  // canonical origin yields https://evil.com/. "\\" is a path separator for special schemes,
  // so the backslash variants parse to the same "//…" pathname. All must be refused.
  for (
    const path of [
      `${SITE_PREFIX}//evil.com`,
      `${SITE_PREFIX}//evil.com/phish?from=webai`,
      `${SITE_PREFIX}//webai.show.evil.com/`,
      `${SITE_PREFIX}/\\evil.com`,
      `${SITE_PREFIX}/\\evil.com\\phish`,
    ]
  ) {
    const response = await handle(new Request(`${CANONICAL_ORIGIN}${path}`));
    assertEquals(response.status, 404, path);
    assertEquals(response.headers.get("location"), null, path);
    assertIsolated(response);
  }
});

Deno.test("legacy prefixed encoded-slash path still redirects, staying on the canonical origin", async () => {
  // %2f is not a path separator in WHATWG URL parsing, so this is a plain path, not a
  // network-path reference. The redirect must keep working and stay same-origin.
  const response = await handle(new Request(`${CANONICAL_ORIGIN}${SITE_PREFIX}/%2f%2fevil.com`));
  assertEquals(response.status, 308);
  const location = response.headers.get("location");
  assertEquals(location && new URL(location).origin, CANONICAL_ORIGIN);
  assertIsolated(response);
});

Deno.test("legacy Deno deployment hosts redirect to webai.show", async () => {
  for (
    const host of [
      "web-ai-showcase.paulkinlan-ea.deno.net",
      "web-ai-showcase-isolated.paulkinlan-ea.deno.net",
    ]
  ) {
    const response = await handle(new Request(`https://${host}/models/demo/?x=1`));
    assertEquals(response.status, 308);
    assertEquals(response.headers.get("location"), `${CANONICAL_ORIGIN}/models/demo/?x=1`);
  }
});

Deno.test("legacy deploy-host redirects never leave the canonical origin", async () => {
  for (
    const host of [
      "web-ai-showcase.paulkinlan-ea.deno.net",
      "web-ai-showcase-isolated.paulkinlan-ea.deno.net",
    ]
  ) {
    for (const path of ["//evil.com", "//evil.com/phish?from=deploy", "/\\evil.com"]) {
      const response = await handle(new Request(`https://${host}${path}`));
      assertEquals(response.status, 404, `${host}${path}`);
      assertEquals(response.headers.get("location"), null, `${host}${path}`);
      assertIsolated(response);
    }
  }
});

Deno.test("proxies root-level directory and model routes through the upstream prefix", async () => {
  for (
    const path of ["/explore/", "/architecture/", "/image-credits/", "/storage/", "/models/demo/"]
  ) {
    const response = await handle(new Request(`${CANONICAL_ORIGIN}${path}`));
    assertEquals(response.status, 200, path);
    assertEquals(new URL(requests.at(-1)!.url).pathname, `${SITE_PREFIX}${path}`);
    const html = await response.text();
    assertMatch(
      html,
      new RegExp(`href="${path === "/models/demo/" ? "/models/demo/" : "/models/demo/"}`),
    );
    assertMatch(html, new RegExp(`<link rel="canonical" href="https://webai\\.show${path}`));
    assertIsolated(response);
  }
});

Deno.test("rewrites root-relative module imports and strips cookies", async () => {
  const response = await handle(new Request(`${CANONICAL_ORIGIN}/lib/media-pipeline.js`));
  assertEquals(response.status, 200);
  assertEquals(await response.text(), 'import "/lib/helper.js";');
  assertEquals(response.headers.get("set-cookie"), null);
  assertIsolated(response);
});

Deno.test("does not transform binary assets", async () => {
  const response = await handle(new Request(`${CANONICAL_ORIGIN}/media/sample.bin`));
  assertEquals([...new Uint8Array(await response.arrayBuffer())], [1, 2, 3]);
  assertEquals(
    response.headers.get("link"),
    '<https://webai.show/media/sample.bin>; rel="canonical"',
  );
});

Deno.test("HEAD responses expose canonical and isolation headers without a body", async () => {
  const response = await handle(
    new Request(`${CANONICAL_ORIGIN}/models/demo/`, { method: "HEAD" }),
  );
  assertEquals(response.status, 200);
  assertEquals(response.headers.get("link"), '<https://webai.show/models/demo/>; rel="canonical"');
  assertEquals(response.headers.get("etag"), null);
  assertEquals(response.headers.get("last-modified"), null);
  assertEquals(await response.text(), "");
  assertIsolated(response);
});

Deno.test("does not expose repository internals", async () => {
  for (const path of ["/.git/config", "/CLAUDE.md", "/scripts/inventory.mjs"]) {
    const response = await handle(new Request(`${CANONICAL_ORIGIN}${path}`));
    assertEquals(response.status, 404, path);
    assertIsolated(response);
  }
});

Deno.test("never emits an off-origin canonical href for network-path-reference paths", async () => {
  // "//models/demo/" is a WHATWG network-path reference: new URL(pathname, CANONICAL_ORIGIN)
  // resolves with "models" as the host, so any canonical derived from it would be cross-origin.
  // The invariant canonicalRedirectTarget() enforces for redirects applies to the href sinks too:
  // the response must carry NO canonical at all rather than one whose origin is not
  // CANONICAL_ORIGIN.
  const response = await handle(new Request(`${CANONICAL_ORIGIN}//models/demo/`));
  assertEquals(response.status, 200);
  const link = response.headers.get("link");
  if (link !== null) {
    const href = link.match(/^<([^>]+)>; rel="canonical"$/)?.[1];
    assertEquals(href && new URL(href).origin, CANONICAL_ORIGIN);
  }
  const html = await response.text();
  assertNotMatch(html, /<link rel="canonical"/, "injected <link rel=canonical>");
  assertNotMatch(html, /og:url/, "injected og:url");
  assertNotMatch(html, /https:\/\/models\//, "cross-origin host must not appear");
  // URL rewriting still applies; only the canonical emission is refused.
  assertMatch(html, /href="\/models\/demo\/"/);
  assertIsolated(response);
});

Deno.test("malformed network-path-reference paths get an honest status, never a synthetic 5xx", async () => {
  // "//" resolves to an empty host, so new URL throws. On the canonical host the upstream
  // response must pass through untouched (the mock's 200) instead of a synthetic 502.
  const passthrough = await handle(new Request(`${CANONICAL_ORIGIN}//`));
  assertEquals(passthrough.status, 200);
  assertEquals(passthrough.headers.get("link"), null);
  assertNotMatch(await passthrough.text(), /<link rel="canonical"/);
  assertIsolated(passthrough);

  // "///evil.com" on a legacy deploy host reaches the redirect path, where the same throw must
  // become a 404 — never an uncaught 500.
  const response = await handle(
    new Request("https://web-ai-showcase.paulkinlan-ea.deno.net///evil.com"),
  );
  assertEquals(response.status, 404);
  assertEquals(response.headers.get("location"), null);
  assertIsolated(response);
});

Deno.test("ordinary pages keep a same-origin canonical target", async () => {
  // Regression: hardening the malformed cases must not break the canonical for real pages.
  const response = await handle(new Request(`${CANONICAL_ORIGIN}/models/demo/`));
  assertEquals(response.status, 200);
  const link = response.headers.get("link");
  const linkHref = link?.match(/^<([^>]+)>; rel="canonical"$/)?.[1];
  assertEquals(linkHref && new URL(linkHref).origin, CANONICAL_ORIGIN);
  assertEquals(linkHref, `${CANONICAL_ORIGIN}/models/demo/`);
  const html = await response.text();
  assertMatch(html, /<link rel="canonical" href="https:\/\/webai\.show\/models\/demo\/"/);
  assertMatch(html, /<meta property="og:url" content="https:\/\/webai\.show\/models\/demo\/"/);
  assertIsolated(response);
});

Deno.test("rejects state-changing methods", async () => {
  const response = await handle(new Request(`${CANONICAL_ORIGIN}/`, { method: "POST" }));
  assertEquals(response.status, 405);
  assertEquals(response.headers.get("allow"), "GET, HEAD");
  assertIsolated(response);
});

Deno.test("returns an isolated 502 when GitHub Pages is unavailable", async () => {
  const failing = createHandler(() => Promise.reject(new Error("offline")));
  const response = await failing(new Request(`${CANONICAL_ORIGIN}/`));
  assertEquals(response.status, 502);
  assertIsolated(response);
});

// ── Phase 0 CSP measurement sink (web-ai-showcase-ega): dev-flagged, OFF by default ──────────────
Deno.test("CSP report sink is 404 and never proxied upstream when the dev flag is off", async () => {
  requests.length = 0;
  const response = await handle(
    new Request("https://webai.show/__csp-report", {
      method: "POST",
      body: JSON.stringify({ "csp-report": { "blocked-uri": "https://evil.example/x" } }),
      headers: { "content-type": "application/csp-report" },
    }),
  );
  assertEquals(response.status, 404);
  assertEquals(requests.length, 0, "the sink must never reach the upstream fetch");
});

Deno.test("ordinary responses carry no CSP header when the dev flag is off", async () => {
  const response = await handle(new Request("https://webai.show/"));
  assertEquals(response.headers.get("content-security-policy-report-only"), null);
  assertEquals(response.headers.get("content-security-policy"), null);
});

Deno.test("sanitizeCspReport strips query secrets, flattens, caps, and rejects junk", async () => {
  const clean = sanitizeCspReport({
    "csp-report": {
      "blocked-uri": "https://us.aws.cdn.hf.co/xet-bridge-us/abc?Signature=SUPERSECRET&Policy=xyz",
      "document-uri": "https://webai.show/models/yolos-detection/?q=secret#frag",
      "violated-directive": "connect-src",
      "status-code": 200,
      "nested": { "smuggled": "https://evil.example?token=abc" },
      "long": "x".repeat(5000),
    },
  });
  assertMatch(JSON.stringify(clean), /us\.aws\.cdn\.hf\.co\/xet-bridge-us\/abc"/);
  assertNotMatch(JSON.stringify(clean), /SUPERSECRET|Policy=xyz|q=secret|smuggled|token=abc/);
  assertEquals(clean!["violated-directive"], "connect-src");
  assertEquals(clean!["status-code"], 200);
  assertEquals((clean!["long"] as string).length, 300);
  assertEquals(sanitizeCspReport(null), null);
  assertEquals(sanitizeCspReport("not-an-object"), null);
});
