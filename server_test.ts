import { assertEquals, assertMatch, assertNotMatch } from "jsr:@std/assert";
import {
  CANONICAL_ORIGIN,
  createHandler,
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

// A redirect chain is the one place where the proxy stops choosing its own destination: the
// target is chosen by whatever the upstream responds with. These tests pin the rule that a hop may
// never leave UPSTREAM_ORIGIN, and — the part that actually matters — that the foreign origin is
// never CONTACTED, not merely that its bytes are discarded afterwards.
function redirectUpstream(location: string, status = 301) {
  const seen: string[] = [];
  const handleRedirect = createHandler((input) => {
    const request = input instanceof Request ? input : new Request(input);
    seen.push(request.url);
    return Promise.resolve(
      new Response(null, { status, headers: { location } }),
    );
  });
  return { handleRedirect, seen };
}

Deno.test("refuses an upstream redirect to another origin and never contacts it", async () => {
  const { handleRedirect, seen } = redirectUpstream("https://evil.example/steal.html");
  const response = await handleRedirect(new Request(`${CANONICAL_ORIGIN}/`));
  assertEquals(response.status, 502);
  assertIsolated(response);
  // The load-bearing assertion: the proxy asked upstream and stopped there. If this ever fails while
  // the status is still 502, the fix has degraded to discarding bytes it already fetched.
  assertEquals(seen.length, 1);
  assertMatch(seen[0], new RegExp(`^${UPSTREAM_ORIGIN}/`));
  assertNotMatch(seen.join(" "), /evil\.example/);
});

Deno.test("refuses a protocol-relative upstream redirect rather than resolving it off-origin", async () => {
  // Location "//evil.example/x" is a network-path reference: resolved against the upstream it
  // becomes https://evil.example/x, so the raw string must never be trusted as a path.
  const { handleRedirect, seen } = redirectUpstream("//evil.example/steal.html");
  const response = await handleRedirect(new Request(`${CANONICAL_ORIGIN}/`));
  assertEquals(response.status, 502);
  assertIsolated(response);
  assertEquals(seen.length, 1);
  assertNotMatch(seen.join(" "), /evil\.example/);
});

Deno.test("follows a same-origin upstream redirect so legitimate canonicalisation still works", async () => {
  const seen: string[] = [];
  const handleRedirect = createHandler((input) => {
    const request = input instanceof Request ? input : new Request(input);
    seen.push(request.url);
    const path = new URL(request.url).pathname;
    if (path === `${SITE_PREFIX}/models/demo`) {
      return Promise.resolve(
        new Response(null, { status: 301, headers: { location: `${SITE_PREFIX}/models/demo/` } }),
      );
    }
    return Promise.resolve(
      new Response("<!doctype html><html><head></head><body>Demo</body></html>", {
        headers: { "content-type": "text/html" },
      }),
    );
  });
  const response = await handleRedirect(new Request(`${CANONICAL_ORIGIN}/models/demo`));
  // Guard against the fix over-reaching into refusing all redirects: this one must still be followed.
  assertEquals(response.status, 200);
  assertEquals(seen.length, 2);
  assertMatch(seen[1], new RegExp(`^${UPSTREAM_ORIGIN}${SITE_PREFIX}/models/demo/$`));
  assertIsolated(response);
});

Deno.test("bounds a same-origin redirect cycle instead of looping forever", async () => {
  const seen: string[] = [];
  const handleLoop = createHandler((input) => {
    const request = input instanceof Request ? input : new Request(input);
    seen.push(request.url);
    return Promise.resolve(
      new Response(null, { status: 302, headers: { location: `${SITE_PREFIX}/models/loop` } }),
    );
  });
  // /models/… is inside the public allowlist; a bare /loop would 404 before upstream is ever called.
  const response = await handleLoop(new Request(`${CANONICAL_ORIGIN}/models/loop`));
  assertEquals(response.status, 502);
  assertIsolated(response);
  assertEquals(seen.length, 6);
});

Deno.test("rejects a redirect chain that leaves the upstream origin on a later hop", async () => {
  // First hop is a legitimate same-origin redirect; the escape is only visible on the second.
  const seen: string[] = [];
  const handleChain = createHandler((input) => {
    const request = input instanceof Request ? input : new Request(input);
    seen.push(request.url);
    const path = new URL(request.url).pathname;
    const location = path.endsWith("/models/hop1")
      ? "https://evil.example/steal.html"
      : `${SITE_PREFIX}/models/hop1`;
    return Promise.resolve(new Response(null, { status: 301, headers: { location } }));
  });
  const response = await handleChain(new Request(`${CANONICAL_ORIGIN}/models/start`));
  assertEquals(response.status, 502);
  assertIsolated(response);
  assertEquals(seen.length, 2);
  assertMatch(seen[1], new RegExp(`^${UPSTREAM_ORIGIN}${SITE_PREFIX}/models/hop1$`));
  assertNotMatch(seen.join(" "), /evil\.example/);
});

Deno.test("a redirect status with no Location is returned as-is rather than guessed at", async () => {
  const seen: string[] = [];
  const handleNoLocation = createHandler((input) => {
    const request = input instanceof Request ? input : new Request(input);
    seen.push(request.url);
    // 301 IS a redirect status, so this exercises the missing-Location branch rather than the
    // ordinary non-redirect path a 304 would take.
    return Promise.resolve(new Response(null, { status: 301 }));
  });
  const response = await handleNoLocation(new Request(`${CANONICAL_ORIGIN}/`));
  assertEquals(response.status, 301);
  assertEquals(seen.length, 1);
  assertIsolated(response);
});

// `Response.url` is a prototype getter and cannot be set through the constructor, so shadow it
// per-instance to simulate a fetch implementation that returned a response from somewhere other than
// UPSTREAM_ORIGIN. That is the only way to exercise the defence-in-depth branch.
function responseFrom(url: string, init: ResponseInit = {}, body = ""): Response {
  const response = new Response(body, init);
  Object.defineProperty(response, "url", { value: url, configurable: true });
  return response;
}

Deno.test("refuses a response whose final url is off-origin even on a non-redirect status", async () => {
  const handleForeign = createHandler(() =>
    Promise.resolve(
      responseFrom(
        "https://evil.example/steal.html",
        { status: 200, headers: { "content-type": "text/html" } },
        "<html>EXTERNAL</html>",
      ),
    )
  );
  const response = await handleForeign(new Request(`${CANONICAL_ORIGIN}/models/demo`));
  assertEquals(response.status, 502);
  assertIsolated(response);
});

Deno.test("refuses an off-origin response on a redirect status carrying no Location", async () => {
  // The exact gap: a redirect status with no Location is handed straight back, so without the
  // origin check running before that branch its body would be published under the canonical origin.
  const handleNoLocationForeign = createHandler(() =>
    Promise.resolve(
      responseFrom("https://evil.example/steal.html", {
        status: 301,
        headers: { "content-type": "text/html" },
      }),
    )
  );
  const response = await handleNoLocationForeign(new Request(`${CANONICAL_ORIGIN}/models/demo`));
  assertEquals(response.status, 502);
  assertIsolated(response);
});

Deno.test("accepts a same-origin final url so the origin check is not rejecting every response", async () => {
  const handleSameOrigin = createHandler(() =>
    Promise.resolve(
      responseFrom(
        `${UPSTREAM_ORIGIN}${SITE_PREFIX}/models/demo/`,
        { status: 200, headers: { "content-type": "text/html" } },
        "<html>fine</html>",
      ),
    )
  );
  const response = await handleSameOrigin(new Request(`${CANONICAL_ORIGIN}/models/demo`));
  assertEquals(response.status, 200);
  assertEquals(await response.text(), "<html>fine</html>");
  assertIsolated(response);
});
