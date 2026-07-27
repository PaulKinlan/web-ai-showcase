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

Deno.test("legacy prefixed paths redirect to the matching canonical path and query", async () => {
  const response = await handle(
    new Request(`${CANONICAL_ORIGIN}${SITE_PREFIX}/models/demo/?mode=inside`),
  );
  assertEquals(response.status, 308);
  assertEquals(response.headers.get("location"), `${CANONICAL_ORIGIN}/models/demo/?mode=inside`);
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
