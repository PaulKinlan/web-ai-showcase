import { assertEquals, assertMatch } from "jsr:@std/assert";
import { createHandler, ISOLATION_HEADERS, SITE_PREFIX, UPSTREAM_ORIGIN } from "./server.ts";

const requests: Request[] = [];
const handle = createHandler((input) => {
  const request = input instanceof Request ? input : new Request(input);
  requests.push(request);
  const path = new URL(request.url).pathname;
  if (path.endsWith("media-pipeline.js")) {
    return Promise.resolve(
      new Response("export const ok = true", {
        headers: { "content-type": "text/javascript", "set-cookie": "not-forwarded=1" },
      }),
    );
  }
  return Promise.resolve(
    new Response("<!doctype html><title>Every model, running in your browser</title>", {
      headers: { "content-type": "text/html" },
    }),
  );
});

function assertIsolated(response: Response) {
  for (const [name, value] of Object.entries(ISOLATION_HEADERS)) {
    assertEquals(response.headers.get(name), value, name);
  }
}

Deno.test("root redirects to the stable site prefix with isolation headers", async () => {
  const response = await handle(new Request("https://example.test/"));
  assertEquals(response.status, 308);
  assertEquals(response.headers.get("location"), `https://example.test${SITE_PREFIX}/`);
  assertIsolated(response);
});

Deno.test("proxies the catalogue from the published GitHub Pages site", async () => {
  const response = await handle(new Request(`https://example.test${SITE_PREFIX}/?q=audio`));
  assertEquals(response.status, 200);
  assertMatch(response.headers.get("content-type") ?? "", /^text\/html/);
  assertMatch(await response.text(), /Every model, running in your browser/);
  const upstream = new URL(requests.at(-1)!.url);
  assertEquals(upstream.origin, UPSTREAM_ORIGIN);
  assertEquals(upstream.pathname, `${SITE_PREFIX}/`);
  assertEquals(upstream.search, "?q=audio");
  assertIsolated(response);
});

Deno.test("proxies top-level directory-index routes", async () => {
  for (const directory of ["explore", "architecture", "image-credits", "storage"]) {
    const path = `${SITE_PREFIX}/${directory}/`;
    const response = await handle(new Request(`https://example.test${path}`));
    assertEquals(response.status, 200, path);
    assertEquals(new URL(requests.at(-1)!.url).pathname, path);
    assertIsolated(response);
  }
});

Deno.test("proxies directory-index model routes with their trailing slash", async () => {
  const path = `${SITE_PREFIX}/models/smolvlm-vision-language/`;
  const response = await handle(new Request(`https://example.test${path}`));
  assertEquals(response.status, 200);
  assertEquals(new URL(requests.at(-1)!.url).pathname, path);
  assertIsolated(response);
});

Deno.test("proxies module assets, strips cookies, and adds isolation headers", async () => {
  const response = await handle(
    new Request(`https://example.test${SITE_PREFIX}/lib/media-pipeline.js`),
  );
  assertEquals(response.status, 200);
  assertMatch(response.headers.get("content-type") ?? "", /javascript/);
  assertEquals(response.headers.get("set-cookie"), null);
  assertIsolated(response);
});

Deno.test("does not expose repository internals", async () => {
  for (const path of ["/.git/config", "/CLAUDE.md", "/scripts/inventory.mjs"]) {
    const response = await handle(new Request(`https://example.test${SITE_PREFIX}${path}`));
    assertEquals(response.status, 404, path);
    assertIsolated(response);
  }
});

Deno.test("rejects state-changing methods", async () => {
  const response = await handle(
    new Request(`https://example.test${SITE_PREFIX}/`, { method: "POST" }),
  );
  assertEquals(response.status, 405);
  assertEquals(response.headers.get("allow"), "GET, HEAD");
  assertIsolated(response);
});

Deno.test("returns an isolated 502 when GitHub Pages is unavailable", async () => {
  const failing = createHandler(() => Promise.reject(new Error("offline")));
  const response = await failing(new Request(`https://example.test${SITE_PREFIX}/`));
  assertEquals(response.status, 502);
  assertIsolated(response);
});
