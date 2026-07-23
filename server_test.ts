import { assertEquals, assertMatch } from "jsr:@std/assert";
import { createHandler, ISOLATION_HEADERS, SITE_PREFIX } from "./server.ts";

const handle = createHandler(Deno.cwd());

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

Deno.test("serves the catalogue under the GitHub Pages-compatible prefix", async () => {
  const response = await handle(new Request(`https://example.test${SITE_PREFIX}/`));
  assertEquals(response.status, 200);
  assertMatch(response.headers.get("content-type") ?? "", /^text\/html/);
  assertMatch(await response.text(), /Every model, running in your browser/);
  assertIsolated(response);
});

Deno.test("serves module assets with isolation headers", async () => {
  const response = await handle(
    new Request(`https://example.test${SITE_PREFIX}/lib/media-pipeline.js`),
  );
  assertEquals(response.status, 200);
  assertMatch(response.headers.get("content-type") ?? "", /javascript/);
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
