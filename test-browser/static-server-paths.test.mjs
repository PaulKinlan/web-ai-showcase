// HTTP-only security regression for the shared dev static server; no Chrome is launched.
import test from "node:test";
import assert from "node:assert/strict";
import { request } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, relative } from "node:path";
import { tmpdir } from "node:os";
import { BASE, repoRoot, startServer } from "../scripts/browser.mjs";

function get(port, path) {
  return new Promise((resolve, reject) => {
    const req = request({ hostname: "127.0.0.1", port, path }, (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on(
        "end",
        () => resolve({ status: res.statusCode, body: Buffer.concat(chunks).toString() }),
      );
    });
    req.on("error", reject);
    req.end();
  });
}

test("dev static server refuses decoded traversal outside its repo while retaining normal routes", async () => {
  const dir = mkdtempSync(join(tmpdir(), "web-ai-static-traversal-"));
  const marker = "OUTSIDE_REPO_STATIC_SENTINEL";
  const outside = join(dir, "secret.txt");
  writeFileSync(outside, marker);
  const traversal = relative(repoRoot, outside).replaceAll("/", "%2f");
  assert.match(traversal, /%2e%2e|\.\./, "fixture must actually escape repoRoot");
  const { server, port } = await startServer();
  try {
    for (const path of [`${BASE}${traversal}`, `/${traversal}`]) {
      const response = await get(port, path);
      assert.equal(response.status, 404, `escaped path ${path} must not be served`);
      assert.ok(!response.body.includes(marker));
    }
    assert.equal((await get(port, `${BASE}index.html`)).status, 200);
    assert.equal((await get(port, BASE)).status, 200);
    assert.equal((await get(port, `${BASE}%00index.html`)).status, 404);
    assert.equal((await get(port, `${BASE}%ZZ`)).status, 404);
  } finally {
    await new Promise((resolve) => server.close(resolve));
    rmSync(dir, { recursive: true, force: true });
  }
});
