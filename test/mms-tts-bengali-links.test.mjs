import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

const root = fileURLToPath(new URL("../", import.meta.url));
const family = "models/mms-tts-bengali/";
const base = "https://example.test/web-ai-showcase/";
const manifest = JSON.parse(readFileSync(join(root, family, "acceptance.json"), "utf8"));
const critique = JSON.parse(readFileSync(join(root, family, "_questions.json"), "utf8"));

for (const { route } of manifest.rungs) {
  test(`${route}: local links resolve to existing files`, () => {
    const html = readFileSync(join(root, route, "index.html"), "utf8");
    for (const [, , href] of html.matchAll(/\bhref\s*=\s*(["'])(.*?)\1/g)) {
      const url = new URL(href, base + route);
      if (!url.href.startsWith(base)) continue;
      const path = join(root, decodeURIComponent(url.pathname.slice("/web-ai-showcase/".length)));
      const file = statSync(path).isDirectory() ? join(path, "index.html") : path;
      assert(statSync(file).isFile(), `${route}: ${href} must resolve`);
    }
  });
}

test("critique describes the same existing routes as the acceptance manifest", () => {
  assert.deepEqual(
    critique.evidenceScope.routes.slice().sort(),
    manifest.rungs.map((r) => r.route).sort(),
  );
  assert.equal(critique.id, "mms-tts-bengali");
});

test("page and critique contain no inherited language-code or region errors", () => {
  const html = readFileSync(join(root, family, "index.html"), "utf8");
  assert.doesNotMatch(html, /Bengali \(tam\)|Bengali Nadu/);
  assert.doesNotMatch(JSON.stringify(critique), /[\u0b80-\u0bff]|lang=ta\b|en\s*→\s*ta\b/);
});
