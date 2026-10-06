// Gallery attribution may include externally sourced manifest metadata. Keep the CI
// suite browser-free while checking the URL policy and forbidding HTML parsing there.
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { safeSourceHref } from "../lib/example-gallery.js";

const base = "https://example.test/web-ai-showcase/models/depth/";

test("gallery attribution links accept HTTP(S) and same-origin relative paths", () => {
  assert.equal(
    safeSourceHref("https://commons.wikimedia.org/wiki/File:Example", base),
    "https://commons.wikimedia.org/wiki/File:Example",
  );
  assert.equal(safeSourceHref("http://example.net/source", base), "http://example.net/source");
  assert.equal(safeSourceHref("./source", base), `${base}source`);
  assert.equal(safeSourceHref("/media/source", base), "https://example.test/media/source");
  assert.equal(safeSourceHref("#credit", base), `${base}#credit`);
});

test("gallery attribution links reject unsafe, empty, and disguised destinations", () => {
  for (
    const source of [
      "",
      "  ",
      null,
      undefined,
      "http://",
      "javascript:alert(1)",
      " JaVaScRiPt:alert(1) ",
      "data:text/html,<script>alert(1)</script>",
      "vbscript:MsgBox(1)",
      "//evil.test/payload",
      "\\\\evil.test/payload",
      "https:evil.test/payload",
      "java\tscript:alert(1)",
      "java\nscript:alert(1)",
      "javascript\t:alert(1)",
      "\tjavascript:alert(1)",
      "htt\nps://evil.test/x",
      "ftp://example.test/file",
    ]
  ) {
    assert.equal(safeSourceHref(source, base), null, `unsafe URL: ${source}`);
  }
});

function unsafeGalleryHTMLAssignments(source) {
  return [...source.matchAll(/\b(?:attr|caption)\.innerHTML\s*=\s*([^;]+);/g)]
    .filter(([, expression]) => !/^(["'])\1$/.test(expression.trim()));
}

test("gallery attribution never parses manifest values as HTML", () => {
  const source = readFileSync(new URL("../lib/example-gallery.js", import.meta.url), "utf8");
  assert.deepEqual(unsafeGalleryHTMLAssignments(source), []);
  assert.match(source, /link\.textContent\s*=\s*asset\.source/);
  assert.match(source, /safeSourceHref\(asset\.sourceUrl\)/);
  // Verify this assertion detects a regression rather than passing vacuously.
  assert.equal(unsafeGalleryHTMLAssignments("attr.innerHTML = `${asset.creator}`;").length, 1);
});
