import { strict as assert } from "node:assert";
import { test } from "node:test";
import { safeSourceHref } from "../lib/safe-url.js";

const base = "https://example.test/web-ai-showcase/image-credits/";
test("accepts http(s) and same-origin relative", () => {
  assert.equal(
    safeSourceHref("https://commons.wikimedia.org/x", base),
    "https://commons.wikimedia.org/x",
  );
  assert.equal(safeSourceHref("/a/b", base), "https://example.test/a/b");
  assert.equal(safeSourceHref("./a", base), `${base}a`);
});
test("rejects hostile, encoded, mixed-case, protocol-relative and empty", () => {
  for (
    const s of [
      "",
      " ",
      null,
      undefined,
      "javascript:alert(1)",
      "JaVaScRiPt:alert(1)",
      "java\tscript:alert(1)",
      "java\nscript:alert(1)",
      " \tjavascript:alert(1)",
      "data:text/html,<script>1</script>",
      "vbscript:x",
      "//evil.test/x",
      "\\\\evil.test/x",
      "https:evil.test/x",
      "htt\nps://evil.test/x",
      "ftp://a/b",
      "http://",
      "javascript&colon;alert(1)",
      "%6aavascript:alert(1)",
    ]
  ) {
    const r = safeSourceHref(s, base);
    if (s == null || typeof s !== "string" || !s.trim() || s === "http://") {
      assert.equal(r, null, String(s));
    }
    if (r !== null) assert.match(new URL(r).protocol, /^https?:$/, `unsafe: ${s} -> ${r}`);
    if (
      typeof s === "string" &&
      /^(\s*)(java\s*script:|data:|vbscript:|\/\/|\\\\|https:evil|htt\nps|ftp)/i.test(s)
    ) {
      assert.equal(r, null, s);
    }
  }
});

// ── non-DOM callers (web-ai-showcase-94s) ────────────────────────────────────────────────────────
// The ambient branch (`base` omitted, `document.baseURI` used) is what the gallery relies on, and it
// is the branch that threw in Worker/Node. These tests state the environment they run in rather than
// assuming it: if a future harness ever defines a global `document`, the first assertion fails loudly
// instead of letting the non-DOM cases silently test the DOM path.
test("non-DOM: absolute http(s) URLs resolve with the base omitted, relative ones stay null", () => {
  assert.equal(typeof document, "undefined", "this test must run without a DOM");
  assert.equal(safeSourceHref("https://example.com"), "https://example.com/");
  assert.equal(
    safeSourceHref("http://example.net/a/b?q=1#f"),
    "http://example.net/a/b?q=1#f",
  );
  assert.equal(safeSourceHref("  https://example.com/x  "), "https://example.com/x");
  assert.equal(safeSourceHref("HTTPS://EXAMPLE.COM/Path"), "https://example.com/Path");
  // No base means no origin to compare a relative path against: fail closed, never invent one.
  for (const relative of ["/a/b", "./a", "a/b", "../up"]) {
    assert.equal(safeSourceHref(relative), null, relative);
  }
  // The hostile set must stay rejected on this path too, not just with an explicit base.
  for (
    const hostile of [
      "javascript:alert(1)",
      "java\tscript:alert(1)",
      "data:text/html,<script>1</script>",
      "vbscript:x",
      "//evil.test/x",
      "\\\\evil.test/x",
      "https:evil.test/x",
      "ftp://a/b",
      "http://",
      "https://",
    ]
  ) {
    assert.equal(safeSourceHref(hostile), null, hostile);
  }
});

test("DOM: the ambient document.baseURI is still the default base, and the policy still applies", () => {
  const saved = globalThis.document;
  try {
    // The only way to exercise the ambient branch without a browser; restored in `finally` so no
    // later test can observe it (this process runs one test file, and node:test runs tests in order).
    Object.defineProperty(globalThis, "document", {
      value: { baseURI: base },
      configurable: true,
      writable: true,
    });
    assert.equal(
      safeSourceHref("https://commons.wikimedia.org/x"),
      "https://commons.wikimedia.org/x",
    );
    assert.equal(safeSourceHref("/a/b"), "https://example.test/a/b");
    assert.equal(safeSourceHref("./a"), `${base}a`);
    assert.equal(safeSourceHref("javascript:alert(1)"), null);
    // A relative URL that escapes the ambient origin is still refused, as before.
    assert.equal(safeSourceHref("https://other.test/a", base), "https://other.test/a");
    assert.equal(safeSourceHref("//other.test/a"), null);
  } finally {
    if (saved === undefined) delete globalThis.document;
    else globalThis.document = saved;
  }
  assert.equal(typeof document, "undefined", "the stubbed document must not leak");
});

test("Worker thread (real, no document and no location): absolute URLs resolve there too", async () => {
  const { Worker } = await import("node:worker_threads");
  const worker = new Worker(new URL("./fixtures/safe-url-worker.mjs", import.meta.url));
  try {
    const msg = await new Promise((resolve, reject) => {
      worker.once("message", resolve);
      worker.once("error", reject);
      worker.once("exit", (code) => {
        if (code !== 0) reject(new Error(`worker exited with code ${code}`));
      });
    });
    assert.equal(
      msg.noDocument,
      true,
      "the worker must have no document for this test to mean anything",
    );
    assert.equal(msg.absolute, "https://example.com/");
    assert.equal(msg.absoluteWithPath, "https://example.com/a/b?q=1#f");
    assert.equal(msg.absoluteUpper, "https://example.com/");
    assert.equal(msg.relative, null, "no base in a Worker: a relative path stays null");
    assert.equal(msg.relativeDot, null);
    assert.equal(msg.protocolRelative, null);
    assert.equal(msg.javascript, null);
    assert.equal(msg.data, null);
    assert.equal(msg.ftp, null);
  } finally {
    await worker.terminate();
  }
});
