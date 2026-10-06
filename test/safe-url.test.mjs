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
    if (r !== null) assert.match(new URL(r).protocol, /^https?:$/, `unsafe: ${s} -> ${r}`);
    if (typeof s === "string" && /^(\s*)(java\s*script:|data:|vbscript:|\/\/|\\\\|https:evil|htt\nps|ftp)/i.test(s)) {
      assert.equal(r, null, s);
    }
  }
});
