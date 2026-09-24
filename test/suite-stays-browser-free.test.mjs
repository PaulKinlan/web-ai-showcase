// The CI step that runs `node --test "test/**/*.test.mjs"` is labelled browser-free and earns that
// label by construction: browser-driven tests live in test-browser/ and run via deno task test:viewport.
// This asserts the property rather than trusting it, because a browser test dropped back into test/
// would silently make the suite slow, dependent on Chrome, and confusing when Chrome is absent
// (web-ai-showcase-cj5; the failure mode was 5x spawn ENOENT behind a multi-minute retry storm).
import test from "node:test";
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
const BROWSER_MARKERS = [/scripts\/browser\.mjs/, /\blaunchChrome\b/];

function testFiles(dir) {
  const out = [];
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...testFiles(p));
    else if (e.name.endsWith(".test.mjs")) out.push(p);
  }
  return out;
}

test("no file under test/ launches a browser (the bare suite stays dependency-free)", () => {
  const files = testFiles(join(ROOT, "test"));
  assert.ok(files.length > 5, "expected the fast suite to exist");
  const offenders = [];
  for (const f of files) {
    const text = readFileSync(f, "utf8");
    if (BROWSER_MARKERS.some((re) => re.test(text))) offenders.push(f.replace(ROOT, ""));
  }
  assert.deepEqual(
    offenders,
    [],
    `browser-driven test(s) inside the fast suite: ${offenders.join(", ")} — move them to test-browser/ ` +
      "and add them to deno task test:viewport (the CI step for test/ is labelled browser-free)",
  );
});
