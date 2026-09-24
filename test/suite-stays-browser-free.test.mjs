// The CI step that runs `node --test "test/**/*.test.mjs"` is labelled browser-free and earns that
// label by construction: browser-driven tests live in test-browser/ and run via deno task test:viewport.
// This asserts the property rather than trusting it, because a browser test dropped back into test/
// would silently make the suite slow, dependent on Chrome, and confusing when Chrome is absent
// (web-ai-showcase-cj5; the failure mode was 5x spawn ENOENT behind a multi-minute retry storm).
//
// The property is that no file under test/ LAUNCHES a browser — not that none of them imports the
// harness. Importing scripts/browser.mjs resolves nothing and spawns nothing, so pure helpers like
// resolveChromeBinary() are exactly the kind of logic that belongs in the fast suite with a unit test
// (web-ai-showcase-fdy), and flagging the import alone would push such tests into the slow,
// browser-gated file where they would be skipped whenever no browser is installed.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const ROOT = new URL("..", import.meta.url).pathname;
// Calls that actually start, drive, or connect to a browser process.
const LAUNCH_MARKERS = [
  /\blaunchChrome\s*\(/,
  /\bstartServer\s*\(/,
  /\bnew CDP\s*\(/,
  /\bopenPage\s*\(/,
  /\bsetViewport\s*\(/,
  /\bpuppeteer\b/,
  /\bplaywright\b/,
];

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
    if (LAUNCH_MARKERS.some((re) => re.test(text))) offenders.push(f.replace(ROOT, ""));
  }
  assert.deepEqual(
    offenders,
    [],
    `browser-driven test(s) inside the fast suite: ${offenders.join(", ")} — move them to test-browser/ ` +
      "and add them to deno task test:viewport (the CI step for test/ is labelled browser-free)",
  );
});

test("those markers are not vacuous: the browser suite does call the launcher", () => {
  // Without this, deleting every marker would make the check above pass silently forever.
  const browserFiles = testFiles(join(ROOT, "test-browser"));
  assert.ok(browserFiles.length > 0, "expected a test-browser/ suite to exist");
  const launching = browserFiles.filter((f) =>
    LAUNCH_MARKERS.some((re) => re.test(readFileSync(f, "utf8")))
  );
  assert.ok(
    launching.length > 0,
    "no file in test-browser/ matches the launch markers, so the fast-suite guard proves nothing",
  );
});

test("the two suites stay declared apart, so the browser one cannot be orphaned", () => {
  const deno = JSON.parse(readFileSync(join(ROOT, "deno.json"), "utf8"));
  assert.match(
    String(deno.tasks?.["test:viewport"] || ""),
    /test-browser/,
    "the browser suite needs a declared task of its own (cj5)",
  );
  // The fast suite has no deno task: CI runs it directly. So the property to assert is that no task
  // anywhere mixes the two globs, and that CI declares the browser step instead of leaving it orphaned.
  for (const [name, cmd] of Object.entries(deno.tasks || {})) {
    if (!String(cmd).includes("node --test")) continue;
    assert.ok(
      !(String(cmd).includes("test-browser") && String(cmd).includes("test/**")),
      `task ${name} runs the fast and browser suites in one glob, so the browser-free label breaks`,
    );
  }
  const workflow = readFileSync(join(ROOT, ".github/workflows/gate.yml"), "utf8");
  // Plain containment, not a regex: escaping a glob full of asterisks is how this check first went
  // green-looking but wrong.
  assert.ok(workflow.includes("node --test \"test/**/*.test.mjs\""), "CI should run the fast glob");
  assert.match(workflow, /node --test test-browser\//, "CI should declare the browser suite");
  const fastStep = workflow.split(/^\s*- name:/m).find((s) => s.includes('test/**/*.test.mjs'));
  assert.ok(fastStep && !fastStep.includes("test-browser"), "the fast CI step must not run the browser suite");
});
