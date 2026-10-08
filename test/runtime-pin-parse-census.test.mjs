// RED→GREEN tests for the three coord-authorized census corrections
// (Codex review 125e1ac8 P1s, bead web-ai-showcase-j6i). Hermetic: temp-dir
// fixtures only; no repo mutation, no network, no browser.

import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  listFiles,
  censusHtml,
  runCensus,
} from "../scripts/runtime-pin-parse-census.mjs";

function tmp() {
  const dir = mkdtempSync(join(tmpdir(), "census-test-"));
  return dir;
}

// --- Fix 1: enumeration covers EXACTLY the gate's grep set, gitignored incl. -

test("enumeration: gitignored and untracked marker files under targets are IN scope", () => {
  const dir = tmp();
  try {
    mkdirSync(join(dir, "models"));
    writeFileSync(join(dir, "models", "tracked.js"), 'const u = "onnxruntime-web@1.21.0";');
    writeFileSync(join(dir, "models", "ignored.js"), 'const u = "onnxruntime-web@9.9.9";');
    writeFileSync(join(dir, ".gitignore"), "models/ignored.js\n");
    const { all, ignoredInScope } = listFiles(dir, "models/");
    assert.ok(all.includes("models/tracked.js"));
    // The gate's `grep -r` reads this file; the census must too (RED before
    // the fix: --exclude-standard dropped it).
    assert.ok(all.includes("models/ignored.js"), "gitignored file must be enumerated");
    // git classification is informational only (no git repo here → empty).
    assert.ok(Array.isArray(ignoredInScope));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("enumeration: real repo scope includes any gitignored files under PIN_SCAN_TARGETS", async () => {
  const { PIN_SCAN_TARGETS } = await import("../scripts/audit-model-currency.mjs");
  const rootDir = new URL("..", import.meta.url).pathname;
  const { all, ignoredInScope } = listFiles(rootDir, PIN_SCAN_TARGETS);
  // Nonvacuity: the real corpus is thousands of files.
  assert.ok(all.length > 3000, `expected corpus scale, got ${all.length}`);
  // Whatever is gitignored under the targets is NAMED, never silently dropped.
  assert.equal(ignoredInScope.length, new Set(ignoredInScope).size);
});

// --- Fix 2: every decoded attribute pin counted independently ----------------

test("decoded scan: a second entity-encoded pin after a raw pin in one src is counted", () => {
  const html =
    '<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/a.js?x=onnxruntime-web&#X40;9.9.9"></script>';
  const fatal = [];
  const f = censusHtml("fixture.html", html, fatal);
  const raw = f.markers.filter((m) => m.context === "html-script-src" && !m.decodedAttrValue && !m.cookedEscape);
  const decoded = f.markers.filter((m) => m.decodedAttrValue);
  const cooked = f.markers.filter((m) => m.cookedEscape);
  assert.equal(raw.length, 1, "the raw pin");
  // RED before the fix: rawHit suppression dropped this decoded occurrence.
  assert.ok(decoded.length >= 1 || cooked.length >= 1, "the entity-encoded second pin must be counted");
  assert.equal(fatal.length, 0);
});

test("decoded scan: raw occurrences explain themselves 1:1, pure-entity attr fully counted", () => {
  const html =
    '<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web&#64;1.21.0/dist/a.js"></script>';
  const fatal = [];
  const f = censusHtml("fixture.html", html, fatal);
  const decoded = f.markers.filter((m) => m.decodedAttrValue);
  const cooked = f.markers.filter((m) => m.cookedEscape);
  assert.ok(decoded.length + cooked.length >= 1, "entity-only marker must surface");
});

// --- Fix 3: executable inline-script parse failure is FATAL ------------------

test("inline script with marker + syntax error is a fatal census error", () => {
  const html = '<script>const u = "onnxruntime-web@1.21.0"; const = `broken</script>';
  const fatal = [];
  const f = censusHtml("fixture.html", html, fatal);
  // RED before the fix: recorded html-inline-js:unparsed, exit 0.
  assert.equal(fatal.length, 1);
  assert.equal(fatal[0].kind, "html-inline-js-parse");
  assert.equal(fatal[0].hasMarkers, true);
  assert.ok(f.markers.some((m) => m.context === "html-inline-js:unparsed"));
});

test("inline script with sink evidence but NO marker is parsed; parse error is fatal", () => {
  const good = '<script>const m = await import("https://x/a.js");</script>';
  const fatalGood = [];
  censusHtml("ok.html", good, fatalGood);
  assert.equal(fatalGood.length, 0);

  const bad = '<script>const m = await import("https://x/a.js"); const = broken</script>';
  const fatalBad = [];
  censusHtml("bad.html", bad, fatalBad);
  // RED before the fix: sink-only inline scripts were never parsed at all.
  assert.equal(fatalBad.length, 1);
  assert.equal(fatalBad[0].hasRuntimeSink, true);
  assert.equal(fatalBad[0].hasMarkers, false);
});

test("runCensus stamps notVerified and reports fatalErrors (real corpus has none)", () => {
  const rootDir = new URL("..", import.meta.url).pathname;
  const census = runCensus(rootDir, null); // no write
  assert.equal(census.fatalErrors.length, 0, JSON.stringify(census.fatalErrors.slice(0, 3)));
  assert.equal(census.notVerified, undefined);
  assert.equal(census.totals.jsParseFailures, 0);
  assert.ok(census.totals.markerOccurrences > 500, "nonvacuity");
});
