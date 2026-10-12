// Deterministic, browser-free guard against design-token drift in the generated report pages
// (bead web-ai-showcase-bj1z).
//
// WHY: reports/conformance/index.html and reports/responsive/index.html shipped a hand-written palette of
// literal colours instead of the site's design tokens, and search/__poc__/bench.html did too. Nothing in
// the repo noticed for months: there was no freshness check (scripts/check-conformance.mjs validates
// reports/conformance/results.json only, never index.html, and the responsive report is not run in CI at
// all) and no lint anywhere that scanned for hardcoded colours. These tests close both halves:
//   1. the pages must consume the one shared stylesheet's tokens (link resolves, colours are var(--token),
//      every var() is actually defined in public/styles.css);
//   2. the committed artifacts must be exactly what their generator emits from the committed evidence, so
//      a hand-edited page or a forgotten regeneration fails instead of drifting.
//
// Cost: pure Node plus a few short `node scripts/... --rollup-only` spawns - no Chrome, no server, no
// network, no writes outside a temp directory.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SHARED_STYLESHEET = join(ROOT, "public", "styles.css");
const TOKENS = readFileSync(SHARED_STYLESHEET, "utf8");
const TOKEN_NAMES = new Set([...TOKENS.matchAll(/--([a-z][a-z0-9-]*)\s*:/g)].map((m) => m[1]));

const GENERATED = [
  {
    label: "conformance rollup",
    generator: "scripts/conformance.mjs",
    artifact: "reports/conformance/index.html",
    evidence: "reports/conformance/results.json",
  },
  {
    label: "responsive matrix",
    generator: "scripts/responsive-check.mjs",
    artifact: "reports/responsive/index.html",
    evidence: "reports/responsive/results.json",
  },
];
const HAND_WRITTEN = ["search/__poc__/bench.html"];
const ALL_PAGES = [...GENERATED.map((p) => p.artifact), ...HAND_WRITTEN];

const styleOf = (html) => (html.match(/<style[^>]*>([\s\S]*?)<\/style>/i) || [, ""])[1];
const stylesheetLinks = (html) =>
  [...html.matchAll(/<link\b[^>]*>/gi)]
    .filter((m) => /rel\s*=\s*["']?stylesheet/i.test(m[0]))
    .map((m) => (m[0].match(/href\s*=\s*["']([^"']+)["']/i) || [, ""])[1]);

test("every report page links the one shared stylesheet, and each link resolves on disk", () => {
  for (const rel of ALL_PAGES) {
    const hrefs = stylesheetLinks(readFileSync(join(ROOT, rel), "utf8"));
    assert.ok(hrefs.length > 0, `${rel} must link the site stylesheet`);
    for (const href of hrefs) {
      assert.ok(!/^https?:|^\/\//.test(href), `${rel} must link the stylesheet by relative path, found ${href}`);
      const resolved = resolve(dirname(join(ROOT, rel)), href);
      assert.equal(
        resolved,
        SHARED_STYLESHEET,
        `${rel} links ${href}, which resolves to ${resolved} instead of the shared stylesheet`,
      );
      assert.ok(existsSync(resolved), `${rel} links ${href}, which does not exist`);
    }
  }
});

test("no report page carries a colour literal of its own", () => {
  // The pages in this check are the site's drift surface: they must reference tokens only. (The tokens
  // themselves naturally hold literals - they live in public/styles.css, which is not scanned here.)
  for (const rel of ALL_PAGES) {
    const css = styleOf(readFileSync(join(ROOT, rel), "utf8"));
    const literals = [...css.matchAll(/#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g)].map((m) => m[0]);
    assert.deepEqual(
      literals,
      [],
      `${rel} must express colours as var(--token); found ${literals.join(", ") || "none"}`,
    );
  }
});

test("every var(--token) a report page references is defined in public/styles.css", () => {
  for (const rel of ALL_PAGES) {
    const css = styleOf(readFileSync(join(ROOT, rel), "utf8"));
    const used = [...css.matchAll(/var\(\s*--([a-z0-9-]+)/gi)].map((m) => m[1]);
    const unknown = [...new Set(used)].filter((name) => !TOKEN_NAMES.has(name));
    assert.deepEqual(unknown, [], `${rel} references undefined tokens: ${unknown.join(", ")}`);
  }
});

test("each generator re-emits its committed page byte-for-byte from the committed evidence", () => {
  const dir = mkdtempSync(join(tmpdir(), "bj1z-rollup-"));
  try {
    for (const page of GENERATED) {
      const out = join(dir, "index.html");
      rmSync(out, { force: true });
      execFileSync(process.execPath, [join(ROOT, page.generator), "--rollup-only", "--out", dir], {
        cwd: ROOT,
        stdio: "pipe",
      });
      assert.ok(existsSync(out), `${page.generator} --rollup-only wrote no page`);
      assert.equal(
        readFileSync(out, "utf8"),
        readFileSync(join(ROOT, page.artifact), "utf8"),
        `${page.artifact} is stale or was edited by hand: refresh it with \`node ${page.generator} --rollup-only\``,
      );
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("--rollup-only fails closed on invalid evidence, writes nothing, and never refreshes the data", () => {
  const dir = mkdtempSync(join(tmpdir(), "bj1z-failclosed-"));
  const page = join(dir, "index.html");
  const invalid = [
    ["missing evidence", null],
    ["malformed evidence", "{ not json "],
    ["evidence with no runs", '{"generatedAt":"2026-01-01T00:00:00Z","runs":[]}'],
  ];
  try {
    for (const item of GENERATED) {
      const before = readFileSync(join(ROOT, item.evidence));
      for (const [label, body] of invalid) {
        const evidence = join(dir, `${label.replace(/\s+/g, "-")}.json`);
        rmSync(evidence, { force: true });
        rmSync(page, { force: true });
        if (body !== null) writeFileSync(evidence, body);
        assert.throws(
          () =>
            execFileSync(
              process.execPath,
              [join(ROOT, item.generator), "--rollup-only", "--results", evidence, "--out", dir],
              { cwd: ROOT, stdio: "pipe" },
            ),
          `${item.generator} must fail closed on ${label}`,
        );
        assert.ok(!existsSync(page), `${item.generator} wrote a page from ${label}`);
      }

      // A rollup-only run must not touch the evidence and must not invent a measurement time.
      rmSync(page, { force: true });
      execFileSync(process.execPath, [join(ROOT, item.generator), "--rollup-only", "--out", dir], {
        cwd: ROOT,
        stdio: "pipe",
      });
      assert.deepEqual(
        readFileSync(join(ROOT, item.evidence)),
        before,
        `${item.generator} --rollup-only must not rewrite ${item.evidence}`,
      );
      const generatedAt = JSON.parse(before.toString("utf8")).generatedAt;
      assert.ok(
        readFileSync(page, "utf8").includes(generatedAt),
        `${item.artifact} must show the evidence's own generatedAt (${generatedAt}), not a new timestamp`,
      );
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
