// Deterministic tests for the model-download route inventory (Task 2b · Phase 1).
import { test } from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";

const inv = JSON.parse(readFileSync("download-routes.json", "utf8"));
const FAMILY = new Set([
  "transformers-pipeline",
  "transformers-from_pretrained",
  "transformers-wrapped",
  "transformers-resumable-prefetch",
  "webllm",
  "mediapipe",
  "raw-ort",
  "browser-builtin",
  "non-applicable",
]);
const RESUME = new Set([
  "resumable",
  "restart-only",
  "runtime-owned",
  "cached-only",
  "non-applicable",
]);
const STATUS = new Set([
  "pending-adoption",
  "adopted",
  "honestly-limited",
  "non-applicable",
  "blocked",
]);

test("inventory parses, totals are self-consistent", () => {
  assert.ok(inv.routes.length > 0);
  assert.equal(inv.totals.routes, inv.routes.length);
  const sum = Object.values(inv.totals.byFamily).reduce((a, b) => a + b, 0);
  assert.equal(sum, inv.routes.length);
});

test("every route has a known family / resume / status and no unknowns", () => {
  for (const r of inv.routes) {
    assert.ok(FAMILY.has(r.family), `${r.slug} family=${r.family}`);
    assert.ok(RESUME.has(r.resume), `${r.slug} resume=${r.resume}`);
    assert.ok(STATUS.has(r.status), `${r.slug} status=${r.status}`);
    assert.notEqual(r.family, "unknown");
  }
});

test("resume claims are consistent with byte-control (no false resumable)", () => {
  for (const r of inv.routes) {
    if (r.resume === "resumable") {
      // genuine per-file resume only when the SITE controls the byte transfer
      assert.equal(
        r.byteControl,
        "site-controlled",
        `${r.slug} claims resumable but byteControl=${r.byteControl}`,
      );
    }
    if (r.byteControl === "runtime-owned") {
      assert.ok(
        ["runtime-owned", "cached-only"].includes(r.resume),
        `${r.slug} runtime-owned but resume=${r.resume}`,
      );
    }
  }
});

test("the inventory route set equals the built demo routes on disk", () => {
  const built = readdirSync("models").filter((s) => {
    try {
      return statSync(join("models", s)).isDirectory() &&
        existsSync(join("models", s, "index.html"));
    } catch {
      return false;
    }
  }).sort();
  const listed = inv.routes.map((r) => r.slug).sort();
  assert.deepEqual(listed, built);
});

test("no duplicate route entries", () => {
  const seen = new Set();
  for (const r of inv.routes) {
    assert.ok(!seen.has(r.slug), `duplicate ${r.slug}`);
    seen.add(r.slug);
  }
});

test("the fail-closed inventory gate passes on the current tree", () => {
  execSync("node scripts/check-download-inventory.mjs", {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  assert.ok(true);
});
