#!/usr/bin/env node
// scripts/check-critique-coverage.mjs — SOFT report (not a hard gate).
//
// Invariant 14 wants every BUILT demo to carry a per-demo critique (_questions.json,
// versioned, with guidanceConsulted). The three hard gates (routes / conformance /
// lineage) do not fail on a missing critique, so this report surfaces the burn-down:
// which built demos still lack a critique, as an honest coverage denominator.
//
// Usage:
//   node scripts/check-critique-coverage.mjs            # summary + list of gaps
//   node scripts/check-critique-coverage.mjs --json     # machine-readable
//   node scripts/check-critique-coverage.mjs --strict   # exit 1 if any gap (opt-in)
//
// Exit code is 0 by default (soft) unless --strict is passed.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const asJson = args.has("--json");
const strict = args.has("--strict");

const raw = JSON.parse(fs.readFileSync(path.join(root, "models.json"), "utf8"));
const models = Array.isArray(raw) ? raw : raw.models;

const built = models.filter((m) => m.status === "built");
const slugOf = (m) => m.slug || (m.demo && m.demo.slug) || null;

const withCritique = [];
const missing = [];
for (const m of built) {
  const slug = slugOf(m);
  if (!slug) continue;
  const f = path.join(root, "models", slug, "_questions.json");
  if (fs.existsSync(f)) {
    // Lightly validate it parses and carries the required guidanceConsulted key.
    try {
      const q = JSON.parse(fs.readFileSync(f, "utf8"));
      const okGuidance = Array.isArray(q.guidanceConsulted) &&
        q.guidanceConsulted.length > 0;
      (okGuidance ? withCritique : missing).push(
        okGuidance ? slug : `${slug} (empty guidanceConsulted)`,
      );
    } catch {
      missing.push(`${slug} (unparseable _questions.json)`);
    }
  } else {
    missing.push(slug);
  }
}

const total = built.length;
const covered = withCritique.length;
const pct = total ? Math.round((covered / total) * 1000) / 10 : 0;

if (asJson) {
  console.log(JSON.stringify(
    { built: total, critiqueAuthored: covered, coveragePct: pct, missing },
    null,
    2,
  ));
} else {
  console.log(`critique coverage: ${covered}/${total} built demos (${pct}%)`);
  if (missing.length) {
    console.log(`\n${missing.length} built demo(s) MISSING a critique (_questions.json):`);
    for (const s of missing.sort()) console.log(`  - ${s}`);
    console.log(
      `\nSOFT report — invariant 14 wants a critique per built demo; not a hard gate.`,
    );
  } else {
    console.log(`\nPASS — every built demo has a critique with guidanceConsulted.`);
  }
}

process.exit(strict && missing.length ? 1 : 0);
