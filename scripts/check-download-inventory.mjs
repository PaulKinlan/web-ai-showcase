#!/usr/bin/env node
// Task 2b · Phase 1 gate — fail-closed model-download route inventory.
//
// Every built demo route MUST have a classification in download-routes.json, and the committed inventory
// MUST match a fresh classification of the current source. So: a NEW downloading demo, or an existing one
// that switches runtime/loader (changing its download surface), makes the committed inventory stale and
// FAILS this gate until it's regenerated (`node scripts/route-download-inventory.mjs`). No route may be
// "unknown". This is the mechanical guarantee that the site-wide download component's coverage denominator
// stays exact as the catalogue grows.
//
// Usage: node scripts/check-download-inventory.mjs
import { execSync } from "node:child_process";
import { existsSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { join } from "node:path";

const problems = [];

if (!existsSync("download-routes.json")) {
  console.error(
    "FAIL — download-routes.json is missing. Run: node scripts/route-download-inventory.mjs",
  );
  process.exit(1);
}
const committed = JSON.parse(readFileSync("download-routes.json", "utf8"));
const bySlug = new Map();
for (const r of committed.routes) {
  if (bySlug.has(r.slug)) problems.push(`DUPLICATE route entry: ${r.slug}`);
  bySlug.set(r.slug, r);
}

// 1) the inventory's route set must exactly equal the built demo routes on disk.
const builtRoutes = readdirSync("models").filter((s) => {
  try {
    return statSync(join("models", s)).isDirectory() && existsSync(join("models", s, "index.html"));
  } catch {
    return false;
  }
});
const builtSet = new Set(builtRoutes);
for (const s of builtRoutes) {
  if (!bySlug.has(s)) {
    problems.push(
      `UNCLASSIFIED ROUTE: models/${s}/ is a built demo with no entry in download-routes.json`,
    );
  }
}
for (const r of committed.routes) {
  if (!builtSet.has(r.slug)) {
    problems.push(
      `ORPHAN ENTRY: ${r.slug} is in the inventory but models/${r.slug}/index.html does not exist`,
    );
  }
}

// 2) no route may be unknown; every field must be from the known vocabulary.
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
for (const r of committed.routes) {
  if (r.family === "unknown" || r.status === "unknown" || r.resume === "unknown") {
    problems.push(
      `UNKNOWN CLASSIFICATION: ${r.slug} (family=${r.family}, resume=${r.resume}, status=${r.status}) — classify it`,
    );
  } else {
    if (!FAMILY.has(r.family)) problems.push(`BAD FAMILY: ${r.slug} → '${r.family}'`);
    if (!RESUME.has(r.resume)) problems.push(`BAD RESUME: ${r.slug} → '${r.resume}'`);
    if (!STATUS.has(r.status)) problems.push(`BAD STATUS: ${r.slug} → '${r.status}'`);
  }
}

// 3) totals must be self-consistent.
if (committed.totals?.routes !== committed.routes.length) {
  problems.push(
    `TOTALS DRIFT: totals.routes=${committed.totals?.routes} but ${committed.routes.length} routes listed`,
  );
}

// 4) the committed inventory must match a FRESH classification of the current source (no silent drift).
const tmp = ".download-routes.check.json";
try {
  execSync(`node scripts/route-download-inventory.mjs --out ${tmp}`, {
    stdio: ["ignore", "ignore", "ignore"],
  });
  const fresh = JSON.parse(readFileSync(tmp, "utf8"));
  const freshBySlug = new Map(fresh.routes.map((r) => [r.slug, r]));
  for (const r of committed.routes) {
    const f = freshBySlug.get(r.slug);
    if (!f) continue; // covered by the orphan check
    for (const k of ["family", "byteControl", "resume", "multiModel", "adoption", "status"]) {
      if (String(r[k]) !== String(f[k])) {
        problems.push(
          `STALE CLASSIFICATION: ${r.slug}.${k} committed='${
            r[k]
          }' but current source classifies as '${f[k]}' — regenerate download-routes.json`,
        );
      }
    }
  }
} finally {
  try {
    rmSync(tmp);
  } catch { /* ignore */ }
}

const t = committed.totals || {};
if (problems.length) {
  console.error(
    `\ndownload-inventory: ${committed.routes.length} routes / ${builtRoutes.length} built`,
  );
  for (const p of problems.slice(0, 40)) console.error("  ✗ " + p);
  if (problems.length > 40) console.error(`  … and ${problems.length - 40} more`);
  console.error(
    `\nFAIL — ${problems.length} inventory problem(s). Regenerate: node scripts/route-download-inventory.mjs`,
  );
  process.exit(1);
}

console.error(
  `download-inventory: ${committed.routes.length} routes · ${t.downloadingRoutes} download · ` +
    `families ${JSON.stringify(t.byFamily)} · resume ${JSON.stringify(t.byResume)}`,
);
console.error(
  "PASS — every built route is classified; inventory matches current source; no unknowns.",
);
process.exit(0);
