#!/usr/bin/env node
// Derive an immutable conformance suite for every built model from real models.json metadata + the
// per-task templates in conformance-lib.mjs, and (optionally) seed the `support` record on each built
// models.json entry.
//
// Genuine, not fake: every assertion is DERIVED from the model's real task / model-id / backend /
// quantisation and the demo's real source. Suites are written to models/<slug>/conformance.json with
// a computed suiteHash. This is IMMUTABLE-safe: it will NOT overwrite an existing suite's assertions
// unless --force is passed, so a committed suite can never be silently regenerated to go green. New
// built models get a fresh suite; existing suites are left untouched (add assertions by hand + rehash
// with --rehash, never weaken).
//
// Usage:
//   node scripts/gen-conformance.mjs                 # write suites for built models that lack one
//   node scripts/gen-conformance.mjs --all           # (re)write only MISSING suites + report
//   node scripts/gen-conformance.mjs --seed-support  # add support:{desktop,mobile:"untested"} to built entries lacking it
//   node scripts/gen-conformance.mjs --rehash <slug> # recompute suiteHash after a hand-added assertion (does not weaken)
//   node scripts/gen-conformance.mjs --force         # DANGER: overwrite existing suites (only for a fresh bootstrap)

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  builtModels,
  computeSuiteHash,
  deriveSuite,
  loadCatalogue,
  repoRoot,
} from "./conformance-lib.mjs";

const args = process.argv.slice(2);
const force = args.includes("--force");
const seedSupport = args.includes("--seed-support");
const rehashIdx = args.indexOf("--rehash");
const generatedAt = process.env.CONFORMANCE_GENERATED_AT || "2026-07-19T00:00:00Z";

function suitePath(slug) {
  return `${repoRoot}models/${slug}/conformance.json`;
}

if (rehashIdx !== -1) {
  const slug = args[rehashIdx + 1];
  const p = suitePath(slug);
  const suite = JSON.parse(readFileSync(p, "utf8"));
  suite.suiteHash = computeSuiteHash(suite.assertions);
  writeFileSync(p, JSON.stringify(suite, null, 2) + "\n");
  console.log(`rehashed ${slug}: ${suite.suiteHash}`);
  process.exit(0);
}

const built = builtModels();
let written = 0, skipped = 0;
for (const m of built) {
  const p = suitePath(m.slug);
  if (existsSync(p) && !force) {
    skipped++;
    continue;
  }
  const suite = deriveSuite(m, { generatedAt, author: "derive-conformance" });
  writeFileSync(p, JSON.stringify(suite, null, 2) + "\n");
  written++;
}
console.log(
  `conformance suites: ${written} written, ${skipped} already present (immutable — left as-is).`,
);

if (seedSupport) {
  // Add a support record to every built entry that lacks one, defaulting to "untested" (honest).
  const catPath = `${repoRoot}models.json`;
  const raw = JSON.parse(readFileSync(catPath, "utf8"));
  const arr = Array.isArray(raw) ? raw : raw.models;
  let seeded = 0;
  for (const m of arr) {
    if (m.status === "built" && !m.support) {
      m.support = { desktop: "untested", mobile: "untested" };
      seeded++;
    }
  }
  writeFileSync(catPath, JSON.stringify(raw, null, 2) + "\n");
  console.log(`support records seeded on ${seeded} built entries (default "untested").`);
}
