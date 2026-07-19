#!/usr/bin/env node
// Route manifest — the machine-readable, append-only record of every PUBLISHED demo identity.
//
// "Published" = live to users: a real route/URL + a catalogue entry. For this repo that is a `built`
// demo (has a page at models/<slug>/index.html) or a `blocked` entry (honestly recorded, no route yet
// but still under the compatibility contract — it must stay recorded, never silently deleted).
// `pending` placeholders are NOT published (no route) and are excluded.
//
// The manifest is the source of truth for identities the durable-demo compatibility contract protects
// (see CLAUDE.md invariant 13). scripts/check-routes.mjs diffs the previously published manifest
// (git show origin/main:models.json, or the committed .route-manifest.baseline.json fallback) against
// the working tree and fails on any destructive change.
//
// Usage:
//   node scripts/route-manifest.mjs             # human summary of the current manifest
//   node scripts/route-manifest.mjs --json      # print the normalized manifest array (stdout)
//   node scripts/route-manifest.mjs --write-baseline   # (re)write .route-manifest.baseline.json

import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

// Published statuses — order is significant for reporting only.
export const PUBLISHED_STATUSES = ["built", "blocked"];

// Normalize a raw catalogue (parsed models.json) into the published manifest array. Deterministic +
// stable-sorted (by id) so a diff is meaningful and the committed baseline has no spurious churn.
export function buildManifest(catalogue) {
  const models = Array.isArray(catalogue) ? catalogue : (catalogue?.models ?? []);
  return models
    .filter((m) => PUBLISHED_STATUSES.includes(m.status))
    .map((m) => ({
      id: m.slug,
      route: `models/${m.slug}/`,
      identity: { hfId: m.hfId ?? null, task: m.task ?? null },
      status: m.status,
      aliases: Array.isArray(m.aliases) ? m.aliases : [],
    }))
    .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

async function readCatalogue(url) {
  return JSON.parse(await readFile(url, "utf8"));
}

async function main() {
  const args = process.argv.slice(2);
  const catUrl = new URL("../models.json", import.meta.url);
  const manifest = buildManifest(await readCatalogue(catUrl));

  if (args.includes("--json")) {
    process.stdout.write(JSON.stringify(manifest, null, 2) + "\n");
    return;
  }

  if (args.includes("--write-baseline")) {
    const out = fileURLToPath(new URL("../.route-manifest.baseline.json", import.meta.url));
    await writeFile(out, JSON.stringify(manifest, null, 2) + "\n");
    console.log(`wrote ${manifest.length} published entries to .route-manifest.baseline.json`);
    return;
  }

  const built = manifest.filter((m) => m.status === "built").length;
  const blocked = manifest.filter((m) => m.status === "blocked").length;
  console.log("=== ROUTE MANIFEST (published demos — durable compatibility contract) ===");
  console.log(`published: ${manifest.length}  (built: ${built}, blocked: ${blocked})`);
  console.log("Run with --json to print the manifest, --write-baseline to refresh the snapshot.");
}

// Only run the CLI when executed directly (so check-routes.mjs can import buildManifest).
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
}
