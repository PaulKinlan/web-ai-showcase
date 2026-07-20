#!/usr/bin/env node
// Task 2b · Phase 5 — fail-closed download-component ADOPTION gate.
//
// Every route that transfers model assets MUST route its download progress through the shared,
// component-rendering loader — createModelLoader (lib/model-loader.js) or createResumableLoader
// (lib/resumable-loader.mjs). Both render <model-download-status> and feed it via the runtime adapters, so
// adopting the shared loader IS the adoption. A downloading route that rolls its own loader/progress UI is
// a BYPASS: it re-introduces the last-callback-wins problem and escapes the honest multi-file UX. This gate
// FAILS on any un-allowlisted bypass, so a new downloading demo can't land without adopting the component.
//
// (The exact classification lives in download-routes.json and is kept honest by check-download-inventory.mjs,
// which re-derives it from source and fails on drift. This gate enforces the POLICY on that classification.)
//
// Allowlist: scripts/download-adoption-allowlist.json — [{ "slug": "...", "reason": "..." }] for the rare
// route that genuinely cannot adopt (e.g. a browser-built-in-AI route where the browser owns the download).
//
// Usage: node scripts/check-download-adoption.mjs
import { existsSync, readFileSync } from "node:fs";

if (!existsSync("download-routes.json")) {
  console.error(
    "FAIL — download-routes.json is missing. Run: node scripts/route-download-inventory.mjs",
  );
  process.exit(1);
}
const inv = JSON.parse(readFileSync("download-routes.json", "utf8"));
let allow = [];
try {
  allow = JSON.parse(readFileSync("scripts/download-adoption-allowlist.json", "utf8"));
} catch { /* no allowlist → none allowed */ }
const allowBy = new Map(allow.map((a) => [a.slug, a.reason]));

const problems = [];
const downloading = inv.routes.filter((r) => r.status !== "non-applicable");
const bypasses = [];
for (const r of downloading) {
  const isBypass = r.adoption === "bypass" || r.status === "blocked";
  if (!isBypass) {
    // adopted route: sanity-check the fields agree
    if (r.adoption !== "central-loader" && r.adoption !== "resumable-loader") {
      problems.push(`INCONSISTENT: ${r.slug} status=adopted but adoption='${r.adoption}'`);
    }
    continue;
  }
  bypasses.push(r.slug);
  if (allowBy.has(r.slug)) continue; // documented exception
  problems.push(
    `BYPASS: ${r.slug} (family=${r.family}) downloads model assets but does NOT call the shared ` +
      `createModelLoader / createResumableLoader — it bypasses <model-download-status>. Route it through ` +
      `the shared loader, or add a documented exception to scripts/download-adoption-allowlist.json.`,
  );
}
// A stale allowlist entry (route no longer a bypass) should be cleaned up.
for (const a of allow) {
  const r = inv.routes.find((x) => x.slug === a.slug);
  if (!r) {
    problems.push(`STALE ALLOWLIST: ${a.slug} is allow-listed but not in download-routes.json`);
  } else if (r.adoption !== "bypass") {
    problems.push(
      `STALE ALLOWLIST: ${a.slug} is allow-listed but now adopts (${r.adoption}) — remove the exception`,
    );
  }
}

const t = inv.totals || {};
if (problems.length) {
  console.error(
    `\ndownload-adoption: ${downloading.length} downloading routes · ${bypasses.length} bypass(es)`,
  );
  for (const p of problems.slice(0, 40)) console.error("  ✗ " + p);
  console.error(
    `\nFAIL — ${problems.length} adoption problem(s). Every downloading route must adopt <model-download-status>.`,
  );
  process.exit(1);
}

console.error(
  `download-adoption: ${downloading.length} downloading routes — all adopt the shared component-rendering ` +
    `loader (${
      JSON.stringify(t.byAdoption)
    }); ${allow.length} documented exception(s); 0 un-allowlisted bypasses.`,
);
console.error("PASS — no downloading route bypasses <model-download-status>.");
process.exit(0);
