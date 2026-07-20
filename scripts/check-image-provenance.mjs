#!/usr/bin/env node
// Fail-closed image-provenance gate. Every tracked raster image on the site MUST map, BY CONTENT HASH,
// to an entry in image-provenance/ledger.json, and every image that depicts an identifiable person MUST
// be rights-cleared with a source URL, creator, and license. Because it re-hashes every file, ANY new or
// modified image whose bytes aren't in the ledger fails the gate — you cannot ship an image without
// recording its provenance first. This is the mechanical enforcement of "no unverified faces ship".
//
// Usage: node scripts/check-image-provenance.mjs [--json]
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { execSync } from "node:child_process";

const RASTER = /\.(png|jpe?g|gif|webp|avif|bmp|tiff?)$/i;
const sha256 = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");

const files = execSync("git ls-files", { encoding: "utf8" })
  .split("\n").filter((f) => f && RASTER.test(f));

let ledger;
try {
  ledger = JSON.parse(readFileSync("image-provenance/ledger.json", "utf8"));
} catch (e) {
  console.error("FAIL — cannot read image-provenance/ledger.json:", e.message);
  process.exit(1);
}

const entryByHash = new Map();
const pathToHash = new Map();
for (const e of ledger.entries) {
  entryByHash.set(e.hash, e);
  for (const p of e.paths) pathToHash.set(p, e.hash);
}

const problems = [];
// 1) every tracked raster must be in the ledger by content hash, listed under its own path.
let peopleFiles = 0, identifiableFiles = 0;
for (const f of files) {
  const h = sha256(f);
  const e = entryByHash.get(h);
  if (!e) {
    problems.push(
      `UNLEDGERED (unknown content hash): ${f} — every image must have a provenance entry; add it to image-provenance/ledger.json`,
    );
    continue;
  }
  if (!e.paths.includes(f)) {
    problems.push(
      `PATH DRIFT: ${f} has the bytes of ledger entry ${
        h.slice(0, 12)
      } but that path is not listed in its paths[]`,
    );
  }
  if (e.depictsPeople) peopleFiles++;
  if (e.depictsIdentifiablePerson) identifiableFiles++;
}

// 2) every ledger path must exist on disk (no stale/orphan ledger rows).
const onDisk = new Set(files);
for (const e of ledger.entries) {
  for (const p of e.paths) {
    if (!onDisk.has(p)) {
      problems.push(
        `ORPHAN LEDGER PATH: ${p} (entry ${e.hash.slice(0, 12)}) is not a tracked file`,
      );
    }
  }
}

// 3) fail-closed people/rights policy: any identifiable person must be cleared + licensed + attributable.
for (const e of ledger.entries) {
  if (!e.depictsIdentifiablePerson) continue;
  const pv = e.provenance || {};
  if (!e.rightsCleared) {
    problems.push(
      `UNCLEARED PERSON: ${e.paths[0]} depicts an identifiable person but rightsCleared=false`,
    );
  }
  if (pv.kind !== "licensed") {
    problems.push(
      `UNLICENSED PERSON: ${
        e.paths[0]
      } depicts an identifiable person but provenance.kind='${pv.kind}' (must be 'licensed')`,
    );
  }
  if (!pv.license) {
    problems.push(`NO LICENSE: ${e.paths[0]} depicts an identifiable person but has no license`);
  }
  // must be traceable to a source: a Commons URL, or the documented composite that references sub-assets.
  const traceable = (pv.sourceUrl && pv.sourceUrl.length > 0) || pv.sourceAsset === "faces-crowd";
  if (!traceable) {
    problems.push(
      `NO SOURCE URL: ${
        e.paths[0]
      } depicts an identifiable person but has no sourceUrl to verify provenance`,
    );
  }
  if (!pv.attribution) {
    problems.push(
      `NO ATTRIBUTION: ${e.paths[0]} depicts an identifiable person but has no attribution string`,
    );
  }
}

// 4) schema sanity: hashes well-formed, archetypes known.
const ARCH = new Set([
  "media-library",
  "licensed-derived",
  "media-derived",
  "procedural",
  "qa-screenshot",
]);
for (const e of ledger.entries) {
  if (!/^[0-9a-f]{64}$/.test(e.hash)) problems.push(`BAD HASH: ${e.paths?.[0]} has malformed hash`);
  if (!ARCH.has(e.archetype)) {
    problems.push(`BAD ARCHETYPE: ${e.paths?.[0]} archetype='${e.archetype}'`);
  }
}

const summary = {
  trackedRasterFiles: files.length,
  ledgerEntries: ledger.entries.length,
  peopleFiles,
  identifiableFiles,
  byArchetype: ledger.entries.reduce(
    (m, e) => ((m[e.archetype] = (m[e.archetype] || 0) + 1), m),
    {},
  ),
  problems: problems.length,
};

if (process.argv.includes("--json")) {
  console.log(JSON.stringify({ ...summary, problemList: problems }, null, 2));
}

if (problems.length) {
  console.error(`\nimage-provenance: ${files.length} files / ${ledger.entries.length} entries`);
  for (const p of problems.slice(0, 40)) console.error("  ✗ " + p);
  if (problems.length > 40) console.error(`  … and ${problems.length - 40} more`);
  console.error(
    `\nFAIL — ${problems.length} provenance problem(s). No image ships without a rights-cleared ledger entry.`,
  );
  process.exit(1);
}

console.error(
  `image-provenance: ${files.length} raster files → ${ledger.entries.length} ledgered content hashes · ` +
    `${peopleFiles} depict people (${identifiableFiles} identifiable, all rights-cleared) · ` +
    `${JSON.stringify(summary.byArchetype)}`,
);
console.error(
  "PASS — every image has a provenance entry; every identifiable person is licensed + attributable; no unverified faces ship.",
);
process.exit(0);
