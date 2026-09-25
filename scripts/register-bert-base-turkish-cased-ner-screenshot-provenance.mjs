#!/usr/bin/env node
// Deterministically replace provenance rows for the Turkish NER acceptance screenshots
// (web-ai-showcase-62m.2). Run only after
// `node scripts/validate-bert-base-turkish-cased-ner.mjs --write-run` has produced a complete,
// passing run record — the ledger is keyed by content hash, so refreshed screenshots fail the
// fail-closed provenance gate until their new bytes are registered here.
//
// House pattern: scripts/register-interactive-segmenter-screenshot-provenance.mjs.

import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { repoRoot } from "./browser.mjs";

const SLUG = "bert-base-turkish-cased-ner";
const ledgerPath = join(repoRoot, "image-provenance", "ledger.json");
const recordPath = join(repoRoot, "models", SLUG, "acceptance-run.json");
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");

const record = JSON.parse(readFileSync(recordPath, "utf8"));
if (record.exitCode !== 0) {
  throw new Error(`acceptance run record did not pass: exitCode=${record.exitCode}`);
}
const shots = Array.isArray(record.screenshots) ? record.screenshots : [];
if (shots.length === 0) throw new Error("acceptance run record has no screenshots to register");

const rows = shots.map((shot) => {
  const relativePath = String(shot.path ?? "");
  if (!relativePath.startsWith(`reports/acceptance/${SLUG}/`)) {
    throw new Error(`screenshot path escapes the family evidence dir: ${relativePath}`);
  }
  const abs = join(repoRoot, relativePath);
  if (!existsSync(abs)) throw new Error(`screenshot missing on disk: ${relativePath}`);
  const bytes = readFileSync(abs);
  return { relativePath, hash: sha256(bytes), bytes: bytes.length };
});
if (new Set(rows.map((r) => r.relativePath)).size !== rows.length) {
  throw new Error("duplicate screenshot path in the run record");
}

const ledger = JSON.parse(readFileSync(ledgerPath, "utf8"));
const targetPaths = new Set(rows.map((r) => r.relativePath));
ledger.entries = ledger.entries
  .map((entry) => {
    const paths = entry.paths.filter((path) => !targetPaths.has(path));
    return { ...entry, paths, fileCount: paths.length };
  })
  .filter((entry) => entry.paths.length > 0);

for (const row of rows) {
  const existing = ledger.entries.find((entry) => entry.hash === row.hash);
  if (existing) {
    if (!existing.paths.includes(row.relativePath)) existing.paths.push(row.relativePath);
    existing.paths.sort();
    existing.fileCount = existing.paths.length;
    continue;
  }
  ledger.entries.push({
    hash: row.hash,
    bytes: row.bytes,
    fileCount: 1,
    archetype: "qa-screenshot",
    depictsPeople: false,
    depictsIdentifiablePerson: false,
    rightsCleared: true,
    provenance: {
      kind: "first-party",
      source: "web-ai-showcase Turkish NER acceptance (generated)",
      sourceUrl: "",
      creator: "web-ai-showcase",
      license: "first-party (this project)",
      attribution: "web-ai-showcase — first-party QA screenshot",
    },
    evidence:
      `route-complete populated full-page-after-scroll of first-party demo UI; no depicted person; ` +
      `bound to implementation ${record.commit}`,
    paths: [row.relativePath],
  });
}
ledger.entries.sort((a, b) => a.hash.localeCompare(b.hash));
const byArchetype = Object.fromEntries(
  [...new Set(ledger.entries.map((entry) => entry.archetype))].sort().map((archetype) => [
    archetype,
    ledger.entries.filter((entry) => entry.archetype === archetype).length,
  ]),
);
if (record.runAt) ledger.generated = String(record.runAt).slice(0, 10);
ledger.totals = {
  uniqueImages: ledger.entries.length,
  files: ledger.entries.reduce((sum, entry) => sum + entry.paths.length, 0),
  byArchetype,
  depictsPeople: ledger.entries.filter((entry) => entry.depictsPeople).length,
  depictsIdentifiablePerson: ledger.entries.filter((entry) => entry.depictsIdentifiablePerson)
    .length,
  entries: ledger.entries.length,
};
writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + "\n");
console.log(
  `registered ${rows.length} Turkish NER screenshot provenance rows for commit ${
    String(record.commit).slice(0, 7)
  }`,
);
