#!/usr/bin/env node
// Deterministically replace provenance rows for the exact 20 MagicTouch acceptance screenshots.
// Run only after the capture and offline validator have produced a complete, hash-bound summary.

import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { repoRoot } from "./browser.mjs";
import {
  EVIDENCE_DEVICES,
  EVIDENCE_ROUTES,
  EVIDENCE_THEMES,
  EXPECTED_SCREENSHOTS,
} from "./interactive-segmenter-evidence.mjs";

const ledgerPath = join(repoRoot, "image-provenance", "ledger.json");
const evidencePath = join(
  repoRoot,
  "models",
  "interactive-segmenter",
  "evidence",
  "acceptance.json",
);
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const expectedKeys = new Set(
  EVIDENCE_DEVICES.flatMap((device) =>
    EVIDENCE_ROUTES.flatMap((route) =>
      EVIDENCE_THEMES.map((theme) => `${route}/${device}/${theme}`)
    )
  ),
);

const evidence = JSON.parse(readFileSync(evidencePath, "utf8"));
if (
  typeof evidence.generatedAt !== "string" ||
  !Number.isFinite(Date.parse(evidence.generatedAt)) ||
  new Date(Date.parse(evidence.generatedAt)).toISOString() !== evidence.generatedAt
) throw new Error("MagicTouch evidence generatedAt must be a canonical immutable timestamp");
if (
  evidence.status !== "completed" || !Array.isArray(evidence.screenshots) ||
  evidence.screenshots.length !== EXPECTED_SCREENSHOTS
) {
  throw new Error("MagicTouch screenshot provenance requires completed 20/20 evidence");
}
const screenshotKeys = new Set(
  evidence.screenshots.map((shot) => `${shot.route}/${shot.device}/${shot.theme}`),
);
if (
  screenshotKeys.size !== expectedKeys.size ||
  [...expectedKeys].some((key) => !screenshotKeys.has(key))
) throw new Error("MagicTouch screenshot provenance denominator/key set mismatch");

const rows = evidence.screenshots.map((shot) => {
  const relativePath = `models/interactive-segmenter/${shot.path}`;
  const bytes = readFileSync(join(repoRoot, relativePath));
  const hash = sha256(bytes);
  if (hash !== shot.sha256 || bytes.length !== shot.bytes) {
    throw new Error(`MagicTouch screenshot bytes differ from evidence: ${relativePath}`);
  }
  return { relativePath, hash, bytes: bytes.length };
});

const ledger = JSON.parse(readFileSync(ledgerPath, "utf8"));
const targetPaths = new Set(rows.map((row) => row.relativePath));
ledger.entries = ledger.entries.map((entry) => {
  const paths = entry.paths.filter((path) => !targetPaths.has(path));
  return { ...entry, paths, fileCount: paths.length };
}).filter((entry) => entry.paths.length > 0);

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
    depictsPeople: true,
    depictsIdentifiablePerson: true,
    rightsCleared: true,
    provenance: {
      kind: "licensed",
      sourceAsset: "interactive-segmenter-rights-cleared-gallery",
      source: "Web AI Showcase MagicTouch demo using media/manifest.json assets",
      sourceUrl: "https://github.com/PaulKinlan/web-ai-showcase",
      creator: "Paul Kinlan; embedded portrait by Jones, W. A. (Public Domain)",
      license: "Public-Domain (portrait); other bundled licenses recorded in media/manifest.json",
      licenseName: "Public domain portrait; all other samples rights-cleared",
      attribution:
        "Jones, W. A. — State Library of Queensland portrait — Public domain; remaining sample credits rendered visibly by public/image-credit.js",
    },
    evidence:
      "Hash-chained chrome-devtools-mcp route/device/theme screenshot after real MagicTouch inference. The only identifiable person is the Public Domain portrait-person asset; all other visible thumbnails are rights-cleared media/manifest.json assets with visible credits.",
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
ledger.generated = evidence.generatedAt.slice(0, 10);
ledger.totals = {
  uniqueImages: ledger.entries.length,
  files: ledger.entries.reduce((sum, entry) => sum + entry.paths.length, 0),
  byArchetype,
  depictsPeople: ledger.entries.filter((entry) => entry.depictsPeople).length,
  depictsIdentifiablePerson:
    ledger.entries.filter((entry) => entry.depictsIdentifiablePerson).length,
  entries: ledger.entries.length,
};
writeFileSync(ledgerPath, JSON.stringify(ledger, null, 2) + "\n");
console.log(
  `registered ${rows.length}/${EXPECTED_SCREENSHOTS} MagicTouch screenshot provenance rows`,
);
