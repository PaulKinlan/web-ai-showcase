#!/usr/bin/env node
// Fail-closed audio-provenance gate. Every audio file on the site (.wav .mp3 .ogg .m4a .flac) MUST map,
// BY CONTENT HASH, to an entry in audio-provenance/ledger.json. A reconciled entry must carry a licence,
// a traceable source and an attribution; an unreconciled entry must be declared legacy AND its content
// hash must be frozen in the ledger's legacy baseline. Because it re-hashes every file, ANY new, changed,
// or byte-different-copy audio file whose bytes aren't in the ledger fails the gate — you cannot ship
// audio without recording its provenance first. This is the mechanical enforcement of "no unverified
// source ships", the audio counterpart to scripts/check-image-provenance.mjs, and the station that was
// missing when ten byte-identical ted.wav copies and seventeen jfk.wav copies shipped through every gate
// unnoticed for six weeks.
//
// SCOPE mirrors the image gate exactly: `git ls-files` filtered by audio extension, i.e. every
// git-TRACKED file under the site tree. Gitignored trees (node_modules/) are out of scope by
// construction, and there is deliberately no filename exclusion — a committed test fixture is still
// bundled bytes and is in scope, exactly as a fixture raster is in scope for the image gate.
//
// The legacy baseline (check-portfolio-acceptance.mjs's "294 legacy families, never retroactively
// passed") is declared in one place: ledger.legacyBaseline. It is an explicit frozen set of content
// HASHES, never a filename pattern, so a new asset cannot be excused by name — declaring it legacy fails
// with LEGACY NOT IN BASELINE, and adding a further copy of a frozen asset fails with PATH DRIFT.
//
// Usage: node scripts/check-audio-provenance.mjs [--json]
// Test hooks (fixtures only, never set in CI): --root <dir> --ledger <path>
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";

const AUDIO = /\.(wav|mp3|ogg|m4a|flac)$/i;
const ARCHETYPES = new Set(["licensed", "first-party", "legacy-unreconciled"]);

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const ROOT = flag("--root", new URL("..", import.meta.url).pathname);
const LEDGER = flag("--ledger", join(ROOT, "audio-provenance/ledger.json"));

const sha256 = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");

let files;
try {
  files = execFileSync("git", ["ls-files"], {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  })
    .split("\n")
    .filter((f) => f && AUDIO.test(f));
} catch (e) {
  console.error(`FAIL — cannot enumerate tracked audio files at ${ROOT}:`, e.message);
  process.exit(1);
}

let ledger;
try {
  ledger = JSON.parse(readFileSync(LEDGER, "utf8"));
} catch (e) {
  console.error(`FAIL — cannot read ${LEDGER}:`, e.message);
  process.exit(1);
}
for (const k of ["policy", "legacyBaseline", "totals", "entries"]) {
  if (!(k in ledger)) {
    console.error(`FAIL — ${LEDGER} has no "${k}"`);
    process.exit(1);
  }
}

const baseline = ledger.legacyBaseline;
const baselineHashes = new Set(Array.isArray(baseline.hashes) ? baseline.hashes : []);
const entryByHash = new Map();
for (const e of ledger.entries) entryByHash.set(e.hash, e);

const problems = [];

// 1) every tracked audio file must be in the ledger by content hash, listed under its own path.
let reconciledFiles = 0;
let legacyFiles = 0;
for (const f of files) {
  const h = sha256(join(ROOT, f));
  const e = entryByHash.get(h);
  if (!e) {
    problems.push(
      `UNLEDGERED AUDIO (unknown content hash): ${f} — every bundled audio file must have a provenance entry; add it to audio-provenance/ledger.json`,
    );
    continue;
  }
  if (!Array.isArray(e.paths) || !e.paths.includes(f)) {
    problems.push(
      `PATH DRIFT: ${f} has the bytes of ledger entry ${
        h.slice(0, 12)
      } but that path is not listed in its paths[] — a new copy of an existing clip is still a new asset`,
    );
  }
  if (e.rightsCleared) reconciledFiles++;
  else legacyFiles++;
}

// 2) every ledger path must be a tracked file (no stale/orphan rows).
const tracked = new Set(files);
for (const e of ledger.entries) {
  for (const p of (Array.isArray(e.paths) && e.paths) || []) {
    if (!tracked.has(p)) {
      problems.push(
        `ORPHAN LEDGER PATH: ${p} (entry ${String(e.hash).slice(0, 12)}) is not a tracked audio file`,
      );
    }
  }
}

// 3) fail-closed contract per entry: reconciled ⇒ licence + source + attribution; legacy ⇒ frozen hash.
for (const e of ledger.entries) {
  const where = (Array.isArray(e.paths) && e.paths[0]) || e.hash;
  if (!/^[0-9a-f]{64}$/.test(String(e.hash))) problems.push(`BAD HASH: ${where} has malformed hash`);
  if (!ARCHETYPES.has(e.archetype)) {
    problems.push(`BAD ARCHETYPE: ${where} archetype='${e.archetype}'`);
  }
  if (e.rightsCleared) {
    if (e.archetype === "legacy-unreconciled") {
      problems.push(`INCOHERENT ENTRY: ${where} is rightsCleared but archetype legacy-unreconciled`);
    }
    const pv = e.provenance || {};
    if (!pv.license) problems.push(`NO LICENSE: ${where} is reconciled but has no provenance.license`);
    if (!pv.kind || !["licensed", "first-party"].includes(pv.kind)) {
      problems.push(
        `BAD PROVENANCE KIND: ${where} is reconciled but provenance.kind='${pv.kind}' (must be 'licensed' or 'first-party')`,
      );
    }
    const traceable = (pv.sourceUrl && pv.sourceUrl.length > 0) ||
      (pv.source && pv.source.length > 0) ||
      (pv.sourceAsset && pv.sourceAsset.length > 0);
    if (!traceable) {
      problems.push(
        `NO SOURCE: ${where} is reconciled but names no source (sourceUrl / source / sourceAsset)`,
      );
    }
    if (!pv.attribution) {
      problems.push(`NO ATTRIBUTION: ${where} is reconciled but has no attribution string`);
    }
    continue;
  }
  // Unreconciled: allowed ONLY for a frozen baseline hash, and only with an explicit reason.
  if (e.archetype !== "legacy-unreconciled") {
    problems.push(
      `UNRECONCILED BUT NOT LEGACY: ${where} has rightsCleared=false and archetype='${e.archetype}'`,
    );
  }
  if (!e.legacyReason) {
    problems.push(`LEGACY WITHOUT REASON: ${where} is legacy-unreconciled but has no legacyReason`);
  }
  if (!baselineHashes.has(e.hash)) {
    problems.push(
      `LEGACY NOT IN BASELINE: ${
        where
      } is declared legacy-unreconciled but its content hash is not in legacyBaseline.hashes — a new asset cannot be declared legacy; reconcile it with a licence, source and attribution`,
    );
  }
}

// 4) the legacy set is a FROZEN baseline: it can only shrink, never grow.
const legacyEntries = ledger.entries.filter((e) => !e.rightsCleared);
if (
  Number.isInteger(baseline.unreconciledContentHashes) &&
  legacyEntries.length > baseline.unreconciledContentHashes
) {
  problems.push(
    `LEGACY SET GREW: ${legacyEntries.length} unreconciled entries > frozen legacy baseline of ${baseline.unreconciledContentHashes} — reconcile the new asset instead of declaring it legacy`,
  );
}

// 5) totals + baseline self-consistency (the ledger must describe itself honestly).
if (ledger.totals.entries !== ledger.entries.length) {
  problems.push(
    `TOTALS DRIFT: totals.entries=${ledger.totals.entries} but entries[] has ${ledger.entries.length}`,
  );
}
const declaredFiles = ledger.entries.reduce(
  (n, e) => n + (Number.isInteger(e.fileCount) ? e.fileCount : 0),
  0,
);
if (declaredFiles !== files.length) {
  problems.push(
    `TOTALS DRIFT: entries[] cover ${declaredFiles} files but ${files.length} are tracked`,
  );
}
if (baselineHashes.size !== baseline.unreconciledContentHashes) {
  problems.push(
    `BASELINE DRIFT: legacyBaseline.hashes has ${baselineHashes.size} entries but unreconciledContentHashes=${baseline.unreconciledContentHashes}`,
  );
}
const summary = {
  trackedAudioFiles: files.length,
  ledgerEntries: ledger.entries.length,
  reconciledFiles,
  legacyFiles,
  legacyBaselineHashes: baselineHashes.size,
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
  console.error(`\naudio-provenance: ${files.length} files / ${ledger.entries.length} entries`);
  for (const p of problems.slice(0, 40)) console.error("  ✗ " + p);
  if (problems.length > 40) console.error(`  … and ${problems.length - 40} more`);
  console.error(
    `\nFAIL — ${problems.length} audio-provenance problem(s). No audio ships without a recorded provenance entry.`,
  );
  process.exit(1);
}

console.error(
  `audio-provenance: ${files.length} audio files → ${ledger.entries.length} content hashes · ` +
    `${reconciledFiles} reconciled (licence + source + attribution) · ${legacyFiles} legacy-unreconciled ` +
    `(frozen baseline ${baseline.sha.slice(0, 7)}, ${baselineHashes.size} hashes, never retroactively passed) · ` +
    `${JSON.stringify(summary.byArchetype)}`,
);
console.error(
  "PASS — every audio file maps by content hash to a declared provenance entry; no new, changed, or copied audio can ship undeclared.",
);
process.exit(0);
