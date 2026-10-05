#!/usr/bin/env node
// Fail-closed audio-provenance gate. Every in-scope file on the site MUST map, BY CONTENT HASH, to an
// entry in audio-provenance/ledger.json. A reconciled entry must carry a licence, a traceable source
// and an attribution; an unreconciled entry must be declared legacy AND its content hash must be part of
// the legacy baseline DERIVED from the anchored baseline commit. Because it re-hashes every file, ANY
// new, changed, or byte-different-copy in-scope file whose bytes aren't in the ledger fails the gate —
// you cannot ship audio without recording its provenance first. This is the mechanical enforcement of
// "no unverified source ships", the audio counterpart to scripts/check-image-provenance.mjs, and the
// station that was missing when ten byte-identical ted.wav copies and seventeen jfk.wav copies shipped
// through every gate unnoticed for six weeks.
//
// SCOPE IS INVERTED AND FAILS CLOSED. This gate USED an ALLOWLIST of audio extensions, so any extension
// not on the list was skipped silently: a committed .mkv (the SAME Matroska container as the .webm this
// gate admits), .mov, .ogv, .3gp, .mka, .caf, .w64 all carried real audio and passed with rc=0. The rule
// is now: every git-TRACKED file is IN SCOPE unless it is on the explicit "cannot contain audio"
// allowlist (scripts/audio-provenance-lib.mjs — source code, markup, styles, text, config, still images,
// fonts, data/model blobs, archives, certificates; extension-less files and opaque .bin blobs are listed
// by exact path). An unknown extension, a new media extension, or an extension-less file therefore FAILS
// the gate until a human adds it to that allowlist with a justification — the allowlist is a code change
// (a reviewable diff), never a silent default. Gitignored trees (node_modules/) stay out of scope by
// construction, and there is deliberately no filename exclusion: a committed test fixture is still
// bundled bytes and is in scope, exactly as a fixture raster is in scope for the image gate.
//
// FALSE-POSITIVE COST, stated plainly: a new legitimate NON-audio file type (a new font format, a doc
// format, a project file) now makes this gate red until its extension is added to the allowlist. That is
// the correct direction — loud, explicit and reviewable beats a real audio asset shipping silently
// because nobody had added its extension to a list.
//
// CONTAINER RULE. This gate does not decode a container to discover whether it actually holds audio, so
// it cannot tell an audio-bearing .webm from a video-only one — and it must not guess; guessing is what
// caused the failure above. Containers are therefore DECLARED in full, audio track or not: every tracked
// .webm/.mp4/.m4v/.mkv/.mov/.ogv/.3gp/.mka/... needs a ledger entry like any other bundled byte, and its
// entry says whether it is known to carry audio. The known-media extension list in the lib is a
// reporting aid, NOT the scope rule, and is explicitly non-exhaustive: scope comes from the
// cannot-contain-audio allowlist, so an extension nobody thought of is in scope by default.
//
// WHAT THIS GATE DOES NOT DO — presence and type, never truth. It checks that provenance fields are
// PRESENT and correctly TYPED, never that they are TRUE: a fabricated licence, source URL and
// attribution, with every field filled in, passes rc=0. And anyone with commit access can edit this
// gate itself. The guarantee is therefore that wrongdoing is VISIBLE IN A REVIEWABLE DIFF — a baseline
// anchor flip, a new legacy entry, a paths[] addition, an allowlist entry, a fabricated provenance
// record each show up as a line someone can review — not that wrongdoing is impossible. Nothing here
// resolves, ratifies or vouches for the rights status of any clip.
//
// THE BASELINE ANCHOR IS PINNED IN CODE. legacyBaseline.sha is ledger data, so on its own it is
// self-declared: repointing it at a commit that contains a new wav (or at "HEAD") and regenerating
// legacyBaseline.hashes would otherwise PASS while printing a new "frozen baseline DERIVED from <sha>"
// line. The anchor is therefore a documented constant in scripts/audio-provenance-lib.mjs
// (BASELINE_SHA); if the ledger's sha disagrees the gate fails with BASELINE ANCHOR MOVED. Moving the
// baseline is a CODE change and therefore a review tripwire, not a ledger edit.
//
// THE LEGACY BASELINE IS DERIVED, NOT TRUSTED. ledger.legacyBaseline.hashes is only a cross-check. At
// gate time this script re-reads the named commit (legacyBaseline.sha) with `git ls-tree -r` +
// `git cat-file --batch`, hashes every in-scope blob by content the same way it hashes the working tree,
// and FAILS with BASELINE MISMATCH if the declared list disagrees — so appending a hash to the ledger and
// bumping the counts cannot make the gate pass, and the printed "frozen baseline <sha>, N hashes" line is
// a re-derived fact. The derivation reads the commit object, never the working tree and never the
// branch's own HEAD~n history, so it is unaffected by a dirty or uncommitted checkout. It does need that
// commit object to exist locally (`git fetch` it if the clone is shallow); failing to derive is a FAIL,
// not a skip.
//
// The baseline is an explicit set of content HASHES, never a filename pattern: declaring a new asset
// legacy fails with LEGACY NOT IN BASELINE, and an undeclared further copy of a frozen asset fails with
// PATH DRIFT (a copy that is added to a ledger entry's paths[] is an explicit, reviewable ledger edit —
// the mechanism enforces that UNDECLARED copies fail, nothing more).
//
// audio-provenance/ledger.schema.json is documentation for readers and tooling; no script loads it. The
// in-code checks in this file are the contract: rightsCleared is compared with `=== true` and every
// provenance field is type-checked here.
//
// Usage: node scripts/check-audio-provenance.mjs [--json]
// Test hooks (fixtures only, never set in CI): --root <dir> --ledger <path> --baseline-sha <sha>
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import {
  BASELINE_SHA,
  extensionOf,
  inScope,
  KNOWN_MEDIA_EXT,
} from "./audio-provenance-lib.mjs";

const ARCHETYPES = new Set(["licensed", "first-party", "legacy-unreconciled"]);
const GIT_MAX = 512 * 1024 * 1024;

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : dflt;
};
const ROOT = flag("--root", new URL("..", import.meta.url).pathname);
const LEDGER = flag("--ledger", join(ROOT, "audio-provenance/ledger.json"));
// The anchored baseline commit: code, not ledger data (see the header). The hook exists so fixture
// repos can point the gate at their own throwaway baseline; it is never set in CI.
const ANCHOR = flag("--baseline-sha", BASELINE_SHA);

const sha256 = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");

// Derive the in-scope baseline set from a COMMIT — not the working tree, not HEAD~n. `git ls-tree -r`
// names the blobs, `git cat-file --batch` streams their bytes back in one process, and each body is
// hashed exactly the way a working-tree file is hashed above.
function audioBlobsAt(commit) {
  const tree = execFileSync("git", ["ls-tree", "-r", "-z", commit], {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: GIT_MAX,
  });
  const selected = [];
  for (const rec of tree.split("\0")) {
    if (!rec) continue;
    const tab = rec.indexOf("\t");
    if (tab < 0) continue;
    const [, type, oid] = rec.slice(0, tab).split(" ");
    const path = rec.slice(tab + 1);
    if (type !== "blob" || !inScope(path)) continue;
    selected.push({ path, oid });
  }
  if (!selected.length) return { files: 0, hashes: new Set() };
  const body = execFileSync("git", ["cat-file", "--batch"], {
    cwd: ROOT,
    input: selected.map((s) => s.oid).join("\n") + "\n",
    maxBuffer: GIT_MAX,
  });
  const hashes = new Set();
  let off = 0;
  for (const s of selected) {
    const nl = body.indexOf(0x0a, off);
    if (nl < 0) throw new Error(`git cat-file --batch returned nothing for ${s.path}`);
    const size = Number(body.subarray(off, nl).toString("utf8").split(" ")[2]);
    if (!Number.isInteger(size) || size < 0) throw new Error(`unreadable blob for ${s.path}`);
    hashes.add(createHash("sha256").update(body.subarray(nl + 1, nl + 1 + size)).digest("hex"));
    off = nl + 1 + size + 1;
  }
  return { files: selected.length, hashes };
}

let files;
try {
  files = execFileSync("git", ["ls-files"], {
    cwd: ROOT,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
  })
    .split("\n")
    .filter((f) => f && inScope(f));
} catch (e) {
  console.error(`FAIL — cannot enumerate tracked in-scope files at ${ROOT}:`, e.message);
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
if (baseline.sha !== ANCHOR) {
  console.error(
    `FAIL — BASELINE ANCHOR MOVED: audio-provenance/ledger.json declares legacyBaseline.sha=${baseline.sha} ` +
      `but this gate pins ${ANCHOR}.`,
  );
  console.error(
    "  The anchor is CODE (scripts/audio-provenance-lib.mjs, BASELINE_SHA), not ledger data, precisely so a" +
      " ledger-only flip — repointing the baseline at a commit that ships a new clip, or at \"HEAD\" — cannot" +
      " make this gate pass. Moving the baseline is a code change and a review tripwire: it must update" +
      " BASELINE_SHA in the same reviewable diff, with a reason.",
  );
  process.exit(1);
}
const declaredHashes = new Set(Array.isArray(baseline.hashes) ? baseline.hashes : []);
const entryByHash = new Map();
for (const e of ledger.entries) entryByHash.set(e.hash, e);

const problems = [];

// 0) DERIVE the legacy baseline from the named commit; the declared list is only a cross-check.
let derived;
try {
  derived = audioBlobsAt(baseline.sha);
} catch (e) {
  console.error(`FAIL — cannot derive the legacy baseline from commit ${baseline.sha}: ${e.message}`);
  console.error(
    "  The legacy set is DERIVED from legacyBaseline.sha, not trusted from the ledger. Fetch that commit (shallow clones: `git fetch origin` / `git fetch --unshallow`) or correct legacyBaseline.sha.",
  );
  process.exit(1);
}
const baselineHashes = derived.hashes;
if (
  declaredHashes.size !== baselineHashes.size ||
  [...declaredHashes].some((h) => !baselineHashes.has(h))
) {
  const undeclared = [...baselineHashes].filter((h) => !declaredHashes.has(h));
  const phantom = [...declaredHashes].filter((h) => !baselineHashes.has(h));
  problems.push(
    `BASELINE MISMATCH: legacyBaseline.hashes declares ${declaredHashes.size} content hashes but commit ` +
      `${baseline.sha.slice(0, 7)} contains ${baselineHashes.size} in-scope content hashes — ` +
      `${undeclared.length} at the commit are undeclared (${undeclared.slice(0, 3).map((h) => h.slice(0, 12)).join(", ")}) and ` +
      `${phantom.length} declared (${phantom.slice(0, 3).map((h) => h.slice(0, 12)).join(", ")}) are not in the commit. ` +
      `The declared list cross-checks the DERIVED set: re-derive it, never append to it.`,
  );
}
if (Number.isInteger(baseline.bundledFiles) && baseline.bundledFiles !== derived.files) {
  problems.push(
    `BASELINE MISMATCH: legacyBaseline.bundledFiles=${baseline.bundledFiles} but commit ${baseline.sha.slice(0, 7)} holds ${derived.files} in-scope files`,
  );
}

// 1) every tracked audio file must be in the ledger by content hash, listed under its own path.
let reconciledFiles = 0;
let legacyFiles = 0;
for (const f of files) {
  const h = sha256(join(ROOT, f));
  const e = entryByHash.get(h);
  if (!e) {
    const ext = extensionOf(f);
    const kind = KNOWN_MEDIA_EXT.has(ext)
      ? `a known media extension (.${ext})`
      : ext
        ? `extension .${ext}, which is NOT on the cannot-contain-audio allowlist`
        : "no extension, which is NOT on the cannot-contain-audio allowlist";
    problems.push(
      `UNLEDGERED AUDIO (unknown content hash): ${f} — this file is in scope (${kind}) and every in-scope file must have a provenance entry; add it to audio-provenance/ledger.json, or — only if it genuinely cannot carry audio — add its extension/path to scripts/audio-provenance-lib.mjs in the same reviewable diff`,
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
  if (e.rightsCleared === true) reconciledFiles++;
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

// 3) fail-closed contract per entry: reconciled ⇒ licence + source + attribution; legacy ⇒ derived hash.
// Only the literal `true` marks an entry reconciled (the STRING "false" is truthy in JS, so a stringy
// ledger must not slip through), and every provenance field is type-checked, not merely truth-tested.
for (const e of ledger.entries) {
  const where = (Array.isArray(e.paths) && e.paths[0]) || e.hash;
  if (!/^[0-9a-f]{64}$/.test(String(e.hash))) problems.push(`BAD HASH: ${where} has malformed hash`);
  if (!ARCHETYPES.has(e.archetype)) {
    problems.push(`BAD ARCHETYPE: ${where} archetype='${e.archetype}'`);
  }
  if (typeof e.rightsCleared !== "boolean") {
    problems.push(
      `BAD TYPE: ${where} rightsCleared must be a JSON boolean, got ${JSON.stringify(
        e.rightsCleared,
      )} — only the literal true marks an entry reconciled (the string "false" is truthy)`,
    );
  }
  if (e.rightsCleared === true) {
    if (e.archetype === "legacy-unreconciled") {
      problems.push(`INCOHERENT ENTRY: ${where} is rightsCleared but archetype legacy-unreconciled`);
    }
    const pv = e.provenance || {};
    if (typeof pv.license !== "string" || !pv.license) {
      problems.push(
        `NO LICENSE: ${where} is reconciled but provenance.license is not a non-empty string`,
      );
    }
    if (typeof pv.kind !== "string" || !["licensed", "first-party"].includes(pv.kind)) {
      problems.push(
        `BAD PROVENANCE KIND: ${where} is reconciled but provenance.kind=${JSON.stringify(
          pv.kind,
        )} (must be the string 'licensed' or 'first-party')`,
      );
    }
    const traceable = [pv.sourceUrl, pv.source, pv.sourceAsset].some(
      (v) => typeof v === "string" && v.length > 0,
    );
    if (!traceable) {
      problems.push(
        `NO SOURCE: ${where} is reconciled but names no source (sourceUrl / source / sourceAsset must be a non-empty string)`,
      );
    }
    if (typeof pv.attribution !== "string" || !pv.attribution) {
      problems.push(`NO ATTRIBUTION: ${where} is reconciled but has no attribution string`);
    }
    continue;
  }
  // Unreconciled: allowed ONLY for a hash derived from the named baseline commit, and only with an
  // explicit reason.
  if (e.archetype !== "legacy-unreconciled") {
    problems.push(
      `UNRECONCILED BUT NOT LEGACY: ${where} has rightsCleared=false and archetype='${e.archetype}'`,
    );
  }
  if (typeof e.legacyReason !== "string" || !e.legacyReason) {
    problems.push(
      `LEGACY WITHOUT REASON: ${where} is legacy-unreconciled but has no legacyReason string`,
    );
  }
  if (!baselineHashes.has(e.hash)) {
    problems.push(
      `LEGACY NOT IN BASELINE: ${
        where
      } is declared legacy-unreconciled but its content hash is not among the ${baselineHashes.size} in-scope content hashes derived from commit ${baseline.sha.slice(
        0,
        7,
      )} — a new asset cannot be declared legacy; reconcile it with a licence, source and attribution`,
    );
  }
}

// 4) the legacy set is a FROZEN baseline (the DERIVED one): it can only shrink, never grow.
const legacyEntries = ledger.entries.filter((e) => e.rightsCleared !== true);
if (legacyEntries.length > baselineHashes.size) {
  problems.push(
    `LEGACY SET GREW: ${legacyEntries.length} unreconciled entries > frozen legacy baseline of ${baselineHashes.size} — reconcile the new asset instead of declaring it legacy`,
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
if (declaredHashes.size !== baseline.unreconciledContentHashes) {
  problems.push(
    `BASELINE DRIFT: legacyBaseline.hashes has ${declaredHashes.size} entries but unreconciledContentHashes=${baseline.unreconciledContentHashes}`,
  );
}
const summary = {
  inScopeFiles: files.length,
  ledgerEntries: ledger.entries.length,
  reconciledFiles,
  legacyFiles,
  legacyBaselineHashes: baselineHashes.size,
  legacyBaselineFiles: derived.files,
  legacyBaselineSha: baseline.sha,
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
  `audio-provenance: ${files.length} in-scope files → ${ledger.entries.length} content hashes · ` +
    `${reconciledFiles} reconciled (licence + source + attribution) · ${legacyFiles} legacy-unreconciled ` +
    `(frozen baseline DERIVED from ${baseline.sha.slice(0, 7)} [anchor pinned in code: BASELINE_SHA]: ` +
    `${baselineHashes.size} hashes over ${derived.files} in-scope files, never retroactively passed) · ` +
    `${JSON.stringify(summary.byArchetype)}`,
);
console.error(
  "PASS — every in-scope file maps by content hash to a declared provenance entry; scope is fail-closed (unknown extension ⇒ in scope), so no new, changed, or copied media can ship undeclared.",
);
process.exit(0);
