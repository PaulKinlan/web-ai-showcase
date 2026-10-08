#!/usr/bin/env node
// Regenerates inventory/runtime-pin-prose-fingerprints.json (bead web-ai-showcase-cdt).
//
// Fingerprints are the REVIEWED grandfather list for prose lines whose numeric runtime pin carries
// trailing prose punctuation (`,` `)` `:`) or a JSON escape artifact (`\` before an escaped quote) —
// candidates the widened VERSION_TOKEN judges whole and would otherwise reject. The generator finds
// every text-scan hit whose whole raw candidate is neither allowlisted nor derived-exempt, groups by
// (path, exact source line, candidate), and writes one entry per group with its occurrence count.
// Every regeneration is a review event: the diff of this file IS the list of prose lines the gate
// would newly fail on. Never hand-edit to suppress an unauthorized pin.
//
// Usage: node scripts/generate-prose-fingerprints.mjs [--check]
//   --check: exit 1 if the committed file differs from a fresh regeneration (used by tests).

import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(HERE, "..");
const OUT_PATH = join(REPO_ROOT, "inventory/runtime-pin-prose-fingerprints.json");

const {
  VERSION_TOKEN,
  PIN_SCAN_TARGETS,
} = await import("./audit-model-currency.mjs");
// Same construction as the gate's ORT/TJS_FULL_VERSION_RE (single-sourced VERSION_TOKEN), built
// locally so the generator does not change the ledgered marker lines in the audit script.
const ORT_FULL_VERSION_RE = new RegExp(`onnxruntime-web@(${VERSION_TOKEN})`, "g");
const TJS_FULL_VERSION_RE = new RegExp(`@huggingface/transformers@(${VERSION_TOKEN})`, "g");

const allowlist = JSON.parse(
  readFileSync(join(REPO_ROOT, "scripts/runtime-pin-allowlist.json"), "utf8"),
);
const allowedOrt = new Set(
  (allowlist.onnxruntimeWeb?.allowedVersions || []).map((v) => v.version),
);
const allowedTjs = new Set([allowlist.transformers?.shared]);
for (const o of allowlist.transformers?.allowedLocalOverrides || []) {
  allowedTjs.add(o.version);
}

const targets = PIN_SCAN_TARGETS.trim().split(/\s+/);
const grep = (pattern) => {
  try {
    return execFileSync("grep", ["-I", "-rnE", pattern, ...targets], {
      cwd: REPO_ROOT,
      encoding: "utf8",
      maxBuffer: 16 * 1024 * 1024,
    });
  } catch (e) {
    if (e.status === 1 && e.signal == null) return "";
    throw e;
  }
};

// Replicate the gate's derived-inventory exemption exactly (see isDerivedInventoryHit in
// audit-model-currency.mjs): wholly-derived files, plus the sw.js generated block (single ordered
// marker pair), version-scoped to the allowlist's measuredVersions.
const derivedInventoryFiles = new Set(allowlist.derivedInventory?.files || []);
const derivedMeasured = allowlist.derivedInventory?.measuredVersions || {};
let swGeneratedRange = null;
try {
  const swLines = readFileSync(join(REPO_ROOT, "sw.js"), "utf8").split("\n");
  const starts = swLines.map((l, n) => (l.includes(">>> runtime-integrity (generated") ? n : -1)).filter((n) => n >= 0);
  const ends = swLines.map((l, n) => (l.includes("<<< runtime-integrity") ? n : -1)).filter((n) => n >= 0);
  if (starts.length === 1 && ends.length === 1 && starts[0] < ends[0]) {
    swGeneratedRange = [starts[0] + 2, ends[0]];
  }
} catch { /* sw.js absent: nothing exempt */ }
const isDerivedHit = (file, lineNo, lib, version) => {
  if (!derivedInventoryFiles.has(file)) return false;
  if (file === "sw.js") {
    if (!swGeneratedRange) return false;
    const n = Number(lineNo);
    if (!(Number.isFinite(n) && n >= swGeneratedRange[0] && n <= swGeneratedRange[1])) return false;
  }
  const allowed = derivedMeasured[lib];
  return Array.isArray(allowed) && allowed.includes(version);
};

const groups = new Map(); // `${path}	${sha}	${candidate}` -> {path, sha, candidate, lineNo, occurrences}
for (const [pattern, re, lib, allowed] of [
  [`onnxruntime-web@${VERSION_TOKEN}`, ORT_FULL_VERSION_RE, "onnxruntime-web", allowedOrt],
  [`@huggingface/transformers@${VERSION_TOKEN}`, TJS_FULL_VERSION_RE, "@huggingface/transformers", allowedTjs],
]) {
  for (const line of grep(pattern).split("\n")) {
    const hit = line.match(/^(.*?):(\d+):(.+)$/);
    if (!hit) continue;
    const [, file, lineNo, content] = hit;
    for (const m of content.matchAll(re)) {
      const candidate = m[1];
      if (!candidate || allowed.has(candidate) || isDerivedHit(file, lineNo, lib, candidate)) continue;
      const sha = createHash("sha256").update(content, "utf8").digest("hex");
      const key = `${file}	${sha}	${candidate}`;
      const g = groups.get(key) || { path: file, lineSha256: sha, candidate, line: Number(lineNo), occurrences: 0 };
      g.occurrences++;
      groups.set(key, g);
    }
  }
}

const fingerprints = [...groups.values()].sort((a, b) =>
  a.path === b.path ? a.candidate.localeCompare(b.candidate) : a.path.localeCompare(b.path),
);

const doc = {
  $comment:
    "Reviewed prose fingerprints (bead web-ai-showcase-cdt): exact source lines whose numeric runtime pin carries prose punctuation/JSON-escape suffixes the widened VERSION_TOKEN judges whole. Regenerate ONLY via scripts/generate-prose-fingerprints.mjs and review the diff — each entry pins (path, full-line SHA-256, raw candidate, occurrence count); editing, duplicating, or removing the line turns the gate RED. Suppression never applies to binary-classified files and never masks the golden-ledger drift net.",
  fingerprints,
};

const out = JSON.stringify(doc, null, 2) + "\n";
if (process.argv.includes("--check")) {
  const committed = existsSync(OUT_PATH) ? readFileSync(OUT_PATH, "utf8") : "";
  if (committed !== out) {
    console.error(
      `runtime-pin-prose-fingerprints.json is stale — run node scripts/generate-prose-fingerprints.mjs and review the diff (${fingerprints.length} entr(ies) fresh vs committed)`,
    );
    process.exit(1);
  }
  console.log(`prose fingerprints current (${fingerprints.length} entries)`);
} else {
  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, out);
  console.log(`wrote ${fingerprints.length} prose fingerprint(s) to inventory/runtime-pin-prose-fingerprints.json`);
}
