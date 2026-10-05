// Deterministic tests for the audio-provenance ledger + fail-closed gate (scripts/check-audio-provenance.mjs).
//
// Why this file exists (bead web-ai-showcase-eba): the image side has had a fail-closed provenance gate for
// months, but nothing inspected audio. Ten byte-identical ted.wav copies (a clip the repo's own record calls
// prohibited) and seventeen jfk.wav copies shipped through every gate for six weeks without a single check
// noticing. This suite pins the replacement station: the ledger's own contract, the gate's PASS on the real
// tree, the wiring that keeps it in the gate chain, and — critically — a POSITIVE and several NEGATIVE
// fixture controls proving a new undeclared audio file really fails.
//
// No browser, no network: git, the ledger, and a throwaway git repo in the OS temp dir.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

const ROOT = process.cwd();
const GATE = "scripts/check-audio-provenance.mjs";
const ledger = JSON.parse(readFileSync("audio-provenance/ledger.json", "utf8"));

const ARCHETYPES = new Set(["licensed", "first-party", "legacy-unreconciled"]);

// ── The ledger record ─────────────────────────────────────────────────────────────────────────

test("audio ledger parses and has the required top-level fields", () => {
  for (const k of ["name", "version", "generated", "policy", "legacyBaseline", "totals", "entries"]) {
    assert.ok(k in ledger, `missing ${k}`);
  }
  assert.ok(ledger.entries.length > 0);
  assert.equal(ledger.totals.entries, ledger.entries.length);
  assert.equal(ledger.legacyBaseline.hashes.length, ledger.legacyBaseline.unreconciledContentHashes);
});

test("every entry has a well-formed hash, known archetype, ≥1 path, and a coherent fileCount", () => {
  for (const e of ledger.entries) {
    const where = (e.paths && e.paths[0]) || e.hash;
    assert.match(e.hash, /^[0-9a-f]{64}$/, `bad hash for ${where}`);
    assert.ok(ARCHETYPES.has(e.archetype), `bad archetype '${e.archetype}' for ${where}`);
    assert.ok(Array.isArray(e.paths) && e.paths.length >= 1, `no paths for ${where}`);
    assert.equal(e.fileCount, e.paths.length, `fileCount != paths.length for ${where}`);
    assert.ok(e.bytes > 0, `non-positive byte count for ${where}`);
    assert.equal(e.rightsCleared, e.archetype !== "legacy-unreconciled", `rightsCleared incoherent for ${where}`);
  }
});

test("every reconciled entry is licensed/attributable; every legacy entry names its reason and is frozen", () => {
  const frozen = new Set(ledger.legacyBaseline.hashes);
  for (const e of ledger.entries) {
    const where = e.paths[0];
    if (e.rightsCleared) {
      const p = e.provenance || {};
      assert.ok(["licensed", "first-party"].includes(p.kind), `${where} has no valid provenance.kind`);
      assert.ok(p.license && p.license.length, `${where} has no license`);
      assert.ok(p.attribution && p.attribution.length, `${where} has no attribution`);
      const traceable = (p.sourceUrl && p.sourceUrl.length) || (p.source && p.source.length) ||
        (p.sourceAsset && p.sourceAsset.length);
      assert.ok(traceable, `${where} has no traceable source`);
      continue;
    }
    assert.equal(e.archetype, "legacy-unreconciled", `${where} is unreconciled but not legacy`);
    assert.ok(e.legacyReason && e.legacyReason.length, `${where} is legacy with no legacyReason`);
    assert.ok(frozen.has(e.hash), `${where} is legacy but its hash is not frozen in legacyBaseline.hashes`);
  }
});

test("no path appears in two entries, and the ledger does not silently drop files", () => {
  const seen = new Set();
  for (const e of ledger.entries) {
    for (const p of e.paths) {
      assert.ok(!seen.has(p), `${p} appears in two entries`);
      seen.add(p);
    }
  }
  const covered = ledger.entries.reduce((n, e) => n + e.fileCount, 0);
  assert.equal(covered, ledger.totals.bundledFiles, "entries do not cover every bundled file");
});

// ── The gate on the real tree ─────────────────────────────────────────────────────────────────

test("the fail-closed audio gate passes on the current tree", () => {
  // Runs the real gate; a non-zero exit is reported with the gate's own stderr, which is the quotable failure.
  const r = spawnSync("node", [GATE], { cwd: ROOT, encoding: "utf8" });
  assert.equal(r.status, 0, `gate exited ${r.status}:\n${r.stderr}`);
  assert.match(r.stderr, /^audio-provenance: \d+ audio files/m);
  assert.match(r.stderr, /PASS — every audio file maps by content hash/);
});

test("the audio gate is wired into the gate chain (deno task gate) and CI", () => {
  const deno = JSON.parse(readFileSync("deno.json", "utf8"));
  assert.match(
    String(deno.tasks?.gate || ""),
    /check-audio-provenance\.mjs/,
    "the audio gate must sit in `deno task gate` beside the image gate",
  );
  // Positioned exactly where the image gate sits, so both provenance stations run before the rest.
  const chain = String(deno.tasks.gate).split(" && ").map((s) => s.trim());
  assert.equal(
    chain[chain.indexOf("node scripts/check-audio-provenance.mjs") - 1],
    "node scripts/check-image-provenance.mjs",
  );
  const workflow = readFileSync(".github/workflows/gate.yml", "utf8");
  assert.ok(
    workflow.includes("node scripts/check-audio-provenance.mjs"),
    "CI must run the audio gate",
  );
});

// ── Positive + negative controls on throwaway git repos ───────────────────────────────────────

const WAV = Buffer.from("RIFF\x00\x00\x00\x00WAVEfmt ", "binary"); // bytes only need to differ, not decode
const wavB = Buffer.concat([WAV, Buffer.from([1])]);
const wavC = Buffer.concat([WAV, Buffer.from([2])]);

/** A throwaway git repo holding `files`, with a ledger written at audio-provenance/ledger.json. */
function fixtureRepo(files, ledgerDoc) {
  const dir = mkdtempSync(join(tmpdir(), "audio-prov-"));
  for (const [rel, buf] of Object.entries(files)) {
    const p = join(dir, rel);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, buf);
  }
  mkdirSync(join(dir, "audio-provenance"), { recursive: true });
  writeFileSync(join(dir, "audio-provenance/ledger.json"), JSON.stringify(ledgerDoc, null, 2));
  execFileSync("git", ["init", "-q"], { cwd: dir });
  execFileSync("git", ["add", "-A"], { cwd: dir });
  return dir;
}

const baseLedger = (over) => ({
  name: "fixture",
  version: 1,
  generated: "2026-10-05",
  policy: ["fixture"],
  legacyBaseline: { sha: "0".repeat(40), note: "fixture", bundledFiles: 0, unreconciledContentHashes: 0, hashes: [] },
  totals: {},
  entries: [],
  ...over,
});

function runGate(dir) {
  return spawnSync("node", [GATE, "--root", dir], { cwd: ROOT, encoding: "utf8" });
}

const sha = (buf) => createHash("sha256").update(buf).digest("hex");

test("POSITIVE CONTROL: a reconciled fixture asset (licence + source + attribution) passes", () => {
  const rel = "models/fixture-demo/sample.wav";
  const dir = fixtureRepo({ [rel]: wavB }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    entries: [{
      hash: sha(wavB),
      bytes: wavB.length,
      fileCount: 1,
      archetype: "licensed",
      rightsCleared: true,
      provenance: {
        kind: "licensed",
        source: "Wikimedia Commons",
        sourceUrl: "https://commons.wikimedia.org/wiki/File:Example.ogg",
        creator: "Someone",
        license: "CC0-1.0",
        attribution: "Someone — Example.ogg — via Wikimedia Commons — CC0 1.0",
      },
      evidence: "fixture",
      paths: [rel],
    }],
  }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 0, `expected PASS, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, /PASS — every audio file maps by content hash/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a new undeclared audio asset FAILS (UNLEDGERED AUDIO)", () => {
  // The whole point of the gate: an asset that is not in the ledger cannot ship.
  const rel = "models/new-demo/sample.wav";
  const dir = fixtureRepo({ [rel]: wavB }, baseLedger({ totals: { bundledFiles: 0, entries: 0 } }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`UNLEDGERED AUDIO \\(unknown content hash\\): ${rel}`));
    assert.match(r.stderr, /FAIL — \d+ audio-provenance problem/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a new asset cannot be laundered through the legacy baseline", () => {
  // Declaring the new file legacy-unreconciled is exactly how a weak allowlist would excuse it; the frozen
  // baseline must reject it, because its content hash is not in legacyBaseline.hashes.
  const rel = "models/new-demo/sample.wav";
  const dir = fixtureRepo({ [rel]: wavB }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    legacyBaseline: {
      sha: "0".repeat(40),
      note: "fixture",
      bundledFiles: 1,
      unreconciledContentHashes: 1,
      hashes: [sha(wavC)], // deliberately not the new asset's hash
    },
    entries: [{
      hash: sha(wavB),
      bytes: wavB.length,
      fileCount: 1,
      archetype: "legacy-unreconciled",
      rightsCleared: false,
      legacyReason: "pretending this is legacy",
      evidence: "fixture",
      paths: [rel],
    }],
  }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`LEGACY NOT IN BASELINE: ${rel}`));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a byte-identical COPY of a frozen clip at a new path FAILS (PATH DRIFT)", () => {
  // The incident: ted.wav copied into a tenth demo. Same bytes, new path. It must not slip past.
  const declared = "models/fixture-a/ted.wav";
  const copy = "models/fixture-b/sample-ted.wav";
  const dir = fixtureRepo({ [declared]: wavC, [copy]: wavC }, baseLedger({
    totals: { bundledFiles: 2, entries: 1 },
    legacyBaseline: {
      sha: "0".repeat(40),
      note: "fixture",
      bundledFiles: 2,
      unreconciledContentHashes: 1,
      hashes: [sha(wavC)],
    },
    entries: [{
      hash: sha(wavC),
      bytes: wavC.length,
      fileCount: 1,
      archetype: "legacy-unreconciled",
      rightsCleared: false,
      legacyReason: "fixture legacy clip",
      evidence: "fixture",
      paths: [declared],
    }],
  }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`PATH DRIFT: ${copy}`));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
