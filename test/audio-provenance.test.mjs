// Deterministic tests for the audio-provenance ledger + fail-closed gate (scripts/check-audio-provenance.mjs).
//
// Why this file exists (bead web-ai-showcase-eba): the image side has had a fail-closed provenance gate for
// months, but nothing inspected audio. Ten byte-identical ted.wav copies (a clip the repo's own record calls
// prohibited) and seventeen jfk.wav copies shipped through every gate for six weeks without a single check
// noticing. This suite pins the replacement station: the ledger's own contract, the gate's PASS on the real
// tree, the wiring that keeps it in the gate chain, the DERIVED baseline (recomputed from git blobs at the
// named commit and compared to the ledger's declared list), and — critically — a POSITIVE and several NEGATIVE
// fixture controls proving a new undeclared file (including an audio-bearing .webm container) really fails,
// that an appended baseline hash cannot make the gate pass, and that a stringy rightsCleared is rejected.
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
    assert.equal(typeof e.rightsCleared, "boolean", `rightsCleared is not a JSON boolean for ${where}`);
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
const WEBM = Buffer.from("\x1a\x45\xdf\xa3-not-really-ebml", "binary");

const GIT_ID = ["-c", "user.email=fixture@example.invalid", "-c", "user.name=fixture"];

/**
 * A throwaway git repo. `baseline` is written and COMMITTED first — that commit is the one the gate
 * DERIVES the legacy baseline from. `now` is then written and STAGED on top, which leaves the tree
 * dirty: that is how these fixtures show the derivation reads the commit, not the working tree.
 * The ledger's legacyBaseline.sha is overwritten with the real baseline commit unless `keepSha` is set
 * (used by the test that the gate refuses to proceed when the baseline cannot be derived).
 */
function fixtureRepo({ baseline = {}, now = {} }, ledgerDoc, { keepSha = false } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "audio-prov-"));
  const writeAll = (files) => {
    for (const [rel, buf] of Object.entries(files)) {
      const p = join(dir, rel);
      mkdirSync(dirname(p), { recursive: true });
      writeFileSync(p, buf);
    }
  };
  writeAll(baseline);
  execFileSync("git", ["init", "-q"], { cwd: dir });
  execFileSync("git", ["add", "-A"], { cwd: dir });
  execFileSync("git", [...GIT_ID, "commit", "-qm", "baseline"], { cwd: dir });
  const baselineSha = execFileSync("git", ["rev-parse", "HEAD"], { cwd: dir, encoding: "utf8" }).trim();
  writeAll(now);
  mkdirSync(join(dir, "audio-provenance"), { recursive: true });
  if (!keepSha) ledgerDoc.legacyBaseline.sha = baselineSha;
  writeFileSync(join(dir, "audio-provenance/ledger.json"), JSON.stringify(ledgerDoc, null, 2));
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

/** One legacy entry for `hash` at `rel`, the shape every fixture below needs. */
const legacyEntry = (hash, rel, buf, over = {}) => ({
  hash,
  bytes: buf.length,
  fileCount: 1,
  archetype: "legacy-unreconciled",
  rightsCleared: false,
  legacyReason: "fixture legacy clip",
  evidence: "fixture",
  paths: [rel],
  ...over,
});

const README = Buffer.from("# fixture\n");

// ── The real tree: the baseline is DERIVED, the container fixture is declared ───────────────────

test("the declared legacy baseline EQUALS the set re-derived from the named commit", () => {
  // Independent of the gate: list the commit's tree, hash every in-scope blob, compare to the ledger.
  // This is what makes "frozen baseline" a re-derived fact instead of a self-declaration.
  const AUDIO = /\.(wav|mp3|ogg|m4a|flac|aac|opus|oga|aiff|aif|wma|webm|mp4|m4v)$/i;
  const tree = execFileSync("git", ["ls-tree", "-r", "-z", ledger.legacyBaseline.sha], {
    cwd: ROOT,
    encoding: "utf8",
  });
  const inScope = tree.split("\0").filter(Boolean).map((rec) => {
    const tab = rec.indexOf("\t");
    return { oid: rec.slice(0, tab).split(" ")[2], path: rec.slice(tab + 1) };
  }).filter((f) => AUDIO.test(f.path));
  const derived = new Set(
    inScope.map((f) => sha(execFileSync("git", ["cat-file", "blob", f.oid], { cwd: ROOT, maxBuffer: 64 * 1024 * 1024 }))),
  );
  assert.deepEqual([...derived].sort(), [...ledger.legacyBaseline.hashes].sort());
  assert.equal(ledger.legacyBaseline.bundledFiles, inScope.length);
  assert.equal(ledger.legacyBaseline.unreconciledContentHashes, derived.size);
});

test("the audio-bearing CONTAINER fixture is declared, not allowlisted away", () => {
  // lib/__capture-selftest__/sample-clip.webm is a VP9 video with a real Opus audio track. It is a test
  // fixture, but it is still bundled bytes — the five-extension allowlist used to skip it silently.
  const webm = "lib/__capture-selftest__/sample-clip.webm";
  const entry = ledger.entries.find((e) => e.hash === sha(readFileSync(webm)));
  assert.ok(entry, `${webm} must have a ledger entry keyed by its content hash`);
  assert.ok(entry.paths.includes(webm));
  assert.match(entry.legacyReason, /Opus/);
});

// ── Fixture controls ───────────────────────────────────────────────────────────────────────────

test("POSITIVE CONTROL: a reconciled fixture asset (licence + source + attribution) passes", () => {
  const rel = "models/fixture-demo/sample.wav";
  const dir = fixtureRepo({ baseline: { [rel]: wavB } }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    legacyBaseline: { sha: "0".repeat(40), note: "fixture", bundledFiles: 1, unreconciledContentHashes: 1, hashes: [sha(wavB)] },
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
    assert.match(r.stderr, /frozen baseline DERIVED from/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a new undeclared audio asset FAILS (UNLEDGERED AUDIO)", () => {
  // The whole point of the gate: an asset that is not in the ledger cannot ship.
  const rel = "models/new-demo/sample.wav";
  const dir = fixtureRepo(
    { baseline: { "README.md": README }, now: { [rel]: wavB } },
    baseLedger({ totals: { bundledFiles: 0, entries: 0 } }),
  );
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`UNLEDGERED AUDIO \\(unknown content hash\\): ${rel}`));
    assert.match(r.stderr, /FAIL — \d+ audio-provenance problem/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a new undeclared .webm/.opus container FAILS (UNLEDGERED AUDIO)", () => {
  // C1 of the cross-family review: the extension allowlist missed audio-bearing containers, so
  // lib/__capture-selftest__/sample-clip.webm (vp9 + opus) shipped undeclared while the gate passed.
  const webm = "models/new-demo/clip.webm";
  const opus = "models/new-demo/clip.opus";
  const dir = fixtureRepo(
    { baseline: { "README.md": README }, now: { [webm]: WEBM, [opus]: wavB } },
    baseLedger({ totals: { bundledFiles: 0, entries: 0 } }),
  );
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`UNLEDGERED AUDIO \\(unknown content hash\\): ${webm}`));
    assert.match(r.stderr, new RegExp(`UNLEDGERED AUDIO \\(unknown content hash\\): ${opus}`));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a new asset cannot be laundered through the legacy baseline", () => {
  // Declaring the new file legacy-unreconciled is exactly how a weak allowlist would excuse it; the
  // DERIVED baseline rejects it, because its content hash is not part of the commit's audio set.
  const frozen = "models/fixture-a/ted.wav";
  const fresh = "models/new-demo/sample.wav";
  const dir = fixtureRepo({ baseline: { [frozen]: wavC }, now: { [fresh]: wavB } }, baseLedger({
    totals: { bundledFiles: 2, entries: 2 },
    legacyBaseline: { sha: "0".repeat(40), note: "fixture", bundledFiles: 1, unreconciledContentHashes: 1, hashes: [sha(wavC)] },
    entries: [legacyEntry(sha(wavC), frozen, wavC), legacyEntry(sha(wavB), fresh, wavB, { legacyReason: "pretending this is legacy" })],
  }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`LEGACY NOT IN BASELINE: ${fresh}`));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: appending a hash to the declared baseline FAILS (BASELINE MISMATCH)", () => {
  // C2 of the cross-family review: appending a hash to legacyBaseline.hashes and bumping the counts
  // used to make the gate PASS and print a false "frozen baseline" line, because nothing pinned it.
  const rel = "models/fixture-a/ted.wav";
  const dir = fixtureRepo({ baseline: { [rel]: wavC } }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    legacyBaseline: {
      sha: "0".repeat(40),
      note: "fixture",
      bundledFiles: 1,
      unreconciledContentHashes: 2,
      hashes: [sha(wavC), "e".repeat(64)], // the appended phantom
    },
    entries: [legacyEntry(sha(wavC), rel, wavC)],
  }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, /BASELINE MISMATCH: legacyBaseline\.hashes declares 2 content hashes but commit/);
    assert.match(r.stderr, /eeeeeeeeeeee/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a baseline commit the gate cannot read FAILS loudly (not a skip)", () => {
  const rel = "models/fixture-a/ted.wav";
  const dir = fixtureRepo({ baseline: { [rel]: wavC } }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    legacyBaseline: { sha: "1".repeat(40), note: "fixture", bundledFiles: 1, unreconciledContentHashes: 1, hashes: [sha(wavC)] },
    entries: [legacyEntry(sha(wavC), rel, wavC)],
  }), { keepSha: true });
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, /FAIL — cannot derive the legacy baseline from commit 1111111/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('NEGATIVE CONTROL: rightsCleared "false" (a string) is rejected, not treated as reconciled', () => {
  // C4 of the cross-family review: the string "false" is truthy, so `if (e.rightsCleared)` used to mark
  // a stringy entry reconciled. Only the literal JSON boolean true may do that.
  const rel = "models/fixture-demo/sample.wav";
  const dir = fixtureRepo({ baseline: { [rel]: wavB } }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    legacyBaseline: { sha: "0".repeat(40), note: "fixture", bundledFiles: 1, unreconciledContentHashes: 1, hashes: [sha(wavB)] },
    entries: [{
      hash: sha(wavB),
      bytes: wavB.length,
      fileCount: 1,
      archetype: "licensed",
      rightsCleared: "false", // the bug: a truthy string
      provenance: {
        kind: "licensed",
        license: "CC0-1.0",
        sourceUrl: "https://example.org/x",
        attribution: "someone",
      },
      evidence: "fixture",
      paths: [rel],
    }],
  }));
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, /BAD TYPE: .* rightsCleared must be a JSON boolean, got "false"/);
    assert.match(r.stderr, /UNRECONCILED BUT NOT LEGACY/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a byte-identical COPY of a frozen clip at a new path FAILS (PATH DRIFT)", () => {
  // The incident: ted.wav copied into a tenth demo. Same bytes, new path. It must not slip past unless
  // the copy is explicitly named in the entry's paths[] (a reviewable ledger edit).
  const declared = "models/fixture-a/ted.wav";
  const copy = "models/fixture-b/sample-ted.wav";
  const dir = fixtureRepo(
    { baseline: { [declared]: wavC }, now: { [copy]: wavC } },
    baseLedger({
      totals: { bundledFiles: 2, entries: 1 },
      legacyBaseline: { sha: "0".repeat(40), note: "fixture", bundledFiles: 2, unreconciledContentHashes: 1, hashes: [sha(wavC)] },
      entries: [legacyEntry(sha(wavC), declared, wavC)],
    }),
  );
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`PATH DRIFT: ${copy}`));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

