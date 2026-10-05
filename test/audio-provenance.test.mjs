// Deterministic tests for the audio-provenance ledger + fail-closed gate (scripts/check-audio-provenance.mjs).
//
// Why this file exists (bead web-ai-showcase-eba): the image side has had a fail-closed provenance gate for
// months, but nothing inspected audio. Ten byte-identical ted.wav copies (a clip the repo's own record calls
// prohibited) and seventeen jfk.wav copies shipped through every gate for six weeks without a single check
// noticing. This suite pins the replacement station: the ledger's own contract, the gate's PASS on the real
// tree, the wiring that keeps it in the gate chain, the DERIVED baseline (recomputed from git blobs at the
// anchored commit and compared to the ledger's declared list), the INVERTED fail-closed scope rule (a .mkv —
// the same Matroska container as .webm — and any unknown extension are in scope), the pinned baseline anchor
// (a ledger-only sha flip is rejected), and — critically — a POSITIVE and several NEGATIVE fixture controls
// proving a new undeclared file really fails, that an appended baseline hash cannot make the gate pass, and
// that a stringy rightsCleared is rejected.
//
// No browser, no network: git, the ledger, and a throwaway git repo in the OS temp dir.

import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import {
  BASELINE_SHA,
  extensionOf,
  inScope,
  NON_AUDIO_EXT,
  NON_AUDIO_PATH,
} from "../scripts/audio-provenance-lib.mjs";

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
  assert.match(r.stderr, /^audio-provenance: \d+ in-scope files/m);
  assert.match(r.stderr, /PASS — every in-scope file maps by content hash/);
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
const MKV = Buffer.from("\x1a\x45\xdf\xa3-not-really-mkv", "binary");

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

function runGate(dir, baselineSha) {
  // Fixture repos pin their own baseline commit; the gate's real anchor is asserted separately against the
  // real ledger, so this only supplies the --baseline-sha test hook (never set in CI).
  const pinned = baselineSha ??
    JSON.parse(readFileSync(join(dir, "audio-provenance/ledger.json"), "utf8")).legacyBaseline.sha;
  return spawnSync("node", [GATE, "--root", dir, "--baseline-sha", pinned], { cwd: ROOT, encoding: "utf8" });
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

test("the declared legacy baseline EQUALS the set re-derived from the anchored commit", () => {
  // Independent of the gate's own derivation: list the commit's tree, hash every in-scope blob with our own
  // ls-tree + cat-file blob calls, compare to the ledger. This is what makes "frozen baseline" a re-derived
  // fact instead of a self-declaration. The scope predicate is the shared one (tested directly below) so the
  // test fails loudly if the allowlist and the ledger ever disagree about what is in scope.
  assert.equal(ledger.legacyBaseline.sha, BASELINE_SHA, "the ledger anchor must equal the code anchor");
  const tree = execFileSync("git", ["ls-tree", "-r", "-z", ledger.legacyBaseline.sha], {
    cwd: ROOT,
    encoding: "utf8",
  });
  const inScopeSet = tree.split("\0").filter(Boolean).map((rec) => {
    const tab = rec.indexOf("\t");
    return { oid: rec.slice(0, tab).split(" ")[2], path: rec.slice(tab + 1) };
  }).filter((f) => inScope(f.path));
  const derived = new Set(
    inScopeSet.map((f) => sha(execFileSync("git", ["cat-file", "blob", f.oid], { cwd: ROOT, maxBuffer: 64 * 1024 * 1024 }))),
  );
  assert.deepEqual([...derived].sort(), [...ledger.legacyBaseline.hashes].sort());
  assert.equal(ledger.legacyBaseline.bundledFiles, inScopeSet.length);
  assert.equal(ledger.legacyBaseline.unreconciledContentHashes, derived.size);
});

test("the scope rule is INVERTED and fails closed on unknown extensions", () => {
  // C1 of the delta re-review: the old rule was an ALLOWLIST of audio extensions, so .mkv — the same
  // Matroska container as the admitted .webm — .mov, .ogv, .3gp, .mka, .caf and .w64 carried real audio and
  // passed rc=0. Now every tracked file is in scope unless it is explicitly known not to carry audio.
  for (const p of ["a.wav", "a.mp3", "a.webm", "a.mp4", "a.mkv", "a.mov", "a.ogv", "a.3gp", "a.mka", "a.caf", "a.w64", "a.opus"]) {
    assert.ok(inScope(p), `${p} must be in scope (it can carry audio)`);
  }
  // Unknown and extension-less files are in scope by default — the whole point of the inversion.
  for (const p of ["a.somebrandnewformat", "a", "models/x/Makefile", "a.bin", "a.dat"]) {
    assert.ok(inScope(p), `${p} must be in scope (unknown ⇒ fail closed)`);
  }
  for (const p of ["a.md", "a.html", "a.css", "a.js", "a.mjs", "a.json", "a.png", "a.jpg", "a.woff2", "a.txt", ".gitignore", "a.yml"]) {
    assert.ok(!inScope(p), `${p} must be out of scope (cannot carry audio)`);
  }
  // Extension-less and opaque .bin files are allowlisted BY EXACT PATH, so a new one is still in scope.
  assert.ok(!inScope("models/speecht5-tts/speakers/awb.bin"));
  assert.ok(!inScope(".beads/hooks/pre-commit"));
  assert.ok(inScope("models/some-new-demo/weights.bin"), "a NEW opaque .bin must not ride the allowlist");
  // Every allowlist entry carries a human-readable justification.
  for (const [ext, why] of NON_AUDIO_EXT) assert.ok(why && why.length, `extension ${ext} has no justification`);
  for (const [p, why] of NON_AUDIO_PATH) assert.ok(why && why.length, `path ${p} has no justification`);
  assert.equal(extensionOf(".gitignore"), "gitignore");
  assert.equal(extensionOf(".beads/hooks/pre-commit"), "");
  assert.equal(extensionOf("a/b/Archive.TAR.GZ"), "gz");
});

test("the gate's baseline anchor is pinned in code, not read from the ledger", () => {
  // Item 2 of fix pass 3: legacyBaseline.sha is ledger data; repointing it (at a commit that ships a new
  // clip, or at "HEAD") and regenerating the hashes used to PASS while printing a new DERIVED line. The
  // anchor lives in scripts/audio-provenance-lib.mjs, so changing the baseline is a code change.
  assert.equal(BASELINE_SHA, "e9c20b75ba5b83866ad4367461a8100a07dc5afc");
  assert.equal(ledger.legacyBaseline.sha, BASELINE_SHA);
  const src = readFileSync("scripts/check-audio-provenance.mjs", "utf8");
  assert.match(src, /BASELINE_SHA/, "the gate must use the pinned anchor");
  assert.match(src, /BASELINE ANCHOR MOVED/, "the gate must reject a ledger-only anchor flip");
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

test("NEGATIVE CONTROL: a .mkv container FAILS (UNLEDGERED AUDIO) — the decisive C1 remux case", () => {
  // Fix pass 3, item 1: .mkv was NOT on yesterday's allowlist. It is the SAME Matroska container as .webm,
  // which the gate admits, so remuxing the existing fixture to a sibling extension walked past the gate.
  const rel = "models/new-demo/remuxed-sample.mkv";
  const dir = fixtureRepo(
    { baseline: { "README.md": README }, now: { [rel]: MKV } },
    baseLedger({ totals: { bundledFiles: 0, entries: 0 } }),
  );
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`UNLEDGERED AUDIO \\(unknown content hash\\): ${rel}`));
    assert.match(r.stderr, /a known media extension \(\.mkv\)/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: an unknown extension FAILS (fail-closed, not skipped)", () => {
  const rel = "models/new-demo/sample.somebrandnewformat";
  const dir = fixtureRepo(
    { baseline: { "README.md": README }, now: { [rel]: wavB } },
    baseLedger({ totals: { bundledFiles: 0, entries: 0 } }),
  );
  try {
    const r = runGate(dir);
    assert.equal(r.status, 1, `expected FAIL, got ${r.status}:\n${r.stderr}`);
    assert.match(r.stderr, new RegExp(`UNLEDGERED AUDIO \\(unknown content hash\\): ${rel}`));
    assert.match(r.stderr, /extension \.somebrandnewformat, which is NOT on the cannot-contain-audio allowlist/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("NEGATIVE CONTROL: a ledger-only baseline anchor flip FAILS (BASELINE ANCHOR MOVED)", () => {
  // Fix pass 3, item 2: legacyBaseline.sha is ledger data, so on its own it is self-declared. The gate pins
  // the anchor in code; a ledger (or ledger-plus-hashes) flip that does not also change the code fails.
  const rel = "models/fixture-a/ted.wav";
  const dir = fixtureRepo({ baseline: { [rel]: wavC } }, baseLedger({
    totals: { bundledFiles: 1, entries: 1 },
    legacyBaseline: { sha: "0".repeat(40), note: "fixture", bundledFiles: 1, unreconciledContentHashes: 1, hashes: [sha(wavC)] },
    entries: [legacyEntry(sha(wavC), rel, wavC)],
  }));
  try {
    // The fixture ledger's own sha is accepted (that is the pin), but a DIFFERENT declared sha is not.
    assert.equal(runGate(dir).status, 0, "the fixture's own pinned anchor must pass");
    const flipped = runGate(dir, "f".repeat(40));
    assert.equal(flipped.status, 1, `expected FAIL, got ${flipped.status}:\n${flipped.stderr}`);
    assert.match(flipped.stderr, /FAIL — BASELINE ANCHOR MOVED/);
    assert.match(flipped.stderr, /The anchor is CODE|anchor is CODE/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
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
    assert.match(r.stderr, /PASS — every in-scope file maps by content hash/);
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

