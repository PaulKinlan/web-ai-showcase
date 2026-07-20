// Deterministic tests for the image-provenance ledger + fail-closed gate.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";

const ledger = JSON.parse(readFileSync("image-provenance/ledger.json", "utf8"));

test("ledger parses and has entries + required top-level fields", () => {
  for (const k of ["name", "version", "generated", "policy", "totals", "entries"]) {
    assert.ok(k in ledger, `missing ${k}`);
  }
  assert.ok(ledger.entries.length > 0);
  assert.equal(ledger.totals.uniqueImages, ledger.entries.length);
});

test("every entry has a well-formed 64-hex hash, known archetype, ≥1 path", () => {
  const ARCH = new Set([
    "media-library",
    "licensed-derived",
    "media-derived",
    "procedural",
    "qa-screenshot",
  ]);
  for (const e of ledger.entries) {
    assert.match(e.hash, /^[0-9a-f]{64}$/, `bad hash ${e.paths?.[0]}`);
    assert.ok(ARCH.has(e.archetype), `bad archetype ${e.archetype}`);
    assert.ok(Array.isArray(e.paths) && e.paths.length >= 1);
    assert.ok(e.provenance && e.provenance.kind && e.provenance.license);
  }
});

test("FAIL-CLOSED: every identifiable person is licensed, cleared, sourced, and attributed", () => {
  const people = ledger.entries.filter((e) => e.depictsIdentifiablePerson);
  assert.ok(people.length > 0, "expected some identifiable-person images");
  for (const e of people) {
    const p = e.provenance;
    assert.equal(e.rightsCleared, true, `${e.paths[0]} not rightsCleared`);
    assert.equal(p.kind, "licensed", `${e.paths[0]} not licensed (kind=${p.kind})`);
    assert.ok(p.license && p.license.length, `${e.paths[0]} no license`);
    assert.ok(p.attribution && p.attribution.length, `${e.paths[0]} no attribution`);
    const traceable = (p.sourceUrl && p.sourceUrl.length > 0) || p.sourceAsset === "faces-crowd";
    assert.ok(traceable, `${e.paths[0]} not traceable to a source`);
  }
});

test("no procedural / qa-screenshot entry claims to depict an identifiable person", () => {
  for (const e of ledger.entries) {
    if (e.archetype === "procedural" || e.archetype === "qa-screenshot") {
      assert.equal(
        e.depictsIdentifiablePerson,
        false,
        `${e.paths[0]} is ${e.archetype} but claims identifiable person`,
      );
      assert.equal(
        e.provenance.kind,
        "first-party",
        `${e.paths[0]} is ${e.archetype} but not first-party`,
      );
    }
  }
});

test("identifiable ⊆ people (an identifiable person is a person)", () => {
  for (const e of ledger.entries) {
    if (e.depictsIdentifiablePerson) assert.equal(e.depictsPeople, true, `${e.paths[0]}`);
  }
});

test("no path appears in two different entries (unique file → one provenance)", () => {
  const seen = new Map();
  for (const e of ledger.entries) {
    for (const p of e.paths) {
      assert.ok(!seen.has(p), `${p} in two entries`);
      seen.set(p, e.hash);
    }
  }
});

test("no demo falsely labels its (now-licensed) sample faces as synthetic/StyleGAN/not-real-people", () => {
  // The face demos were relicensed to real Wikimedia Commons portraits; the old "synthetic StyleGAN2 /
  // not real people" copy is now false and must be gone. gfpgan-face-restoration legitimately describes
  // the MODEL's StyleGAN2 face prior, so it's excluded.
  const hits = execSync(
    "grep -rliE 'stylegan|synthetic (face|person|portrait|people)|not real (people|person)|thispersondoesnotexist' models/ --include=*.html --include=*.js || true",
    { encoding: "utf8" },
  ).trim().split("\n").filter((f) => f && !f.includes("gfpgan-face-restoration"));
  assert.deepEqual(hits, [], `demos still contain false synthetic-face claims: ${hits.join(", ")}`);
});

test("the fail-closed gate passes on the current tree", () => {
  // Runs the real gate; throws (non-zero exit) if any image is unledgered or any face is unverified.
  const out = execSync("node scripts/check-image-provenance.mjs", {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  // (stderr carries the PASS line; execSync throwing would fail the test)
  assert.ok(true, out);
});
