// EmbeddingGemma 2 acceptance: validator-owned split-run HALF artifacts + mechanical 10-cell merge.
// Browser-free (verified by test/suite-stays-browser-free.test.mjs): imports only the pure merge module,
// never the Chrome harness. Every fail-closed rule below is exercised AND, during development, was
// falsified by temporarily breaking the corresponding check (see the report).
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  EXPECTED_ROUTES,
  mergeHalves,
  validateHalfPair,
} from "../scripts/embeddinggemma-2-half-merge.mjs";

const COMMIT = "a".repeat(40);
const BLOB = "b".repeat(64);
const SIBLING = new URL("../models/all-distilroberta-v1/acceptance-run.json", import.meta.url);

function cells(viewport) {
  return EXPECTED_ROUTES.map((route) => ({ route, viewport, pass: true }));
}

function half(viewport, overrides = {}) {
  return {
    kind: "embeddinggemma-2-half",
    viewport,
    commit: COMMIT,
    validatorBlobSha: BLOB,
    ranAt: "2026-10-08T00:00:00.000Z",
    exitCode: 0,
    pass: true,
    results: cells(viewport),
    ...overrides,
  };
}

/** Write halves to a temp dir and run the real disk+merge pipeline, optionally seeding a final record. */
function runMerge(desktop, mobile, { currentCommit = COMMIT, currentBlob = BLOB, seedFinal = null } = {}) {
  const dir = mkdtempSync(join(tmpdir(), "egemma2-half-"));
  const desktopPath = join(dir, "desktop-half.json");
  const mobilePath = join(dir, "mobile-half.json");
  const finalPath = join(dir, "acceptance-run.json");
  if (desktop !== null) writeFileSync(desktopPath, JSON.stringify(desktop, null, 2) + "\n");
  if (mobile !== null) writeFileSync(mobilePath, JSON.stringify(mobile, null, 2) + "\n");
  if (seedFinal !== null) writeFileSync(finalPath, seedFinal);
  const outcome = mergeHalves({
    desktopHalfPath: desktopPath,
    mobileHalfPath: mobilePath,
    finalRecordPath: finalPath,
    currentCommit,
    currentValidatorBlobSha: currentBlob,
  });
  return {
    dir,
    finalPath,
    outcome,
    finalExists: existsSync(finalPath),
    finalContent: () => (existsSync(finalPath) ? readFileSync(finalPath, "utf8") : null),
  };
}

test("missing half -> merge refuses with a clear reason and writes nothing", () => {
  const r = runMerge(half("desktop"), null);
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /missing half/);
    assert.match(r.outcome.reason, /mobile/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("failed/timed-out half (exitCode != 0) -> merge refuses", () => {
  const r = runMerge(half("desktop", { exitCode: 1 }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /exitCode is 1, not 0/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("a half explicitly marked failed (pass false) -> merge refuses", () => {
  const r = runMerge(half("desktop", { pass: false }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /marked failed/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("stale half — differing source commit sha -> merge refuses", () => {
  const r = runMerge(half("desktop", { commit: "c".repeat(40) }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /source commit differs/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("stale half — differing validator blob sha -> merge refuses", () => {
  const r = runMerge(half("desktop", { validatorBlobSha: "d".repeat(64) }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /validator blob sha differs/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("stale half — half cites a commit older than HEAD -> merge refuses", () => {
  const r = runMerge(half("desktop"), half("mobile"), { currentCommit: "e".repeat(40) });
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /HEAD is eeeeeee/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("duplicate half — the same viewport supplied twice -> merge refuses", () => {
  const r = runMerge(half("desktop"), half("desktop"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /duplicate half/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("wrong cell count (4) -> merge refuses", () => {
  const r = runMerge(half("desktop", { results: cells("desktop").slice(0, 4) }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /desktop half has 4 results \(expected exactly 5\)/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("wrong cell count (6) -> merge refuses", () => {
  const six = [...cells("desktop"), { route: "models/embeddinggemma-2/extra/", viewport: "desktop", pass: true }];
  const r = runMerge(half("desktop", { results: six }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /desktop half has 6 results \(expected exactly 5\)/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("duplicate route within a viewport -> merge refuses", () => {
  const dup = cells("desktop");
  dup[4] = { ...dup[0] }; // multimodel replaced by a second overview → 5 cells, 4 unique routes
  const r = runMerge(half("desktop", { results: dup }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /duplicate route within the viewport/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("a half cell that claims the other viewport -> merge refuses (cannot claim an un-run viewport)", () => {
  const wrong = cells("desktop");
  wrong[0] = { ...wrong[0], viewport: "mobile" };
  const r = runMerge(half("desktop", { results: wrong }), half("mobile"));
  try {
    assert.equal(r.outcome.ok, false);
    assert.match(r.outcome.reason, /claims viewport "mobile" but the half ran "desktop"/);
    assert.equal(r.finalExists, false);
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("GOOD case: two valid halves merge into the exact 10-cell sibling schema + split metadata, idempotently", () => {
  const sibling = JSON.parse(readFileSync(SIBLING, "utf8"));
  const first = runMerge(half("desktop"), half("mobile"));
  try {
    assert.equal(first.outcome.ok, true, first.outcome.reason);
    assert.equal(first.finalExists, true);
    const bytes1 = first.finalContent();

    const record = JSON.parse(bytes1);
    // Field-for-field match against the sibling schema's top-level keys…
    assert.deepEqual(Object.keys(sibling).sort(), ["commit", "exitCode", "ranAt", "results"].sort());
    for (const key of Object.keys(sibling)) {
      assert.ok(key in record, `merged record missing sibling key "${key}"`);
    }
    assert.equal(typeof record.commit, "string");
    assert.equal(record.commit, COMMIT);
    assert.equal(typeof record.ranAt, "string");
    assert.equal(record.exitCode, 0);
    assert.ok(Array.isArray(record.results));
    // …plus explicit split-method metadata naming both halves and their provenance.
    assert.ok(record.splitMethod, "merged record must carry splitMethod metadata");
    assert.equal(record.splitMethod.kind, "viewport-split-merge");
    assert.equal(record.splitMethod.halves.length, 2);
    assert.deepEqual(
      record.splitMethod.halves.map((h) => h.viewport).sort(),
      ["desktop", "mobile"],
    );
    for (const h of record.splitMethod.halves) {
      assert.equal(h.commit, COMMIT);
      assert.equal(h.validatorBlobSha, BLOB);
      assert.equal(h.exitCode, 0);
      assert.equal(h.pass, true);
    }

    // Exactly 10 cells: the expected 5 routes once per viewport, each {route, viewport, pass}.
    assert.equal(record.results.length, 10);
    const desktopCells = record.results.filter((c) => c.viewport === "desktop");
    const mobileCells = record.results.filter((c) => c.viewport === "mobile");
    assert.equal(desktopCells.length, 5);
    assert.equal(mobileCells.length, 5);
    assert.deepEqual(desktopCells.map((c) => c.route), EXPECTED_ROUTES);
    assert.deepEqual(mobileCells.map((c) => c.route), EXPECTED_ROUTES);
    assert.ok(record.results.every((c) => c.pass === true));
    // Cell shape is exactly the sibling's three keys.
    for (const c of record.results) {
      assert.deepEqual(Object.keys(c).sort(), ["pass", "route", "viewport"]);
    }

    // Idempotent: re-merging the same halves must be byte-identical.
    const again = mergeHalves({
      desktopHalfPath: join(first.dir, "desktop-half.json"),
      mobileHalfPath: join(first.dir, "mobile-half.json"),
      finalRecordPath: first.finalPath,
      currentCommit: COMMIT,
      currentValidatorBlobSha: BLOB,
    });
    assert.equal(again.ok, true, again.reason);
    assert.equal(first.finalContent(), bytes1);
  } finally {
    rmSync(first.dir, { recursive: true, force: true });
  }
});

test("atomicity: a failing merge never touches a pre-existing good record", () => {
  const sentinel = "PRE-EXISTING GOOD RECORD\n";
  // Failing because the mobile half is missing.
  const r = runMerge(half("desktop"), null, { seedFinal: sentinel });
  try {
    assert.equal(r.outcome.ok, false);
    assert.equal(r.finalContent(), sentinel, "a refused merge must leave the existing record byte-for-byte");
  } finally {
    rmSync(r.dir, { recursive: true, force: true });
  }
});

test("validateHalfPair is pure: the good path returns a record with no disk side effects", () => {
  const verdict = validateHalfPair({
    desktop: half("desktop"),
    mobile: half("mobile"),
    currentCommit: COMMIT,
    currentValidatorBlobSha: BLOB,
  });
  assert.equal(verdict.ok, true, verdict.reason);
  assert.equal(verdict.record.results.length, 10);
  assert.equal(verdict.record.commit, COMMIT);
});
