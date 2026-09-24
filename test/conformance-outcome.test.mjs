// Deterministic tests for the conformance-gate OUTCOME rules (node:test, no browser, no network).
// Run: node --test test/conformance-outcome.test.mjs
//
// Why this file exists (bead web-ai-showcase-wty): scripts/check-conformance.mjs used to READ
// reports/conformance/results.json, print `fail N`, and exit 0 regardless — it verified that every
// suite was PRESENT and INTACT but never that the demo PASSED it. A red assertion therefore landed
// on main (mms-tts-bengali, bead web-ai-showcase-qjp). These tests pin the behaviour that closes
// that hole, and — just as importantly — pin the cases that must NOT turn red, so a future tightening
// cannot start failing the backlog.
//
// The decision logic lives in evaluateRecordedOutcome() as a pure function precisely so each branch
// is asserted here rather than only exercised by hand against the live repo.

import { test } from "node:test";
import assert from "node:assert/strict";
import { evaluateRecordedOutcome } from "../scripts/conformance-lib.mjs";

const TALLY = ["total", "tested", "pass", "fail", "blocked", "manual"];

/** Build a run record whose counters are, by construction, consistent with its results. */
function run(slug, states) {
  const results = states.map((state, i) => ({
    id: `assertion-${i}`,
    category: "runtime-config",
    deviceClass: "both",
    kind: "page-text",
    state,
    evidence: `synthetic ${state}`,
  }));
  const count = (s) => results.filter((r) => r.state === s).length;
  return {
    slug,
    total: results.length,
    tested: count("pass") + count("fail") + count("blocked"),
    pass: count("pass"),
    fail: count("fail"),
    blocked: count("blocked"),
    manual: count("manual"),
    results,
  };
}

/** Wrap runs in a rollup whose aggregate is, by construction, the sum of those runs. */
function rollup(runs, overrides = {}) {
  const aggregate = Object.fromEntries(
    TALLY.map((k) => [k, runs.reduce((a, r) => a + (r[k] ?? 0), 0)]),
  );
  return {
    generatedAt: "2026-09-24T00:00:00Z",
    suites: runs.length,
    aggregate,
    runs,
    ...overrides,
  };
}

const failuresOf = (doc, opts) => evaluateRecordedOutcome(doc, opts).failures;

// ── Rule 8: a recorded failure must fail the gate ──────────────────────────────────────────────

test("a recorded failing assertion produces a gate failure naming slug + assertion id", () => {
  const doc = rollup([run("demo-a", ["pass", "fail", "manual"])]);
  const f = failuresOf(doc);
  assert.equal(f.length, 1);
  assert.match(f[0], /FAILING ASSERTION/);
  assert.match(f[0], /demo-a/);
  assert.match(f[0], /assertion-1/);
  // the operator must be told the rule, not just the symptom
  assert.match(f[0], /fix the DEMO, never weaken the assertion/);
});

test("every failing assertion is reported, not just the first", () => {
  const doc = rollup([
    run("demo-a", ["fail", "pass", "fail"]),
    run("demo-b", ["fail"]),
  ]);
  const f = failuresOf(doc).filter((x) => x.startsWith("FAILING ASSERTION"));
  assert.equal(f.length, 3);
  assert.equal(f.filter((x) => x.includes("demo-a")).length, 2);
  assert.equal(f.filter((x) => x.includes("demo-b")).length, 1);
});

test("a failing assertion with no evidence still reports honestly", () => {
  const doc = rollup([run("demo-a", ["fail"])]);
  doc.runs[0].results[0].evidence = "";
  assert.match(failuresOf(doc)[0], /no evidence recorded/);
});

// ── The backlog must stay green ────────────────────────────────────────────────────────────────

test("an all-passing record is clean", () => {
  const doc = rollup([run("demo-a", ["pass", "pass", "manual"])]);
  assert.deepEqual(failuresOf(doc), []);
});

test("PARTIAL ROLLUP: suites with no run record are backlog, never failures", () => {
  // results.json is merge-by-slug: a targeted `--slug` run leaves every other suite untouched.
  const doc = rollup([run("demo-a", ["pass"])]);
  assert.deepEqual(failuresOf(doc, { builtCount: 327 }), []);
  const { lines } = evaluateRecordedOutcome(doc, { builtCount: 327 });
  assert.match(lines[0], /1\/327 suites have a run record/);
  assert.match(lines[0], /no record = backlog/);
});

test("manual and blocked states are not failures", () => {
  // `manual` needs an agent verdict; `blocked` is honest device/feature unavailability.
  const doc = rollup([run("demo-a", ["manual", "blocked", "manual"])]);
  assert.deepEqual(failuresOf(doc), []);
});

test("an empty rollup is not a failure (nothing recorded yet is not a red assertion)", () => {
  assert.deepEqual(failuresOf(rollup([])), []);
});

// ── Rule 9: the counters must agree, so no edit can hide a failure ─────────────────────────────

test("TAMPER: zeroing a run's fail counter is caught even though the state is still fail", () => {
  const doc = rollup([run("demo-a", ["fail", "pass"])]);
  doc.runs[0].fail = 0; // the lie
  doc.aggregate = Object.fromEntries(
    TALLY.map((k) => [k, doc.runs.reduce((a, r) => a + (r[k] ?? 0), 0)]),
  );
  const f = failuresOf(doc);
  // caught twice, on purpose: once by state (rule 8), once by counter (rule 9a)
  assert.equal(f.filter((x) => x.startsWith("FAILING ASSERTION")).length, 1);
  const inconsistent = f.filter((x) => x.startsWith("INCONSISTENT RECORD"));
  assert.equal(inconsistent.length, 1);
  assert.match(inconsistent[0], /reports fail=0 but its results array contains 1/);
  assert.match(inconsistent[0], /re-run/); // tells the operator the fix, not "edit the file"
});

test("TAMPER: a stale aggregate that hides a failure is caught by rule 8 and rule 9b", () => {
  const doc = rollup([run("demo-a", ["pass", "pass"])]);
  // inject a failure into the results WITHOUT touching any counter — the "stale summary" attack
  doc.runs[0].results[1].state = "fail";
  const f = failuresOf(doc);
  assert.equal(f.filter((x) => x.startsWith("FAILING ASSERTION")).length, 1);
  assert.equal(f.filter((x) => x.startsWith("INCONSISTENT RECORD")).length, 1);
});

test("a drifted aggregate is caught even with zero failing assertions", () => {
  const doc = rollup([run("demo-a", ["pass", "pass"])]);
  doc.aggregate.pass = 9999;
  const f = failuresOf(doc);
  assert.equal(f.length, 1);
  assert.match(f[0], /aggregate disagrees with its own runs/);
  assert.match(f[0], /pass 9999!=2/);
});

test("aggregate drift names every drifted key", () => {
  const doc = rollup([run("demo-a", ["pass", "manual"])]);
  doc.aggregate.pass = 5;
  doc.aggregate.manual = 7;
  const f = failuresOf(doc);
  assert.match(f[0], /pass 5!=1/);
  assert.match(f[0], /manual 7!=1/);
});

// ── Malformed / absent evidence must never read as "nothing failed" ────────────────────────────

test("a rollup with no runs array is malformed, not empty", () => {
  const f = failuresOf({ aggregate: {} });
  assert.equal(f.length, 1);
  assert.match(f[0], /MALFORMED/);
  assert.match(f[0], /runs/);
});

test("a rollup with no aggregate is malformed", () => {
  const f = failuresOf({ runs: [run("demo-a", ["pass"])] });
  assert.equal(f.filter((x) => /no `aggregate` summary/.test(x)).length, 1);
});

test("null / non-object input is malformed rather than silently clean", () => {
  for (const bad of [null, undefined, 42, "nope", []]) {
    const f = failuresOf(bad);
    assert.ok(f.length >= 1, `expected a failure for ${JSON.stringify(bad)}`);
    assert.match(f[0], /MALFORMED/);
  }
});

// ── Staleness is a prompt, not a failure ───────────────────────────────────────────────────────

test("a run record older than its suite's assertion count reports stale, does not fail", () => {
  const doc = rollup([run("demo-a", ["pass", "pass"])]); // recorded total = 2
  const opts = { suiteAssertionCounts: new Map([["demo-a", 5]]), builtCount: 1 };
  assert.deepEqual(failuresOf(doc, opts), []);
  const { lines } = evaluateRecordedOutcome(doc, opts);
  assert.ok(lines.some((l) => /stale: 1 run record/.test(l)));
});

test("a run record matching its suite's assertion count is not stale", () => {
  const doc = rollup([run("demo-a", ["pass", "pass"])]);
  const { lines } = evaluateRecordedOutcome(doc, {
    suiteAssertionCounts: new Map([["demo-a", 2]]),
    builtCount: 1,
  });
  assert.ok(!lines.some((l) => /stale/.test(l)));
});

// ── The report itself ──────────────────────────────────────────────────────────────────────────

test("the printed summary is recomputed from the runs, not copied from the aggregate", () => {
  const doc = rollup([run("demo-a", ["pass", "fail", "manual"])]);
  doc.aggregate.pass = 9999; // a lying summary must not become the printed number
  const { lines } = evaluateRecordedOutcome(doc, { builtCount: 10 });
  assert.match(lines[0], /pass 1 /);
  assert.ok(!lines[0].includes("9999"));
});
