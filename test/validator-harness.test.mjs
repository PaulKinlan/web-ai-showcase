// web-ai-showcase-5m1: Acceptance validator integrity helpers.
// Path logic, formatting, and process management only — browser-free unit tests (verified by
// test/suite-stays-browser-free.test.mjs).
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { constants as osConstants, tmpdir } from "node:os";
import { join } from "node:path";
import {
  activeChromeInstances,
  assertHeadUnchanged,
  captureHeadCommit,
  cleanupAllChromeInstances,
  createIsolatedProfileDir,
  formatAcceptanceSummary,
  getRunLogPath,
  printAcceptanceSummary,
  writeAcceptanceRunRecord,
} from "../scripts/browser.mjs";

test("formatAcceptanceSummary: complete passing run returns ok: true with full denominator", () => {
  const results = [
    { route: "models/codegen-350m/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/basics/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/practical/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/wild/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/", viewport: "mobile", pass: true },
    { route: "models/codegen-350m/basics/", viewport: "mobile", pass: true },
    { route: "models/codegen-350m/practical/", viewport: "mobile", pass: true },
    { route: "models/codegen-350m/wild/", viewport: "mobile", pass: true },
  ];
  const summary = formatAcceptanceSummary({
    passed: 32,
    total: 32,
    expectedChecks: 32,
    results,
    expectedCells: 8,
  });
  assert.equal(summary.ok, true);
  assert.equal(summary.message, "32/32 checks passed across 8/8 route cells");
});

test("formatAcceptanceSummary: truncated run (e.g. 28/28 reached out of 32) reports INCOMPLETE and never masquerades as passed", () => {
  const results = [
    { route: "models/codegen-350m/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/basics/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/practical/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/wild/", viewport: "desktop", pass: true },
    { route: "models/codegen-350m/", viewport: "mobile", pass: true },
    { route: "models/codegen-350m/basics/", viewport: "mobile", pass: true },
    { route: "models/codegen-350m/practical/", viewport: "mobile", pass: true },
  ];
  const summary = formatAcceptanceSummary({
    passed: 28,
    total: 28,
    expectedChecks: 32,
    results,
    expectedCells: 8,
  });
  assert.equal(summary.ok, false);
  assert.doesNotMatch(summary.message, /28\/28 checks passed/);
  assert.match(summary.message, /REACHED 28 of EXPECTED 32 checks/);
  assert.match(summary.message, /INCOMPLETE/);
});

test("formatAcceptanceSummary: incomplete route cells reports INCOMPLETE even if all checks passed", () => {
  const results = [
    { route: "models/sample/", viewport: "desktop", pass: true },
  ];
  const summary = formatAcceptanceSummary({
    passed: 10,
    total: 10,
    expectedChecks: 10,
    results,
    expectedCells: 2,
  });
  assert.equal(summary.ok, false);
  assert.match(summary.message, /only 1 of EXPECTED 2 route cells — INCOMPLETE/);
});

test("formatAcceptanceSummary: failing checks report FAILED", () => {
  const results = [
    { route: "models/sample/", viewport: "desktop", pass: false },
  ];
  const summary = formatAcceptanceSummary({
    passed: 8,
    total: 10,
    expectedChecks: 10,
    results,
    expectedCells: 1,
  });
  assert.equal(summary.ok, false);
  assert.match(summary.message, /8\/10 checks passed \(2 failed\) — FAILED/);
});

test("formatAcceptanceSummary: failing route cell reports FAILED", () => {
  const results = [
    { route: "models/sample/", viewport: "desktop", pass: true },
    { route: "models/sample/", viewport: "mobile", pass: false },
  ];
  const summary = formatAcceptanceSummary({
    passed: 10,
    total: 10,
    expectedChecks: 10,
    results,
    expectedCells: 2,
  });
  assert.equal(summary.ok, false);
  assert.match(summary.message, /1 route cells failed/);
});

test("printAcceptanceSummary returns ok boolean matching formatAcceptanceSummary", () => {
  const logged = [];
  const origLog = console.log;
  console.log = (msg) => logged.push(msg);
  try {
    const ok = printAcceptanceSummary({
      passed: 5,
      total: 5,
      expectedChecks: 5,
    });
    assert.equal(ok, true);
    assert.match(logged[0], /5\/5 checks passed/);
  } finally {
    console.log = origLog;
  }
});

test("captureHeadCommit captures 40-hex commit hash", () => {
  const commit = captureHeadCommit();
  assert.match(commit, /^[0-9a-f]{40}$/);
});

test("assertHeadUnchanged passes when commit matches HEAD", () => {
  const commit = captureHeadCommit();
  assert.equal(assertHeadUnchanged(commit), true);
});

test("assertHeadUnchanged throws when commit differs from HEAD", () => {
  const fakeCommit = "0123456789abcdef0123456789abcdef01234567";
  assert.throws(
    () => assertHeadUnchanged(fakeCommit),
    /HEAD moved during acceptance run/,
  );
});

test("createIsolatedProfileDir produces unique path under tmpdir with prefix and pid", () => {
  const dir1 = createIsolatedProfileDir("test1");
  const dir2 = createIsolatedProfileDir("test1");
  assert.ok(dir1.startsWith(tmpdir()));
  assert.match(dir1, /webai-chrome-profile-test1-/);
  assert.notEqual(dir1, dir2);
});

test("getRunLogPath produces unique log path under tmpdir with slug and pid", () => {
  const log1 = getRunLogPath("mms-forced-alignment");
  const log2 = getRunLogPath("mms-forced-alignment");
  assert.ok(log1.startsWith(tmpdir()));
  assert.match(log1, /acceptance-mms-forced-alignment-/);
  assert.match(log1, new RegExp(`${process.pid}.*\\.log$`));
  assert.notEqual(log1, log2);
});

test("cleanupAllChromeInstances manages active set cleanly", () => {
  cleanupAllChromeInstances();
  assert.equal(activeChromeInstances.size, 0);

  let killed = false;
  const mockInstance = {
    kill: () => {
      killed = true;
    },
  };
  activeChromeInstances.add(mockInstance);
  cleanupAllChromeInstances();
  assert.equal(killed, true);
  assert.equal(activeChromeInstances.size, 0);
});

test("writeAcceptanceRunRecord writes JSON when HEAD matches and returns true", () => {
  const commit = captureHeadCommit();
  const tmpFile = join(tmpdir(), `test-run-record-${Date.now()}-${process.pid}.json`);
  try {
    const results = [{ route: "models/test/", viewport: "desktop", pass: true }];
    const ok = writeAcceptanceRunRecord({
      runRecordPath: tmpFile,
      startCommit: commit,
      results,
    });
    assert.equal(ok, true);
    assert.ok(existsSync(tmpFile));
    const data = JSON.parse(readFileSync(tmpFile, "utf8"));
    assert.equal(data.commit, commit);
    assert.equal(data.exitCode, 0);
    assert.deepEqual(data.results, results);
  } finally {
    if (existsSync(tmpFile)) rmSync(tmpFile);
  }
});

test("writeAcceptanceRunRecord refuses cleanly without throwing when HEAD moved", () => {
  const fakeCommit = "0123456789abcdef0123456789abcdef01234567";
  const tmpFile = join(tmpdir(), `test-run-record-${Date.now()}-${process.pid}.json`);
  const errors = [];
  const origError = console.error;
  console.error = (msg) => errors.push(msg);
  try {
    const ok = writeAcceptanceRunRecord({
      runRecordPath: tmpFile,
      startCommit: fakeCommit,
      results: [],
    });
    assert.equal(ok, false);
    assert.equal(existsSync(tmpFile), false, "must not create run record when refused");
    assert.ok(errors.length > 0);
    assert.match(errors[0], /REFUSAL: HEAD moved during acceptance run/);
  } finally {
    console.error = origError;
    if (existsSync(tmpFile)) rmSync(tmpFile);
  }
});

test("signal exit codes follow 128 + signal convention", () => {
  assert.equal(128 + osConstants.signals["SIGINT"], 130);
  assert.equal(128 + osConstants.signals["SIGTERM"], 143);
  assert.equal(128 + osConstants.signals["SIGHUP"], 129);
});

