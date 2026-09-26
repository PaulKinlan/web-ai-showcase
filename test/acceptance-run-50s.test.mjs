// 50s: deep acceptance suites must survive a stalled renderer and must not lose their evidence.
// Browser-free tests (verified by test/suite-stays-browser-free.test.mjs) for the shared CDP retry
// lever, the load helpers and the acceptance runner's abort-record behaviour.
import test from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  evaluateRetryCount,
  isTransientEvaluateTimeout,
} from "../scripts/browser.mjs";
import {
  parseCheckLines,
  parseRunnerArgs,
  resolveRunRecordPath,
  runAcceptanceRunner,
} from "../scripts/acceptance-run.mjs";

const ROOT = new URL("..", import.meta.url).pathname;

test("parseCheckLines keeps PASS/FAIL evidence and ignores progress noise", () => {
  const output = [
    "  [cdp evaluate retry 1/3] CDP timeout after 120000ms: Runtime.evaluate",
    "PASS  overview: bundled sample recognised — got \"x\"",
    "\u001b[32mPASS\u001b[0m  basics: real OCR",
    "FAIL  practical: queued region — timeout",
    "RESULT 44/44 checks; 10/10 route cells",
  ].join("\n");
  assert.deepEqual(parseCheckLines(output), [
    { name: "overview: bundled sample recognised", state: "pass", detail: 'got "x"' },
    { name: "basics: real OCR", state: "pass" },
    { name: "practical: queued region", state: "fail", detail: "timeout" },
  ]);
});

test("parseRunnerArgs separates runner flags, validator args and `--` passthrough", () => {
  const parsed = parseRunnerArgs([
    "scripts/validate-foo.mjs",
    "--write-run",
    "--max-load",
    "25",
    "--load-warn",
    "20",
    "--record",
    "reports/foo.json",
    "--retries",
    "5",
    "--",
    "--slug",
    "bar",
  ]);
  assert.equal(parsed.validator, "scripts/validate-foo.mjs");
  assert.deepEqual(parsed.passthrough, ["--write-run", "--slug", "bar"]);
  assert.equal(parsed.maxLoad, 25);
  assert.equal(parsed.loadWarn, 20);
  assert.equal(parsed.record, "reports/foo.json");
  assert.equal(parsed.retries, 5);
  assert.throws(() => parseRunnerArgs(["--bogus"]), /unknown flag/);
  assert.throws(() => parseRunnerArgs(["x.mjs", "--max-load"]), /needs a value/);
});

test("resolveRunRecordPath reads the family manifest, and falls back to the convention", () => {
  const fromManifest = resolveRunRecordPath("scripts/validate-manga-ocr.mjs");
  assert.equal(fromManifest, join(ROOT, "models/manga-ocr/acceptance-run.json"));
  const fallback = resolveRunRecordPath("scripts/validate-not-a-real-slug.mjs");
  assert.equal(fallback, join(ROOT, "models/not-a-real-slug/acceptance-run.json"));
  assert.equal(resolveRunRecordPath("scripts/validate-manga-ocr.mjs", "/tmp/x.json"), "/tmp/x.json");
});

test("evaluateRetryCount is opt-in, bounded and defensive", () => {
  assert.equal(evaluateRetryCount({}), 0);
  assert.equal(evaluateRetryCount({ CDP_EVALUATE_RETRIES: "0" }), 0);
  assert.equal(evaluateRetryCount({ CDP_EVALUATE_RETRIES: "3" }), 3);
  assert.equal(evaluateRetryCount({ CDP_EVALUATE_RETRIES: "2.7" }), 2);
  assert.equal(evaluateRetryCount({ CDP_EVALUATE_RETRIES: "99" }), 10);
  assert.equal(evaluateRetryCount({ CDP_EVALUATE_RETRIES: "nope" }), 0);
  assert.equal(evaluateRetryCount({ CDP_EVALUATE_RETRIES: "-1" }), 0);
});

test("isTransientEvaluateTimeout only matches a timed-out Runtime.evaluate", () => {
  const timeout = new Error("CDP timeout after 120000ms: Runtime.evaluate");
  assert.equal(isTransientEvaluateTimeout("Runtime.evaluate", timeout), true);
  assert.equal(isTransientEvaluateTimeout("Page.enable", timeout), false);
  assert.equal(
    isTransientEvaluateTimeout("Runtime.evaluate", new Error("Target closed")),
    false,
  );
});

test("runner writes a diagnostic failing record when the validator aborts", () => {
  const dir = mkdtempSync(join(tmpdir(), "acceptance-run-"));
  try {
    const fake = join(dir, "fake-validator.mjs");
    writeFileSync(
      fake,
      'console.log("PASS  alpha — ok");\nconsole.log("FAIL  beta — stalled");\nprocess.exit(1);\n',
    );
    const record = join(dir, "acceptance-run.json");
    const exit = runAcceptanceRunner([fake, "--record", record, "--load-warn", "99999"], {
      env: { ...process.env, CDP_EVALUATE_RETRIES: "0" },
    });
    assert.equal(exit, 1);
    assert.ok(existsSync(record), "diagnostic record must exist after an abort");
    const rec = JSON.parse(readFileSync(record, "utf8"));
    assert.equal(rec.aborted, true);
    assert.equal(rec.exitCode, 1);
    assert.match(rec.abortReason, /exited 1/);
    assert.deepEqual(
      rec.assertions.map((a) => [a.name, a.state]),
      [["alpha", "pass"], ["beta", "fail"]],
    );
    assert.equal(rec.summary.checks, 2);
    assert.equal(rec.summary.passed, 1);
    assert.equal(rec.summary.failed, 1);
    assert.ok(rec.notes.some((n) => /DIAGNOSTIC RECORD/.test(n)));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("runner leaves a passing run's record to the validator (writes nothing itself)", () => {
  const dir = mkdtempSync(join(tmpdir(), "acceptance-run-"));
  try {
    const fake = join(dir, "fake-validator-pass.mjs");
    writeFileSync(fake, 'console.log("PASS  alpha");\nprocess.exit(0);\n');
    const record = join(dir, "acceptance-run.json");
    const exit = runAcceptanceRunner([fake, "--record", record, "--load-warn", "99999"], {
      env: { ...process.env, CDP_EVALUATE_RETRIES: "0" },
    });
    assert.equal(exit, 0);
    assert.equal(existsSync(record), false, "the validator owns the passing record");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("runner refuses to start above --max-load, before any validator work", () => {
  const dir = mkdtempSync(join(tmpdir(), "acceptance-run-"));
  try {
    const fake = join(dir, "fake-validator-should-not-run.mjs");
    writeFileSync(fake, 'console.log("SHOULD NOT RUN");\nprocess.exit(0);\n');
    const record = join(dir, "acceptance-run.json");
    // --max-load -1 is always below the real load average, so the refusal is deterministic.
    const exit = runAcceptanceRunner([fake, "--record", record, "--max-load", "-1"], {
      env: { ...process.env, CDP_EVALUATE_RETRIES: "0" },
    });
    assert.equal(exit, 3);
    assert.equal(existsSync(record), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
