// Shared gate-interaction helpers for the runtime-pin currency tests (bead web-ai-showcase-ehf).
//
// The default-suite half (test/runtime-pins.test.mjs) and the scheduled adversarial half
// (test-adversarial/runtime-pins-adversarial.test.mjs) both drive the REAL gate inside the
// isolated repository copy materialised by test/lib/isolated-repo.mjs. These helpers live here,
// single-sourced, so the two tiers can never diverge in how they run or judge the gate — the same
// guarantee test/lib/isolated-repo.mjs already gives the two mutating isolation tests.
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

/** Build the gate helpers bound to one isolated copy root and its imported audit module. */
export function makeGateHelpers({ copyRoot, audit }) {
  /** Runs the currency gate inside the isolated copy. Returns its exit status plus stdout/stderr. */
  function runGate() {
    try {
      const stdout = execFileSync(
        "node",
        [join(copyRoot, "scripts/audit-model-currency.mjs"), "--check"],
        { cwd: copyRoot, stdio: "pipe", encoding: "utf8" },
      );
      return { status: 0, stdout, stderr: "" };
    } catch (error) {
      return {
        status: typeof error.status === "number" ? error.status : 1,
        stdout: String(error.stdout ?? ""),
        stderr: String(error.stderr ?? ""),
      };
    }
  }

  /** Run the same grep the scanner runs (single-sourced pattern + targets) over the isolated copy.
   *  execFileSync argv (no shell) — mirrors the gate's grepScan; exit 1 = no matches. */
  function grepScan(pattern) {
    try {
      return execFileSync(
        "grep",
        ["-I", "-rnE", pattern, ...audit.PIN_SCAN_TARGETS.trim().split(/\s+/)],
        { cwd: copyRoot, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 },
      );
    } catch (e) {
      if (e.status === 1 && e.signal == null) return "";
      throw e;
    }
  }

  // --- binary-classified files (bead web-ai-showcase-5s4) ---------------------------------
  // `grep -I` skips a file the moment it contains a NUL byte, so a pin written into an ordinary
  // .js file with one stray control byte would never reach the version/route scans.
  function writeBinaryProbe(relPath, text) {
    writeFileSync(
      join(copyRoot, relPath),
      Buffer.concat([Buffer.from(text, "utf8"), Buffer.from([0x00]), Buffer.from("\n")]),
    );
    return join(copyRoot, relPath);
  }

  /** Mutate `relativePath` inside the copy, run the gate, assert it failed naming `needle`, restore. */
  function expectVersionFailure(relativePath, mutate, needle, message) {
    const path = join(copyRoot, relativePath);
    const original = readFileSync(path, "utf8");
    try {
      writeFileSync(path, mutate(original));
      const result = runGate();
      assert.equal(result.status, 1, message);
      assert.ok(result.stderr.includes(needle), `gate error must name ${needle}: ${result.stderr}`);
    } finally {
      writeFileSync(path, original, "utf8");
    }
    assert.equal(readFileSync(path, "utf8"), original, `${relativePath} must be restored byte-exactly`);
  }

  /** Append an executable URL string literal to a model worker, run the gate, assert the verdict. */
  function expectExecutableUrlVerdict(url, { status, needle }, message) {
    const p = join(copyRoot, "models/animegan-cartoonization/worker.js");
    const orig = readFileSync(p, "utf8");
    writeFileSync(p, `${orig}\nconst __probe_url = ${JSON.stringify(url)};\n`, "utf8");
    try {
      const result = runGate();
      assert.equal(result.status, status, message);
      if (needle) {
        assert.ok(result.stderr.includes(needle), `gate error must name "${needle}": ${result.stderr}`);
      }
    } finally {
      writeFileSync(p, orig, "utf8");
    }
    assert.equal(readFileSync(p, "utf8"), orig, "models/animegan-cartoonization/worker.js must be restored byte-exactly");
  }

  /** Apply several mutations at once, run the gate once, assert it fails naming EVERY needle, restore. */
  function expectBatchedFailures(mutations, needles, message) {
    const originals = mutations.map(({ path: rel }) => [rel, readFileSync(join(copyRoot, rel), "utf8")]);
    try {
      for (const { path: rel, mutate } of mutations) {
        writeFileSync(join(copyRoot, rel), mutate(readFileSync(join(copyRoot, rel), "utf8")));
      }
      const result = runGate();
      assert.equal(result.status, 1, message);
      for (const needle of needles) {
        assert.ok(result.stderr.includes(needle), `gate error must name ${needle}: ${result.stderr}`);
      }
    } finally {
      for (const [rel, text] of originals) writeFileSync(join(copyRoot, rel), text, "utf8");
    }
    for (const [rel, text] of originals) {
      assert.equal(readFileSync(join(copyRoot, rel), "utf8"), text, `${rel} must be restored byte-exactly`);
    }
  }

  return { runGate, grepScan, writeBinaryProbe, expectVersionFailure, expectExecutableUrlVerdict, expectBatchedFailures };
}
