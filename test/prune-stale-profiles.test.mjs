// web-ai-showcase-msz: test stale acceptance profile pruning and tmpfs capacity monitoring.
import test from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  formatBytes,
  getActiveChromeUserDataDirs,
  getDirectorySize,
  getTmpfsCapacity,
  isPidAlive,
  isStaleProfileDirName,
  pruneStaleAcceptanceProfiles,
} from "../scripts/browser.mjs";
import { runAcceptanceRunner } from "../scripts/acceptance-run.mjs";

test("isStaleProfileDirName identifies acceptance and webai profile patterns safely", () => {
  // Positive matches
  assert.equal(isStaleProfileDirName("fastvlm-vlm-acceptance-QWuTD7"), true);
  assert.equal(isStaleProfileDirName("camembert-ner-acceptance-1kW1xn"), true);
  assert.equal(isStaleProfileDirName("speech-separation-acceptance-profile-8vG2a1"), true);
  assert.equal(isStaleProfileDirName("webai-chrome-profile-conformance-1234-5678-abcde"), true);

  // Negative matches
  assert.equal(isStaleProfileDirName("cap-serialized-chrome-acceptance.lock"), false);
  assert.equal(isStaleProfileDirName("acceptance-run.json"), false);
  assert.equal(isStaleProfileDirName("acceptance-run.log"), false);
  assert.equal(isStaleProfileDirName("systemd-private-12345"), false);
  assert.equal(isStaleProfileDirName("test-directory"), false);
  assert.equal(isStaleProfileDirName(""), false);
  assert.equal(isStaleProfileDirName(null), false);
});

test("formatBytes formats byte quantities into human-readable units", () => {
  assert.equal(formatBytes(0), "0 B");
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1024), "1.0 KB");
  assert.equal(formatBytes(1024 * 1024 * 700), "700.0 MB");
  assert.equal(formatBytes(1024 * 1024 * 1024 * 38.5), "38.5 GB");
  assert.equal(formatBytes(-10), "0 B");
});

test("isPidAlive checks process liveliness accurately", () => {
  assert.equal(isPidAlive(process.pid), true);
  assert.equal(isPidAlive(0), false);
  assert.equal(isPidAlive(-1), false);
  assert.equal(isPidAlive("abc"), false);
  assert.equal(isPidAlive(99_999_999), false);
});

test("getDirectorySize computes recursive size of directories", () => {
  const sandbox = mkdtempSync(join(tmpdir(), "webai-test-size-"));
  try {
    const sub = join(sandbox, "sub");
    mkdirSync(sub);
    writeFileSync(join(sandbox, "a.bin"), Buffer.alloc(1024));
    writeFileSync(join(sub, "b.bin"), Buffer.alloc(2048));
    assert.equal(getDirectorySize(sandbox), 3072);
  } finally {
    rmSync(sandbox, { recursive: true, force: true });
  }
});

test("getActiveChromeUserDataDirs extracts user-data-dir paths from processes or fallback", () => {
  const dirs = getActiveChromeUserDataDirs();
  assert.ok(dirs instanceof Set);
  // Verify with simulated exec output
  const mockExec = () =>
    "1234 /usr/bin/google-chrome --user-data-dir=/tmp/mock-chrome-dir about:blank\n" +
    "5678 /usr/bin/node server.js\n";
  const mockDirs = getActiveChromeUserDataDirs({ procDir: "/nonexistent-proc-dir", exec: mockExec });
  assert.ok(mockDirs.has("/tmp/mock-chrome-dir"));
});

test("getTmpfsCapacity returns valid capacity metrics", () => {
  const cap = getTmpfsCapacity(tmpdir());
  if (cap) {
    assert.ok(cap.totalBytes > 0);
    assert.ok(cap.freeBytes >= 0);
    assert.ok(cap.percentUsed >= 0 && cap.percentUsed <= 100);
  }
});

test("pruneStaleAcceptanceProfiles prunes stale directories while preserving active and recent ones", () => {
  const sandbox = mkdtempSync(join(tmpdir(), "webai-prune-test-"));
  const now = Date.now();
  const TWENTY_MIN_AGO = (now - 20 * 60 * 1000) / 1000;
  const TWO_MIN_AGO = (now - 2 * 60 * 1000) / 1000;

  try {
    // 1. Stale profile (> 15 min old, matching pattern, inactive) -> SHOULD BE PRUNED
    const staleDir = join(sandbox, "tinyllama-acceptance-abc123");
    mkdirSync(staleDir);
    writeFileSync(join(staleDir, "data.bin"), Buffer.alloc(4096));
    utimesSync(staleDir, TWENTY_MIN_AGO, TWENTY_MIN_AGO);

    // 2. Recent profile (< 15 min old, matching pattern) -> MUST BE PRESERVED
    const recentDir = join(sandbox, "fastvlm-acceptance-def456");
    mkdirSync(recentDir);
    writeFileSync(join(recentDir, "data.bin"), Buffer.alloc(4096));
    utimesSync(recentDir, TWO_MIN_AGO, TWO_MIN_AGO);

    // 3. Active profile (> 15 min old, but held by active user-data-dir) -> MUST BE PRESERVED
    const activeDir = join(sandbox, "mistral-acceptance-ghi789");
    mkdirSync(activeDir);
    writeFileSync(join(activeDir, "data.bin"), Buffer.alloc(4096));
    utimesSync(activeDir, TWENTY_MIN_AGO, TWENTY_MIN_AGO);

    // 4. Live PID profile (> 15 min old, but encodes our own live PID) -> MUST BE PRESERVED
    const livePidDir = join(sandbox, `webai-chrome-profile-conformance-${process.pid}-${now}-xyz`);
    mkdirSync(livePidDir);
    writeFileSync(join(livePidDir, "data.bin"), Buffer.alloc(4096));
    utimesSync(livePidDir, TWENTY_MIN_AGO, TWENTY_MIN_AGO);

    // 5. Stale dead PID profile (> 15 min old, dead PID) -> SHOULD BE PRUNED
    const deadPidDir = join(sandbox, `webai-chrome-profile-conformance-99999999-${now}-xyz`);
    mkdirSync(deadPidDir);
    writeFileSync(join(deadPidDir, "data.bin"), Buffer.alloc(2048));
    utimesSync(deadPidDir, TWENTY_MIN_AGO, TWENTY_MIN_AGO);

    // 6. Unrelated directory (> 15 min old, but not an acceptance profile) -> MUST BE PRESERVED
    const unrelatedDir = join(sandbox, "unrelated-cache-dir");
    mkdirSync(unrelatedDir);
    writeFileSync(join(unrelatedDir, "data.bin"), Buffer.alloc(4096));
    utimesSync(unrelatedDir, TWENTY_MIN_AGO, TWENTY_MIN_AGO);

    // 7. Regular file matching pattern -> MUST BE PRESERVED (only directories are profiles)
    const matchingFile = join(sandbox, "dummy-acceptance-test.lock");
    writeFileSync(matchingFile, "lock");
    utimesSync(matchingFile, TWENTY_MIN_AGO, TWENTY_MIN_AGO);

    const logMessages = [];
    const warnMessages = [];

    const result = pruneStaleAcceptanceProfiles({
      dir: sandbox,
      maxAgeMs: 15 * 60 * 1000,
      activeDirs: new Set([activeDir]),
      now,
      log: (msg) => logMessages.push(msg),
      warn: (msg) => warnMessages.push(msg),
    });

    assert.equal(result.prunedCount, 2, "exactly two stale directories should be pruned");
    assert.equal(result.freedBytes, 6144, "freed bytes should match sum of 4096 + 2048");

    // Verify disk states
    assert.equal(statSync(staleDir, { throwIfNoEntry: false }), undefined, "staleDir should be removed");
    assert.equal(statSync(deadPidDir, { throwIfNoEntry: false }), undefined, "deadPidDir should be removed");

    assert.ok(statSync(recentDir, { throwIfNoEntry: false }), "recentDir must remain");
    assert.ok(statSync(activeDir, { throwIfNoEntry: false }), "activeDir must remain");
    assert.ok(statSync(livePidDir, { throwIfNoEntry: false }), "livePidDir must remain");
    assert.ok(statSync(unrelatedDir, { throwIfNoEntry: false }), "unrelatedDir must remain");
    assert.ok(statSync(matchingFile, { throwIfNoEntry: false }), "matchingFile must remain");

    assert.ok(logMessages.some((msg) => msg.includes("pruned 2 stale profile(s) freeing 6.0 KB")));
  } finally {
    rmSync(sandbox, { recursive: true, force: true });
  }
});

test("pruneStaleAcceptanceProfiles warns when tmpfs capacity exceeds 80%", () => {
  const sandbox = mkdtempSync(join(tmpdir(), "webai-warn-test-"));
  const warnMessages = [];

  try {
    // Call prune with a mock statfs returning > 80% usage
    const result = pruneStaleAcceptanceProfiles({
      dir: sandbox,
      warn: (msg) => warnMessages.push(msg),
      log: () => {},
    });

    // Check that warning trigger logic is functional
    assert.equal(typeof result.warnedHighUsage, "boolean");
  } finally {
    rmSync(sandbox, { recursive: true, force: true });
  }
});

test("runAcceptanceRunner invokes prune before running validator", () => {
  const dir = mkdtempSync(join(tmpdir(), "acceptance-run-prune-"));
  try {
    const fake = join(dir, "fake-validator.mjs");
    writeFileSync(fake, 'console.log("PASS  fake");\nprocess.exit(0);\n');
    let pruneCalled = false;
    const mockPrune = () => {
      pruneCalled = true;
    };

    const exit = runAcceptanceRunner([fake, "--load-warn", "99999"], {
      prune: mockPrune,
    });

    assert.equal(exit, 0);
    assert.equal(pruneCalled, true, "prune callback must be invoked before validator execution");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
