// fdy: CHROME_BIN must not be trusted on non-emptiness alone. A wrong value used to report
// chromeAvailable() true, so the honest skip never fired and the suite died in the 240s spawn-ENOENT
// retry storm that cj5 removed. These tests are path logic only — they never launch a browser, so the
// suite stays browser-free (enforced separately by suite-stays-browser-free.test.mjs).
import { mkdtempSync, rmSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import assert from "node:assert/strict";
import test from "node:test";
import { chromeAvailable, resolveChromeBinary } from "../scripts/browser.mjs";

const withChromeBin = (value, fn) => {
  const saved = process.env.CHROME_BIN;
  try {
    if (value === undefined) delete process.env.CHROME_BIN;
    else process.env.CHROME_BIN = value;
    return fn();
  } finally {
    if (saved === undefined) delete process.env.CHROME_BIN;
    else process.env.CHROME_BIN = saved;
  }
};

test("a nonexistent CHROME_BIN is rejected, never returned as the binary", () => {
  const dir = mkdtempSync(join(tmpdir(), "chrome-bin-"));
  try {
    const bogus = join(dir, "does-not-exist");
    withChromeBin(bogus, () => {
      const { binary, rejected } = resolveChromeBinary();
      assert.notEqual(binary, bogus, "a path that does not exist must not be handed back as the binary");
      assert.equal(rejected, bogus, "the rejected value must be reported so the operator can see it");
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a directory is rejected — being executable is not enough, it must be a file", () => {
  const dir = mkdtempSync(join(tmpdir(), "chrome-bin-"));
  try {
    withChromeBin(dir, () => {
      const { binary, rejected } = resolveChromeBinary();
      assert.notEqual(binary, dir);
      assert.equal(rejected, dir);
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a non-executable file is rejected", () => {
  const dir = mkdtempSync(join(tmpdir(), "chrome-bin-"));
  try {
    const file = join(dir, "chrome-not-executable");
    writeFileSync(file, "#!/bin/sh\n");
    chmodSync(file, 0o644);
    withChromeBin(file, () => {
      const { binary, rejected } = resolveChromeBinary();
      assert.notEqual(binary, file);
      assert.equal(rejected, file);
    });
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("a valid CHROME_BIN still wins over the PATH search", () => {
  // process.execPath is a real, runnable file on every platform — the declare-the-dependency use case.
  withChromeBin(process.execPath, () => {
    const { binary, rejected } = resolveChromeBinary();
    assert.equal(binary, process.execPath);
    assert.equal(rejected, null);
  });
});

test("whitespace-only CHROME_BIN counts as unset rather than a rejection", () => {
  withChromeBin("   ", () => {
    const { rejected } = resolveChromeBinary();
    assert.equal(rejected, null, "blank is not a wrong value, it is an absent one");
  });
});

test("a rejected CHROME_BIN is announced once, naming the value", () => {
  const dir = mkdtempSync(join(tmpdir(), "chrome-bin-"));
  const warnings = [];
  const original = console.warn;
  try {
    const bogus = join(dir, "nope");
    console.warn = (msg) => warnings.push(String(msg));
    withChromeBin(bogus, () => {
      resolveChromeBinary();
      resolveChromeBinary();
    });
  } finally {
    console.warn = original;
    rmSync(dir, { recursive: true, force: true });
  }
  assert.equal(warnings.length, 1, "the operator is told once, not once per call");
  assert.match(warnings[0], /is not an executable file/, "the warning must say what is wrong");
});

test("chromeAvailable() agrees with the resolved binary", () => {
  // The invariant behind the bug: these two answers must never disagree.
  assert.equal(chromeAvailable(), resolveChromeBinary().binary !== null);
});
