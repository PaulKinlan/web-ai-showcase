// Shared isolation helper for tests that mutate a repository file and then run the currency gate.
//
// `node --test` runs test FILES in parallel, so a test that rewrites sw.js / the allowlist /
// models/* in place races with the other pin tests and, on a SIGKILL or timeout, leaves those
// tracked files corrupted forever (the finally never ran). This helper materialises a clean,
// immutable copy of the committed HEAD and lets the caller mutate and run the gate there instead.
//
// HEAD (unlike the live working tree, which other test FILES mutate in parallel) is a stable tree,
// so two mutating tests can never capture one another's half-written files. Running `--check` from
// the copy exercises the audit's `import.meta.url`-relative root resolution (everything resolves
// against the copy, not this checkout).
//
// OVERLAY — the file under test is the WORKING-TREE copy of the audit script, not the committed one:
//   scripts/audit-model-currency.mjs
// It is overlaid onto the isolated copy here, in this helper, so the two mutating isolation tests
// (test/runtime-pins.test.mjs and test/runtime-integrity-exemption.test.mjs) can NEVER diverge in what
// they exercise. Before this overlay moved into the helper, runtime-pins overlaid the working tree
// while runtime-integrity-exemption silently ran the COMMITTED HEAD script, so an uncommitted edit to
// the audit was judged as if it were absent by one of the two tests. In CI, where the test and the
// script land in the same commit, the overlay is a no-op; during local development it makes every
// mutation be judged by the CURRENT source rather than the last committed snapshot.
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

/** Shell-quote a value for interpolation into a single shell pipeline. */
export const shq = (s) => `'${String(s).replaceAll("'", "'\\''")}'`;

/**
 * Materialise a clean, immutable copy of the repository from the committed HEAD.
 * The caller is responsible for removing the returned directory when done.
 */
export function materializeIsolatedRepo() {
  const dir = mkdtempSync(join(tmpdir(), "web-ai-currency-"));
  execSync(`git archive HEAD | tar -x -C ${shq(dir)}`, {
    cwd: ROOT,
    stdio: "pipe",
    maxBuffer: 64 * 1024 * 1024,
  });
  assert.ok(
    existsSync(join(dir, "scripts/audit-model-currency.mjs")),
    "the isolated copy must contain the audit script",
  );
  // Overlay the working-tree audit script (the file under test) onto the committed snapshot. See the
  // OVERLAY note above for why this lives in the helper rather than in the individual tests.
  copyFileSync(
    join(ROOT, "scripts/audit-model-currency.mjs"),
    join(dir, "scripts/audit-model-currency.mjs"),
  );
  return dir;
}
