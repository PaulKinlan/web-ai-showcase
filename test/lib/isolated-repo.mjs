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
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
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
  return dir;
}
