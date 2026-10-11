// Unit tests for the tracked fleet gate entrypoint (bead web-ai-showcase-j7s).
//
// These test the CHAIN CONSTRUCTION and its contracts without running any stage:
//   * the static prefix is exactly the nine CI stages, in .github/workflows/gate.yml order, with
//     portfolio-acceptance absent unless --with-portfolio (the 9je exclusion is a tracked contract);
//   * the default chain runs the whole default Node suite + the Deno server suite (no browser,
//     ever — test-browser/ is jdz-gated and lives outside this chain);
//   * the fast chain ends with the fail-closed changed-test router (bead 360);
//   * the child environment strips NODE_TEST_CONTEXT — a child `node --test` that inherits it
//     skips every file and exits 0 (vacuous green), the exact trap scripts/test-changed.mjs guards;
//   * the deps precondition names the context-correct installer and never installs;
//   * --list prints the chain and runs nothing; unknown flags fail closed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  STATIC_STAGES,
  PORTFOLIO_STAGE,
  SUITE_STAGE,
  DENO_STAGE,
  ROUTER_STAGE,
  parseArgs,
  buildChain,
  childEnv,
  checkDeps,
} from "../scripts/gate-fleet.mjs";

const SCRIPT = join(dirname(fileURLToPath(import.meta.url)), "..", "scripts", "gate-fleet.mjs");
const ROOT = join(dirname(SCRIPT), "..");

/** The exact nine static stages CI runs, in order (source of truth: .github/workflows/gate.yml). */
const CI_STATICS = [
  ["scripts/check-routes.mjs"],
  ["scripts/check-conformance.mjs"],
  ["scripts/check-lineage.mjs"],
  ["scripts/check-image-provenance.mjs"],
  ["scripts/check-audio-provenance.mjs"],
  ["scripts/check-download-inventory.mjs"],
  ["scripts/check-download-adoption.mjs"],
  ["scripts/check-ladder-links.mjs"],
  ["scripts/audit-model-currency.mjs", "--check"],
];

test("static prefix is exactly the nine CI stages, in gate.yml order", () => {
  assert.equal(STATIC_STAGES.length, 9, "nine static stages, no more, no fewer");
  assert.deepEqual(
    STATIC_STAGES.map((s) => s.args),
    CI_STATICS,
    "the static chain must mirror .github/workflows/gate.yml step-for-step",
  );
  for (const s of STATIC_STAGES) assert.equal(s.cmd, "node");
});

test("portfolio-acceptance is excluded by default and only reachable via --with-portfolio (9je)", () => {
  const flat = (chain) => chain.map((s) => s.args[0]).join(" ");
  assert.ok(!flat(buildChain({ mode: "default" })).includes("portfolio"), "default excludes portfolio");
  assert.ok(!flat(buildChain({ mode: "fast" })).includes("portfolio"), "fast excludes portfolio");
  assert.ok(
    flat(buildChain({ mode: "default", withPortfolio: true })).includes("check-portfolio-acceptance.mjs"),
    "--with-portfolio adds it",
  );
  assert.equal(PORTFOLIO_STAGE.args[0], "scripts/check-portfolio-acceptance.mjs");
  assert.match(PORTFOLIO_STAGE.label, /9je/, "the stage label must keep naming the red-main reason");
});

test("default chain: statics -> full default Node suite -> Deno server suite; never a browser", () => {
  const chain = buildChain({ mode: "default" });
  assert.equal(chain.length, 11);
  assert.equal(chain[9], SUITE_STAGE);
  assert.equal(chain[10], DENO_STAGE);
  assert.deepEqual(SUITE_STAGE.args, ["--test", "test/**/*.test.mjs"]);
  assert.ok(!SUITE_STAGE.args.some((a) => a.includes('"')), "no literal quote characters in argv (spawn has no shell)");
  assert.ok(!SUITE_STAGE.args[1].includes("test-browser"), "the suite stage must stay browser-free");
  assert.equal(DENO_STAGE.args.join(" "), "task test:server");
});

test("fast chain: statics -> fail-closed changed-test router; no suite, no deno duplication", () => {
  const chain = buildChain({ mode: "fast" });
  assert.equal(chain.length, 10);
  assert.equal(chain[9], ROUTER_STAGE);
  assert.deepEqual(ROUTER_STAGE.args, ["scripts/test-changed.mjs"]);
  assert.ok(!chain.includes(SUITE_STAGE), "the fast tier routes; it does not run the whole suite by default");
});

test("with-portfolio inserts the stage after the statics, before the tier tail", () => {
  const chain = buildChain({ mode: "default", withPortfolio: true });
  assert.equal(chain.length, 12);
  assert.equal(chain[9], PORTFOLIO_STAGE);
  assert.equal(chain[10], SUITE_STAGE);
});

test("parseArgs: known flags, defaults, and fail-closed on unknown", () => {
  assert.deepEqual(parseArgs([]), { mode: "default", withPortfolio: false, list: false, help: false });
  assert.equal(parseArgs(["--fast"]).mode, "fast");
  assert.equal(parseArgs(["--with-portfolio"]).withPortfolio, true);
  assert.equal(parseArgs(["--list"]).list, true);
  assert.equal(parseArgs(["--help"]).help, true);
  assert.throws(() => parseArgs(["--fastt"]), /unknown argument/, "a gate must not shrug off a typo");
});

test("childEnv strips NODE_TEST_CONTEXT (vacuous-green guard) and keeps the rest", () => {
  const env = childEnv({ NODE_TEST_CONTEXT: "child-victim", PATH: "/usr/bin", CUSTOM: "keep" });
  assert.ok(!("NODE_TEST_CONTEXT" in env), "NODE_TEST_CONTEXT must never reach a child node --test");
  assert.equal(env.PATH, "/usr/bin");
  assert.equal(env.CUSTOM, "keep");
  assert.equal(env.GIT_TERMINAL_PROMPT, "0");
});

test("checkDeps: satisfied in a repo with node_modules; fail-closed message names the installer elsewhere", () => {
  assert.equal(checkDeps(), null, "this checkout has node_modules (fleet-deps store)");
  const empty = mkdtempSync(join(tmpdir(), "gate-fleet-deps-"));
  try {
    const msg = checkDeps(empty);
    assert.ok(msg, "an empty dir must fail the precondition");
    assert.match(msg, /fleet-deps/, "fleet VMs get the fleet-deps instruction");
    assert.match(msg, /npm ci --ignore-scripts/, "CI gets the npm ci instruction");
    assert.match(msg, /node_modules/, "the missing thing is named");
  } finally {
    rmSync(empty, { recursive: true, force: true });
  }
});

test("--list prints the chain and executes nothing (safe under a test runner)", () => {
  // NODE_TEST_CONTEXT is set the way a parent `node --test` would leave it: --list must still work,
  // and its output is stage lines, not stage EXECUTION banners.
  const res = spawnSync(process.execPath, [SCRIPT, "--list"], {
    encoding: "utf8",
    env: { ...process.env, NODE_TEST_CONTEXT: "child-victim" },
    timeout: 30_000,
  });
  assert.equal(res.status, 0, res.stderr);
  assert.match(res.stdout, /\[gate-fleet\] chain \(default\)/);
  assert.match(res.stdout, /node-suite/);
  assert.match(res.stdout, /deno-server/);
  assert.ok(!res.stdout.includes("stage 1/11:"), "--list must not execute stages");
  const fast = spawnSync(process.execPath, [SCRIPT, "--list", "--fast"], { encoding: "utf8", timeout: 30_000 });
  assert.equal(fast.status, 0, fast.stderr);
  assert.match(fast.stdout, /chain \(fast\)/);
  assert.match(fast.stdout, /changed-router/);
  assert.ok(!fast.stdout.includes("node-suite"), "the fast chain must not list the full suite");
});

test("unknown CLI flag exits non-zero with the help text (fail-closed invocation)", () => {
  const res = spawnSync(process.execPath, [SCRIPT, "--bogus"], { encoding: "utf8", timeout: 30_000 });
  assert.notEqual(res.status, 0);
  assert.match(res.stdout + res.stderr, /unknown argument/);
});

test("the suite stage argv matches what CI runs byte-for-byte (single source of truth)", () => {
  // gate.yml: node --test "test/**/*.test.mjs" — the quotes are SHELL syntax; argv carries the bare
  // glob, which Node's own test runner expands.
  const ci = execFileSync(
    "grep",
    ["-c", 'node --test "test/\\*\\*/\\*.test\\.mjs"', ".github/workflows/gate.yml"],
    { cwd: ROOT, encoding: "utf8" },
  ).trim();
  assert.ok(Number(ci) >= 1, "CI's quoted-glob step is the reference invocation");
});
