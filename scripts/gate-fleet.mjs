#!/usr/bin/env node
// Tracked fleet gate entrypoint (bead web-ai-showcase-j7s, ADR web-ai-showcase-cxw).
//
// WHY THIS EXISTS
// ~/.fleet/check.conf is host-local, unversioned, and hand-maintained: its CHECK_CMD ran the nine
// static stages and ZERO of the test files, while CI (.github/workflows/gate.yml) — tracked —
// already enforced the same statics PLUS the full Node suite PLUS the Deno server suite. Two
// gate definitions diverged, and the weaker one (the fleet's) drifted by hand-edit. This script is
// the ONE tracked definition of the fleet gate chain; the host check.conf delegates to it
// (CHECK_CMD="node scripts/gate-fleet.mjs", CHECK_FAST_CMD="node scripts/gate-fleet.mjs --fast"),
// so a reprovisioned VM inherits the gate from the repo instead of from a stale local file.
//
// THE CHAIN (mirrors gate.yml exactly; CI order)
//   DEFAULT: deps precondition -> 9 static stages -> node --test "test/**/*.test.mjs" (the 48-file
//            default suite: no browser, no network) -> deno task test:server.
//   --fast:  deps precondition -> 9 static stages -> node scripts/test-changed.mjs (the fail-closed
//            changed-test router, bead 360: unmapped/deleted/unresolvable anything falls back to
//            the whole unit suite — never a green "nothing to run"; heavy tests are deferred with a
//            visible line, browser-facing changes are REPORTED as BROWSER_UNVERIFIED per jdz).
//   --with-portfolio appends check-portfolio-acceptance.mjs after the statics (OFF by default: it
//            is RED on main per BLOCKED bead web-ai-showcase-9je — six inherited failures that are
//            no branch's fault. RE-INSTATE the stage by default when 9je closes; the exclusion and
//            this flag are the documented, dated contract from ~/.fleet/check.conf, now tracked).
//   --list prints the chain without executing anything (no deps check, no spawns) — for humans,
//            grep, and the unit tests.
//
// CONTRACTS
//   * NEVER installs dependencies. The deps precondition only CHECKS that node_modules exists
//     (the suite's parser tests import pinned acorn/parse5) and fails closed naming the right
//     installer for the context: `fleet-deps` on fleet VMs (hard-links the lockfile store),
//     `npm ci --ignore-scripts` in CI. npm ci inside a worktree is forbidden by fleet policy.
//   * NO Chrome anywhere: test-browser/ stays outside this chain (jdz cgroup containment BLOCKED);
//     the scheduled adversarial tier (deno task test:adversarial) is separately wired in gate.yml.
//   * Child env strips NODE_TEST_CONTEXT: when this script itself runs under `node --test` (its own
//     unit tests), an inherited NODE_TEST_CONTEXT makes a child `node --test` skip every file and
//     exit 0 — a vacuous green. Same guard as scripts/test-changed.mjs's childEnv().
//   * Stage argv is passed to spawnSync WITHOUT a shell and WITHOUT literal quote characters: the
//     quoted-glob discipline ("test/**/*.test.mjs") is a SHELL requirement; here the glob is one
//     bare argv element expanded by Node's own test runner.
//   * A missing `deno` binary is a named fail-closed diagnostic, not an ENOENT stack trace.
//
// Exit code: 0 all stages green; otherwise the first failing stage's exit code (or 1).

import { spawnSync } from "node:child_process";
import { existsSync, realpathSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** The nine static stages, in .github/workflows/gate.yml order (portfolio-acceptance excluded — 9je). */
export const STATIC_STAGES = [
  { id: "routes", label: "Route regression gate (durable-demo contract)", cmd: "node", args: ["scripts/check-routes.mjs"] },
  { id: "conformance", label: "Conformance + parity gate (immutable suites, schema, coverage)", cmd: "node", args: ["scripts/check-conformance.mjs"] },
  { id: "lineage", label: "Lineage gate (identity preservation, reproducibility)", cmd: "node", args: ["scripts/check-lineage.mjs"] },
  { id: "image-provenance", label: "Image provenance gate (fail-closed ledger)", cmd: "node", args: ["scripts/check-image-provenance.mjs"] },
  { id: "audio-provenance", label: "Audio provenance gate (fail-closed ledger)", cmd: "node", args: ["scripts/check-audio-provenance.mjs"] },
  { id: "download-inventory", label: "Download-route inventory gate (every downloading route classified)", cmd: "node", args: ["scripts/check-download-inventory.mjs"] },
  { id: "download-adoption", label: "Download-component adoption gate (no loader bypasses)", cmd: "node", args: ["scripts/check-download-adoption.mjs"] },
  { id: "ladder-links", label: "Ladder-link gate (in-family links resolve)", cmd: "node", args: ["scripts/check-ladder-links.mjs"] },
  { id: "currency", label: "Currency & runtime pin gate (unauthorized pins, catalogue drift)", cmd: "node", args: ["scripts/audit-model-currency.mjs", "--check"] },
];

/** Opt-in only: RED on main until bead web-ai-showcase-9je closes (see header). */
export const PORTFOLIO_STAGE = {
  id: "portfolio-acceptance",
  label: "Portfolio acceptance (OPT-IN: red on main per BLOCKED 9je until re-instated)",
  cmd: "node",
  args: ["scripts/check-portfolio-acceptance.mjs"],
};

export const SUITE_STAGE = {
  id: "node-suite",
  label: 'Node default test suite (node --test "test/**/*.test.mjs" — no browser, no network)',
  cmd: "node",
  args: ["--test", "test/**/*.test.mjs"],
};

export const DENO_STAGE = {
  id: "deno-server",
  label: "Deno server test suite (same-origin proxy/redirect invariants)",
  cmd: "deno",
  args: ["task", "test:server"],
};

export const ROUTER_STAGE = {
  id: "changed-router",
  label: "Changed-test router (fail-closed fast tier, bead 360)",
  cmd: "node",
  args: ["scripts/test-changed.mjs"],
};

/** Parse CLI arguments. Unknown flags throw (fail-closed): a gate must not shrug off a typo. */
export function parseArgs(argv) {
  const out = { mode: "default", withPortfolio: false, list: false, help: false };
  for (const a of argv) {
    if (a === "--fast") out.mode = "fast";
    else if (a === "--with-portfolio") out.withPortfolio = true;
    else if (a === "--list") out.list = true;
    else if (a === "--help" || a === "-h") out.help = true;
    else throw new Error(`unknown argument: ${a} (expected --fast | --with-portfolio | --list | --help)`);
  }
  return out;
}

/** The full chain for one invocation. Statics first (CI order), then the tier-specific tail. */
export function buildChain(opts) {
  const chain = [...STATIC_STAGES];
  if (opts.withPortfolio) chain.push(PORTFOLIO_STAGE);
  if (opts.mode === "fast") {
    chain.push(ROUTER_STAGE);
  } else {
    chain.push(SUITE_STAGE, DENO_STAGE);
  }
  return chain;
}

/**
 * Environment for child stages. NODE_TEST_CONTEXT is STRIPPED: when this script runs under
 * `node --test` (its own unit tests do), the variable makes a child `node --test` skip every file
 * and exit 0 — a vacuous green (the same trap scripts/test-changed.mjs guards against).
 */
export function childEnv(base = process.env) {
  const env = { ...base, GIT_TERMINAL_PROMPT: "0" };
  delete env.NODE_TEST_CONTEXT;
  return env;
}

/**
 * Deps precondition: node_modules must exist (the suite's parser tests import pinned acorn/parse5).
 * Returns null when satisfied, or a fail-closed message naming the context-correct installer.
 * This script NEVER installs anything.
 */
export function checkDeps(root = ROOT) {
  if (existsSync(join(root, "node_modules"))) return null;
  return (
    "node_modules is missing: the test suite imports pinned parser deps (acorn/parse5). " +
    "Install with `fleet-deps` on fleet VMs (hard-links the lockfile store; never npm ci in a " +
    "worktree) or `npm ci --ignore-scripts` in CI, then re-run the gate."
  );
}

function runStage(stage, index, total) {
  const banner = `[gate-fleet] stage ${index + 1}/${total}: ${stage.label}`;
  console.log(`\n${banner}`);
  const t0 = Date.now();
  const res = spawnSync(stage.cmd, stage.args, { cwd: ROOT, stdio: "inherit", env: childEnv() });
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  if (res.error) {
    if (res.error.code === "ENOENT") {
      console.error(
        `\n[gate-fleet] FAIL-CLOSED at stage ${index + 1}/${total} (${stage.id}): ` +
          `'${stage.cmd}' is not on PATH. This stage declares that dependency — see deno.json ` +
          `tasks and .github/workflows/gate.yml for how CI provisions it.`,
      );
      return 1;
    }
    console.error(`\n[gate-fleet] FAIL at stage ${index + 1}/${total} (${stage.id}): ${res.error.message}`);
    return 1;
  }
  if (res.status !== 0) {
    const how = res.status !== null ? `exit ${res.status}` : `signal ${res.signal}`;
    console.error(`\n[gate-fleet] FAIL at stage ${index + 1}/${total} (${stage.id}) after ${secs}s: ${how}`);
    return res.status ?? 1;
  }
  console.log(`[gate-fleet] ok (${secs}s)`);
  return 0;
}

const HELP = `gate-fleet — tracked fleet gate entrypoint (web-ai-showcase-j7s)

  node scripts/gate-fleet.mjs                 full chain: deps check, 9 static stages,
                                              node --test "test/**/*.test.mjs", deno task test:server
  node scripts/gate-fleet.mjs --fast          fast tier: 9 static stages + the fail-closed
                                              changed-test router (scripts/test-changed.mjs)
  node scripts/gate-fleet.mjs --with-portfolio also run check-portfolio-acceptance.mjs (opt-in;
                                              red on main per BLOCKED bead 9je until it closes)
  node scripts/gate-fleet.mjs --list          print the chain, run nothing

No Chrome (test-browser/ is jdz-gated), no network, no installs (deps are a precondition).`;

function main() {
  let opts;
  try {
    opts = parseArgs(process.argv.slice(2));
  } catch (e) {
    console.error(String(e.message));
    console.error(HELP);
    process.exitCode = 2;
    return;
  }
  if (opts.help) {
    console.log(HELP);
    return;
  }
  const chain = buildChain(opts);
  if (opts.list) {
    console.log(`[gate-fleet] chain (${opts.mode}${opts.withPortfolio ? " +portfolio" : ""}):`);
    chain.forEach((s, i) => console.log(`  ${i + 1}. ${s.id.padEnd(20)} ${s.cmd} ${s.args.join(" ")}`));
    return;
  }
  const depsError = checkDeps();
  if (depsError) {
    console.error(`\n[gate-fleet] FAIL-CLOSED (deps precondition): ${depsError}`);
    process.exitCode = 1;
    return;
  }
  const t0 = Date.now();
  for (let i = 0; i < chain.length; i++) {
    const code = runStage(chain[i], i, chain.length);
    if (code !== 0) {
      process.exitCode = code;
      return;
    }
  }
  const secs = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n[gate-fleet] PASS — ${chain.length} stage(s) in ${secs}s (${opts.mode} tier).`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === realpathSync(process.argv[1])) main();
