#!/usr/bin/env node
// cap-evidence/acceptance runner with a load precondition and abort evidence (bead web-ai-showcase-50s).
//
// Deep acceptance suites (validators) drive multi-hundred-MB WASM stages for 15-30 minutes. On this
// shared box, heavy lanes push load average past 50 and a single stalled `Runtime.evaluate` used to
// abort the run with NO record — losing every route check it had already collected and leaving tracked
// screenshots half-rewritten. This runner wraps a validator so that:
//
//   1. the box load is printed, is a warning above --load-warn (default 30), and is a REFUSAL to start
//      above --max-load (opt-in), so a lane does not burn an hour on a doomed run;
//   2. the validator runs with CDP_EVALUATE_RETRIES (default 3) so transient evaluate stalls retry
//      inside the shared CDP client instead of killing the run (the lever every validator already
//      goes through — no per-family edit, so no acceptance re-run is triggered);
//   3. on a non-zero exit the runner parses the validator's PASS/FAIL lines and writes a DIAGNOSTIC
//      failing record (exitCode 1, aborted: true, every check that did pass) to the family's run
//      record path, so check-portfolio-acceptance still fails on the family but the evidence and the
//      failure point survive (scripts/browser.mjs writeAbortedAcceptanceRun).
//
// Usage:
//   node scripts/acceptance-run.mjs scripts/validate-<slug>.mjs --write-run
//   node scripts/acceptance-run.mjs scripts/validate-<slug>.mjs --max-load 25 --load-warn 20
//   node scripts/acceptance-run.mjs /abs/path/validator.mjs --record /tmp/run.json --retries 5
//   node scripts/acceptance-run.mjs scripts/validate-<slug>.mjs -- --slug foo   # extra args after --
//
// Exit codes: the validator's own code; 2 = bad usage/missing validator; 3 = refused on load.

import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  captureHeadCommit,
  loadAverage,
  loadWarning,
  repoRoot,
  writeAbortedAcceptanceRun,
} from "./browser.mjs";

/** Strip ANSI so parsed check lines do not carry colour codes. */
export function stripAnsi(text) {
  return String(text ?? "").replace(/\u001b\[[0-9;]*m/g, "");
}

/**
 * Parse the house check lines a validator prints: `PASS  name — detail` / `FAIL  name — detail`.
 * Only these lines are evidence; retry/progress noise is ignored by construction.
 */
export function parseCheckLines(output) {
  const assertions = [];
  for (const raw of stripAnsi(output).split("\n")) {
    const line = raw.trimEnd();
    const match = /^(PASS|FAIL)\s{1,3}(.+?)(?:\s+—\s+(.*))?$/.exec(line);
    if (!match) continue;
    assertions.push({
      name: match[2].trim().slice(0, 300),
      state: match[1] === "PASS" ? "pass" : "fail",
      ...(match[3] ? { detail: match[3].slice(0, 400) } : {}),
    });
  }
  return assertions;
}

/** Derive a family's run-record path from its validator (`scripts/validate-<slug>.mjs`). */
export function resolveRunRecordPath(validatorPath, override = null) {
  if (override) return isAbsolute(override) ? override : join(repoRoot, override);
  const slug = String(validatorPath).replace(/^.*\//, "").replace(/^validate-/, "").replace(/\.mjs$/, "");
  const manifest = join(repoRoot, "models", slug, "acceptance.json");
  if (existsSync(manifest)) {
    try {
      const parsed = JSON.parse(readFileSync(manifest, "utf8"));
      if (parsed?.runRecord) return join(repoRoot, parsed.runRecord);
    } catch { /* fall through to the convention */ }
  }
  return join(repoRoot, "models", slug, "acceptance-run.json");
}

const USAGE =
  "usage: node scripts/acceptance-run.mjs <validator.mjs> [--write-run] [--max-load N] [--load-warn N] [--record path] [--retries N] [-- extra...]";

/** Split runner flags from the validator's own arguments. `--` passes everything after it through. */
export function parseRunnerArgs(argv) {
  const out = { validator: null, passthrough: [], maxLoad: null, loadWarn: 30, record: null, retries: 3 };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--") {
      out.passthrough.push(...argv.slice(i + 1));
      break;
    }
    if (arg === "--write-run") {
      out.passthrough.push(arg);
      continue;
    }
    if (arg === "--max-load" || arg === "--load-warn" || arg === "--record" || arg === "--retries") {
      const value = argv[++i];
      if (value === undefined) throw new Error(`${arg} needs a value`);
      if (arg === "--max-load") out.maxLoad = Number(value);
      else if (arg === "--load-warn") out.loadWarn = Number(value);
      else if (arg === "--record") out.record = value;
      else out.retries = Number(value);
      continue;
    }
    if (arg.startsWith("-")) throw new Error(`unknown flag ${arg}`);
    if (out.validator === null) out.validator = arg;
    else out.passthrough.push(arg);
  }
  return out;
}

export function runAcceptanceRunner(argv, { env = process.env, spawn = spawnSync } = {}) {
  let args;
  try {
    args = parseRunnerArgs(argv);
  } catch (error) {
    console.error(`[acceptance-run] ${error.message}\n${USAGE}`);
    return 2;
  }
  if (!args.validator) {
    console.error(`[acceptance-run] no validator given\n${USAGE}`);
    return 2;
  }
  const validator = isAbsolute(args.validator) ? args.validator : join(repoRoot, args.validator);
  if (!existsSync(validator)) {
    console.error(`[acceptance-run] validator not found: ${validator}`);
    return 2;
  }

  const load = loadAverage();
  console.log(
    `[acceptance-run] box load ${load.one.toFixed(1)} / ${load.five.toFixed(1)} / ${load.fifteen.toFixed(1)} (1/5/15m) · ${args.validator}`,
  );
  if (args.maxLoad !== null && Number.isFinite(args.maxLoad) && load.one > args.maxLoad) {
    console.error(
      `[acceptance-run] refusing to start: load ${load.one.toFixed(1)} > --max-load ${args.maxLoad}. ` +
        "Deep suites stall under shared-box load; re-run in a quieter window or raise the limit deliberately.",
    );
    return 3;
  }
  const warning = loadWarning(Number.isFinite(args.loadWarn) ? args.loadWarn : 30, load);
  if (warning) console.warn(`[acceptance-run] WARNING: ${warning}`);

  const startCommit = captureHeadCommit();
  const retries = Number.isFinite(args.retries) && args.retries > 0 ? Math.floor(args.retries) : 0;
  console.log(`[acceptance-run] CDP_EVALUATE_RETRIES=${env.CDP_EVALUATE_RETRIES ?? retries}`);

  const result = spawn(process.execPath, [validator, ...args.passthrough], {
    cwd: repoRoot,
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    env: { ...env, CDP_EVALUATE_RETRIES: env.CDP_EVALUATE_RETRIES ?? String(retries) },
  });
  const stdout = result.stdout ?? "";
  const stderr = result.stderr ?? "";
  if (stdout) process.stdout.write(stdout);
  if (stderr) process.stderr.write(stderr);

  const assertions = parseCheckLines(`${stdout}\n${stderr}`);
  const exitCode = typeof result.status === "number" ? result.status : 1;
  console.log(
    `[acceptance-run] validator exit ${exitCode}; parsed ${assertions.length} check line(s)`,
  );

  if (exitCode !== 0) {
    const runRecordPath = resolveRunRecordPath(args.validator, args.record);
    const wrote = writeAbortedAcceptanceRun({
      runRecordPath,
      startCommit,
      reason: `validator ${args.validator} exited ${exitCode}`,
      assertions,
      notes: [`command: node ${args.validator} ${args.passthrough.join(" ")}`.trim()],
    });
    console.log(
      wrote
        ? `[acceptance-run] diagnostic record written: ${runRecordPath}`
        : "[acceptance-run] diagnostic record NOT written (HEAD moved during the run)",
    );
  } else if (args.record) {
    console.log("[acceptance-run] exit 0 — the validator owns its run record; nothing written by the runner");
  }
  return exitCode;
}

const isMain = typeof process !== "undefined" && process.argv?.[1]
  ? resolve(process.argv[1]) === fileURLToPath(import.meta.url)
  : false;

if (isMain) process.exit(runAcceptanceRunner(process.argv.slice(2)));
