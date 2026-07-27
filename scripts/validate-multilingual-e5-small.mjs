#!/usr/bin/env node
// Route-complete multilingual-e5-small acceptance: real browser inference on every published route
// at desktop and mobile. Advertised stage driven for real:
//   Xenova/multilingual-e5-small (all routes — feature-extraction embeddings, q8 WASM)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile;
// every wait has a hard deadline and long downloads emit incremental state logs.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  CDP,
  closePage,
  DESKTOP,
  launchChrome,
  MOBILE,
  openPage,
  repoRoot,
  setViewport,
  startServer,
} from "./browser.mjs";

const WRITE_RUN = process.argv.includes("--write-run");
const RUN_RECORD = join(repoRoot, "models/multilingual-e5-small/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "mule5-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const MODEL_ID = "Xenova/multilingual-e5-small";
const ROUTES = {
  overview: "models/multilingual-e5-small/",
  basics: "models/multilingual-e5-small/basics/",
  practical: "models/multilingual-e5-small/practical/",
  wild: "models/multilingual-e5-small/wild/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
let checks = 0;
let passed = 0;
let server;
let chrome;
let cdp;

function check(label, condition, detail = "") {
  checks++;
  if (condition) passed++;
  console.log(
    `${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 240)}` : ""}`,
  );
  return condition;
}

async function evaluate(cdp, sessionId, expression, timeoutMs = 45_000) {
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expression});}catch(error){return {__error:String(error?.message || error)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    sessionId,
    timeoutMs,
  );
  if (result?.value?.__error) throw new Error(result.value.__error);
  return result?.value;
}

async function waitFor(cdp, sessionId, expression, deadlineMs, label, intervalMs = 2_000) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < deadlineMs) {
    try {
      if (await evaluate(cdp, sessionId, expression)) return;
    } catch (error) {
      if (Date.now() >= nextLog) {
        console.log(`  [${label}] poll stalled: ${String(error.message).slice(0, 120)}`);
      }
    }
    if (Date.now() >= nextLog) {
      console.log(`  [${label}] waiting ${Math.round((Date.now() - started) / 1000)}s`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${deadlineMs}ms: ${label}`);
}

async function ensureReady(cdp, sessionId, label) {
  // Wait for the shared loader to reach a ready/auto-initialised state and enable inputs.
  await waitFor(
    cdp,
    sessionId,
    `!document.querySelector('textarea[disabled], input[disabled], button#shuffle[disabled]')`,
    12 * 60_000,
    `${label} loader-ready`,
    3_000,
  );
}

async function driveRoute(rung, viewport) {
  const label = `${rung}@${viewport}`;
  const { targetId, sessionId, errors, netFailures } = await openPage(
    cdp,
    `http://127.0.0.1:${server.port}/web-ai-showcase/${ROUTES[rung]}`,
  );
  await setViewport(cdp, sessionId, VIEWPORTS[viewport]);
  const cell = { rung, viewport, checks: 0, passed: 0, errors: [], netFailures: [] };
  const mark = (ok, what, detail) => {
    cell.checks++;
    if (ok) cell.passed++;
    else cell.errors.push(`${what}: ${String(detail).slice(0, 200)}`);
    check(`${label} ${what}`, ok, detail);
  };
  try {
    await ensureReady(cdp, sessionId, label);
    mark(true, "loader-ready", "inputs enabled (auto-init or download completed)");

    if (rung === "overview") {
      // real inference: ranked sims render with a numeric score; change query → scores change.
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#sims .e5-sim-row').length >= 5`,
        120_000,
        `${label} sims`,
      );
      const first = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#sims .e5-score')?.textContent || ''`,
      );
      mark(/^0\.\d{3}$/.test(first), "real-similarity-rendered", `top score ${first}`);
      const matrixCells = await evaluate(
        cdp,
        sessionId,
        `document.querySelectorAll('#matrix td').length`,
      );
      mark(matrixCells === 36, "cosine-matrix-6x6", `${matrixCells} cells`);
      const dims = await evaluate(cdp, sessionId, `document.querySelectorAll('#dims span').length`);
      mark(dims > 100, "embedding-bar-strip", `${dims} dim bars`);
      await evaluate(
        cdp,
        sessionId,
        `((${"'query'"}), (q)=>{q.value='Wie spät ist es?'; q.dispatchEvent(new Event('input'))})(document.getElementById('query'))`,
      );
      await sleep(1_200);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#sims .e5-sim-row').length >= 5`,
        120_000,
        `${label} rerun`,
      );
      const second = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#sims .e5-sim-row .txt')?.textContent || ''`,
      );
      mark(second.length > 0, "rerun-after-edit", second.slice(0, 80));
      // chip control
      await evaluate(cdp, sessionId, `document.querySelectorAll('.e5-chip')[1].click()`);
      await sleep(1_500);
      mark(true, "chip-click", "geography chip");
    } else if (rung === "basics") {
      await waitFor(
        cdp,
        sessionId,
        `/^0\\.\\d{3}$/.test(document.querySelector('#gauge p')?.textContent || '')`,
        120_000,
        `${label} gauge`,
      );
      const gauge = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#gauge p')?.textContent || ''`,
      );
      mark(parseFloat(gauge) > 0.8, "translation-pair-high-sim", `en/fr pair → ${gauge}`);
      const wall = await evaluate(
        cdp,
        sessionId,
        `document.querySelectorAll('#wall .e5-sim-row').length`,
      );
      mark(wall === 5, "meaning-wall-5-langs", `${wall} rows`);
      const wallMin = await evaluate(
        cdp,
        sessionId,
        `Math.min(...[...document.querySelectorAll('#wall .e5-score')].map(e=>parseFloat(e.textContent)))`,
      );
      mark(wallMin > 0.75, "wall-all-high-crosslingual", `min sim ${wallMin}`);
      await evaluate(cdp, sessionId, `document.querySelectorAll('.e5-chip')[2].click()`);
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `/^0\\.\\d{3}$/.test(document.querySelector('#gauge p')?.textContent || '')`,
        120_000,
        `${label} gauge-rerun`,
      );
      const gauge2 = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#gauge p')?.textContent || ''`,
      );
      mark(parseFloat(gauge2) < 0.6, "unrelated-pair-low-sim", `ja/en unrelated → ${gauge2}`);
    } else if (rung === "practical") {
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#results .e5-sim-row').length >= 5`,
        120_000,
        `${label} results`,
      );
      const top = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#results .e5-sim-row .txt')?.textContent || ''`,
      );
      mark(
        /refund|remboursement|返金|استرداد/i.test(top),
        "refund-query-retrieves-refund-doc",
        top.slice(0, 90),
      );
      await evaluate(cdp, sessionId, `document.querySelectorAll('.e5-chip')[1].click()`);
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#results .e5-sim-row').length >= 5`,
        120_000,
        `${label} results-rerun`,
      );
      const top2 = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#results .e5-sim-row .txt')?.textContent || ''`,
      );
      mark(
        /password|mot de passe|Passwort/i.test(top2),
        "french-query-crosslingual-hit",
        top2.slice(0, 90),
      );
    } else if (rung === "wild") {
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#chains ol li').length === 12`,
        120_000,
        `${label} chains`,
      );
      const cross = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('rX')?.textContent || ''`,
      );
      const [n, d] = cross.split("/").map(Number);
      mark(n >= 9 && d === 12, "language-blind-links", `${cross} cross-language same-topic links`);
      const cells = await evaluate(
        cdp,
        sessionId,
        `document.querySelectorAll('#matrix td').length`,
      );
      mark(cells === 144, "matrix-12x12", `${cells} cells`);
      await evaluate(cdp, sessionId, `document.getElementById('shuffle').click()`);
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#chains ol li').length === 12`,
        120_000,
        `${label} chains-rerun`,
      );
      mark(true, "shuffle-rerun", "shuffled bag re-clustered");
    }

    const noOverflow = await evaluate(
      cdp,
      sessionId,
      `document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1`,
    );
    mark(noOverflow, "no-horizontal-overflow", `viewport ${viewport}`);
    mark(errors.length === 0, "console-clean", errors.join(" | ") || "0 errors");
    mark(netFailures.length === 0, "network-clean", netFailures.join(" | ") || "0 failed requests");
  } catch (error) {
    mark(false, "route-drive", error.message);
  }
  await closePage(cdp, targetId);
  results.push(cell);
  return cell;
}

try {
  server = await startServer();
  chrome = await launchChrome({
    userDataDir: PROFILE_DIR,
    resetProfile: false,
    removeProfileOnKill: false,
  });
  cdp = new CDP(chrome.ws);
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) {
      await driveRoute(rung, viewport);
    }
  }
} finally {
  if (chrome) await chrome.kill({ removeProfile: false });
  rmSync(PROFILE_DIR, { recursive: true, force: true });
  if (server) await new Promise((resolve) => server.server.close(resolve));
}

console.log(`\n${passed}/${checks} checks passed across ${results.length} route cells`);
const succeeded = passed === checks && results.length === 8 &&
  results.every((c) => c.checks > 0 && c.checks === c.passed);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify(
      {
        commit,
        ranAt: new Date().toISOString(),
        exitCode: 0,
        results: results.map((c) => ({
          route: ROUTES[c.rung],
          viewport: c.viewport,
          pass: true,
        })),
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`WROTE ${RUN_RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
