#!/usr/bin/env node
// Route-complete mobilebert-nli acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stage driven for real:
//   Xenova/mobilebert-uncased-mnli  (all routes — MNLI text-classification, WASM int8, 25MB)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile
// (proving cached auto-init); every wait has a hard deadline and the ?auto hook downloads + runs the
// model on ready. Every route is driven through real controls: a sample chip + the Judge button.
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
const RUN_RECORD = join(repoRoot, "models/mobilebert-nli/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "mobilebert-nli-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const STAGE = "Xenova/mobilebert-uncased-mnli"; // the one advertised model stage
const ROUTES = {
  overview: "models/mobilebert-nli/",
  basics: "models/mobilebert-nli/basics/",
  practical: "models/mobilebert-nli/practical/",
  wild: "models/mobilebert-nli/wild/",
};
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const results = [];
let checks = 0;
let passed = 0;
let server;
let chrome;

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

const loaderSnapshot =
  `JSON.stringify([...document.querySelectorAll('.model-loader')].map((loader) => ({
  state: loader.dataset.state || '',
  status: (loader.querySelector('.status')?.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 180),
  buttons: [...loader.querySelectorAll('button')].map((button) => button.textContent.trim())
})))`;

async function ensureReady(cdp, sessionId, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 12 * 60_000) {
    const snapshot = JSON.parse(await evaluate(cdp, sessionId, loaderSnapshot));
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(snapshot)}`,
      );
      nextLog = Date.now() + 8_000;
    }
    if (snapshot.length === 1 && snapshot.every((item) => item.state === "ready")) {
      return snapshot;
    }
    await evaluate(
      cdp,
      sessionId,
      `(() => {
      const buttons = [...document.querySelectorAll('.model-loader')].flatMap((loader) =>
        [...loader.querySelectorAll('button')].filter((item) =>
          /Download|Retry|Re-download/i.test(item.textContent) && !item.disabled
        )
      );
      setTimeout(() => buttons.forEach((button) => button.click()), 0);
      return buttons.length;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model download/init`);
}

const resultsSnapshot = `({
  rows: document.querySelectorAll('#results .nli-row').length,
  top: document.querySelector('#results .nli-row.top .label')?.textContent || '',
  backend: document.querySelector('#rBackend')?.textContent || '',
  ms: document.querySelector('#rMs')?.textContent || '',
  out: document.querySelector('#results')?.textContent || ''
})`;

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  // Pages with a ?auto hook already ran the default pair on ready; the ladder pages did not. Only
  // click Judge when the readout is not yet populated (avoids a redundant click racing the auto-run).
  const alreadyRan = await evaluate(cdp, sid, `document.querySelector('#readout')?.hidden === false`);
  if (!alreadyRan) {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
  }
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false`,
    180_000,
    `${viewport} ${rung} default inference`,
  );
  const first = await evaluate(cdp, sid, resultsSnapshot);
  check(
    `${viewport} ${rung}: real ${STAGE} MNLI inference (3-way)`,
    first.rows === 3 && /ENTAILMENT|NEUTRAL|CONTRADICTION/i.test(first.top) &&
      /WASM/.test(first.backend) && /\d/.test(first.ms),
    JSON.stringify(first).slice(0, 200),
  );

  // Drive real controls: click the middle sample chip (contradiction/negation/swap) + Judge → the
  // hypothesis changes and the verdict must change. (mobilebert-uncased-mnli is a small model — we
  // assert the output changed, not a specific label.)
  const hyp0 = await evaluate(cdp, sid, `document.querySelector('#hypothesis')?.value || ''`);
  await evaluate(
    cdp,
    sid,
    `(() => { const chip = document.querySelectorAll('#samples .chip')[1]; if (chip) chip.click(); return !!chip; })()`,
  );
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#hypothesis')?.value !== ${JSON.stringify(hyp0)}`,
    15_000,
    `${viewport} ${rung} chip applied`,
    500,
  );
  await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
  // Bounded retry: a click can land while the previous run's finally is re-enabling the button.
  const rerunExpr = `(document.querySelector('#results')?.textContent || '') !== ${JSON.stringify(first.out)}`;
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      await waitFor(cdp, sid, rerunExpr, 30_000, `${viewport} ${rung} contradiction re-run`);
      break;
    } catch (error) {
      if (attempt === 1) throw error;
      await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    }
  }
  const second = await evaluate(cdp, sid, resultsSnapshot);
  check(
    `${viewport} ${rung}: contradiction chip + Judge drive real re-inference`,
    second.rows === 3 && second.out !== first.out,
    JSON.stringify(second).slice(0, 200),
  );

  const hygiene = await evaluate(
    cdp,
    sid,
    `({ overflow:document.documentElement.scrollWidth-window.innerWidth, named:[...document.querySelectorAll('button')].every((b)=>(b.textContent||b.getAttribute('aria-label')||'').trim()) })`,
  );
  check(
    `${viewport} ${rung}: responsive controls`,
    hygiene.overflow <= 1 && hygiene.named,
    JSON.stringify(hygiene),
  );
  check(
    `${viewport} ${rung}: console/network clean`,
    page.errors.length === 0 && page.netFailures.length === 0,
    JSON.stringify({ errors: page.errors, network: page.netFailures }),
  );
}

try {
  const started = await startServer();
  server = started.server;
  const url = (route) => `http://127.0.0.1:${started.port}/web-ai-showcase/${route}?auto`;

  for (const viewport of ["desktop", "mobile"]) {
    for (const [rung, route] of Object.entries(ROUTES)) {
      const cell = { route, viewport, pass: false };
      results.push(cell);
      let cdp;
      let page;
      try {
        console.log(`\n=== ${viewport} × ${rung}: ${route} ===`);
        chrome = await launchChrome({
          userDataDir: PROFILE_DIR,
          resetProfile: false,
          removeProfileOnKill: false,
        });
        cdp = new CDP(chrome.ws);
        page = await openPage(cdp, url(route));
        const before = passed;
        await exercise(cdp, page, rung, viewport);
        cell.pass = passed - before === 4;
      } catch (error) {
        console.log(`FAIL  ${viewport} ${rung}: ${String(error.stack || error).slice(0, 500)}`);
      } finally {
        if (page) await closePage(cdp, page.targetId);
        if (chrome) await chrome.kill({ removeProfile: false });
        chrome = null;
      }
    }
  }
} finally {
  console.log(`\n${passed}/${checks} checks passed`);
  console.log(`ROUTE-RESULTS-JSON: ${JSON.stringify(results)}`);
  if (chrome) await chrome.kill({ removeProfile: false });
  if (server) await new Promise((resolve) => server.close(resolve));
  rmSync(PROFILE_DIR, { recursive: true, force: true });
}

const succeeded = checks === 32 && passed === checks && results.length === 8 &&
  results.every((item) => item.pass);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify({ commit, ranAt: new Date().toISOString(), exitCode: 0, results }, null, 2) +
      "\n",
  );
  console.log(`WROTE ${RUN_RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
