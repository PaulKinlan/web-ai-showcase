#!/usr/bin/env node
// Route-complete protein-mutation-oracle acceptance: real browser inference on the published overview
// route at desktop and mobile. Advertised stage driven for real:
//   Xenova/esm2_t12_35M_UR50D  (overview — ESM-2 masked-LM zero-shot variant-effect scoring, WASM q8, ~35 MB)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile
// (proving cached auto-init); every wait has a hard deadline and the loader auto-initialises on ready.
// Every route is driven through real controls: a sample chip + the Analyse button.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
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
const RUN_RECORD = join(repoRoot, "models/protein-mutation-oracle/acceptance-run.json");
const PROFILE_DIR = join(homedir(), ".cache", "webai-validator-profiles", "protein-mutation-oracle");
mkdirSync(PROFILE_DIR, { recursive: true });
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const STAGE = "Xenova/esm2_t12_35M_UR50D"; // the one advertised model stage
// Single published rung (overview) — the family's acceptance.json declares overview only.
const ROUTES = {
  overview: "models/protein-mutation-oracle/",
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
  while (Date.now() - started < 14 * 60_000) {
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
          /Download|Retry|Re-download|Continue/i.test(item.textContent) && !item.disabled
        )
      );
      // In check-timeout the Retry button re-enters the same stalled check while Continue is the
      // honest forward path (it verifies the cache / downloads); prefer Continue when present.
      const cont = buttons.find((b) => /^Continue/i.test(b.textContent.trim()));
      setTimeout(() => (cont || buttons[0])?.click(), 0);
      return buttons.length;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 840000ms: ${label} model download/init`);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  // The page pre-loads hemoglobin β and auto-runs the sickle-cell position (E6) once the model is
  // ready. Wait for that auto scan to render its readout.
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false &&
     (document.querySelector('#rBackend')?.textContent || '').length > 0`,
    300_000,
    `${viewport} ${rung} auto scan readout`,
  );
  const first = await evaluate(
    cdp,
    sid,
    `({
      backend: document.querySelector('#rBackend')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || '',
      pos: document.querySelector('#rPos')?.textContent || '',
      bars: document.querySelectorAll('#preds .pg-bar-row').length,
      muts: document.querySelectorAll('#muts .pg-mut-row').length,
      sub: document.querySelector('#predSub')?.textContent || ''
    })`,
  );
  check(
    `${viewport} ${rung}: real ${STAGE} masked-LM scan rendered`,
    first.backend === "WASM" && /^\d+ ms$/.test(first.ms) && first.pos.length >= 2 &&
      first.bars === 8 && first.muts === 20 && /wild-type is [A-Z]/.test(first.sub),
    JSON.stringify(first).slice(0, 220),
  );

  // Drive real controls: switch to the ubiquitin sample, then Analyse → a NEW scan of position 1 (M1).
  const seq0 = await evaluate(cdp, sid, `document.querySelector('#seqin')?.value || ''`);
  await evaluate(
    cdp,
    sid,
    `(() => { const chip = [...document.querySelectorAll('#samples .pg-btn')].find((b) => /ubiquitin/i.test(b.textContent)); if (chip) chip.click(); return !!chip; })()`,
  );
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#seqin')?.value !== ${JSON.stringify(seq0)}`,
    15_000,
    `${viewport} ${rung} sample chip applied`,
    500,
  );
  await evaluate(cdp, sid, `(() => { document.querySelector('#analyseBtn').click(); return true; })()`);
  await waitFor(
    cdp,
    sid,
    `document.querySelector('#readout')?.hidden === false &&
     (document.querySelector('#rPos')?.textContent || '') !== ${JSON.stringify(first.pos)} &&
     /^[A-Z]\\d+$/.test(document.querySelector('#rPos')?.textContent || '')`,
    300_000,
    `${viewport} ${rung} second scan`,
  );
  const second = await evaluate(
    cdp,
    sid,
    `({
      pos: document.querySelector('#rPos')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || '',
      bars: document.querySelectorAll('#preds .pg-bar-row').length,
      muts: document.querySelectorAll('#muts .pg-mut-row').length
    })`,
  );
  check(
    `${viewport} ${rung}: sample chip + Analyse drive a real second scan`,
    second.pos !== first.pos && /^\d+ ms$/.test(second.ms) && second.bars === 8 && second.muts === 20,
    JSON.stringify(second).slice(0, 220),
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

}

const succeeded = checks === 8 && passed === checks && results.length === 2 &&
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
