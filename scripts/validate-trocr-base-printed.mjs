#!/usr/bin/env node
// Route-complete trocr-base-printed acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real:
//   Xenova/trocr-base-printed (all routes — printed-line OCR, q8 WASM, ~340 MB)
//   Xenova/m2m100_418M        (multi-model — English → target-language translation)
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
const RUN_RECORD = join(repoRoot, "models/trocr-base-printed/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "trocr-base-printed-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const OCR_ID = "Xenova/trocr-base-printed";
const TR_ID = "Xenova/m2m100_418M";
const ROUTES = {
  overview: "models/trocr-base-printed/",
  basics: "models/trocr-base-printed/basics/",
  practical: "models/trocr-base-printed/practical/",
  wild: "models/trocr-base-printed/wild/",
  multimodel: "models/trocr-base-printed/multi-model/",
};
const EXPECTED = { overview: 3, basics: 3, practical: 4, wild: 3, multimodel: 3 };
const TOTAL = Object.values(EXPECTED).reduce((a, b) => a + b, 0) * 2;
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

async function ensureReady(cdp, sessionId, expectedCount, label) {
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
    if (snapshot.length === expectedCount && snapshot.every((item) => item.state === "ready")) {
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
      // Let CDP receive this evaluation result before model download/initialisation can saturate
      // the renderer. The next ensureReady poll observes the real state transition.
      setTimeout(() => buttons.forEach((button) => button.click()), 0);
      return buttons.length;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model download/init`);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, rung === "multimodel" ? 2 : 1, `${viewport} ${rung}`);

  if (rung === "overview") {
    // Default sample (Grand Total receipt line) is preselected; click Read and verify real output.
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      180_000,
      `${viewport} overview inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#textOut')?.textContent || '',
      tokens: document.querySelectorAll('#trace .tok').length,
      backend: document.querySelector('#rBackend')?.textContent || '',
      tps: document.querySelector('#rTps')?.textContent || ''
    })`,
    );
    check(
      `${viewport} overview: real ${OCR_ID} transcription + see-inside token trace`,
      evidence.out.includes("GRAND TOTAL") && evidence.out.includes("42.99") &&
        evidence.tokens > 5 && /WASM/.test(evidence.backend),
      JSON.stringify(evidence),
    );
  } else if (rung === "basics") {
    // Default sample preselected; Read runs the real model on the cropped line.
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      180_000,
      `${viewport} basics inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#textOut')?.textContent || '',
      tok: document.querySelector('#rTok')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || ''
    })`,
    );
    check(
      `${viewport} basics: real line read from ${OCR_ID}`,
      evidence.out.includes("GRAND TOTAL") && Number(evidence.tok) > 3 &&
        /ms/.test(evidence.ms),
      JSON.stringify(evidence),
    );
  } else if (rung === "practical") {
    // Workflow 1 — receipt digitiser: pick the Grand Total line sample, read it into the list.
    await evaluate(
      cdp,
      sid,
      `(() => { [...document.querySelectorAll('#samples .sample-thumb')].find((t) => t.dataset.src.includes('sample-total')).click(); return true; })()`,
    );
    await sleep(500);
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelectorAll('#rows .row-line').length >= 1`,
      180_000,
      `${viewport} practical receipt line`,
    );
    const rowsEvidence = await evaluate(
      cdp,
      sid,
      `({ rows: document.querySelector('#rows')?.textContent || '', n: document.querySelectorAll('#rows .row-line input').length })`,
    );
    check(
      `${viewport} practical: receipt line digitised into editable list`,
      rowsEvidence.n >= 1 && rowsEvidence.rows.includes("GRAND TOTAL"),
      JSON.stringify(rowsEvidence).slice(0, 220),
    );
    // Workflow 2 — serial capture + format validation on the bundled label photo.
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#readSerial').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#serialCheck')?.hidden === false`,
      180_000,
      `${viewport} practical serial read`,
    );
    const serialEvidence = await evaluate(
      cdp,
      sid,
      `({ code: document.querySelector('#serialOut')?.textContent || '', fmt: document.querySelector('#rFmt')?.textContent || '' })`,
    );
    check(
      `${viewport} practical: serial captured + format-checked`,
      /SN/.test(serialEvidence.code) && /9021/.test(serialEvidence.code) &&
        /VALID/.test(serialEvidence.fmt),
      JSON.stringify(serialEvidence).slice(0, 220),
    );
  } else if (rung === "wild") {
    // Headless has no camera: drive the honest bundled-photo fallback (same real read path).
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#readSample').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      180_000,
      `${viewport} wild fallback inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      out: document.querySelector('#textOut')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || ''
    })`,
    );
    check(
      `${viewport} wild: real bundled-photo read (camera fallback) from ${OCR_ID}`,
      evidence.out.includes("PLATFORM 9") && /WASM/.test(evidence.backend),
      JSON.stringify(evidence),
    );
  } else {
    // Multi-model: OCR the preselected fox line, then translate it to French with m2m100.
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      300_000,
      `${viewport} ${OCR_ID} → ${TR_ID}`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      read: document.querySelector('#textOut')?.textContent || '',
      trans: document.querySelector('#transOut')?.textContent || '',
      ocr: document.querySelector('#rOcr')?.textContent || '',
      tr: document.querySelector('#rTr')?.textContent || ''
    })`,
    );
    check(
      `${viewport} multi-model: ${OCR_ID} then ${TR_ID} both ran`,
      /QUICK|BROWN/i.test(evidence.read) && evidence.trans.length > 5 &&
        !/QUICK BROWN FOX JUMPS/.test(evidence.trans) && /ms/.test(evidence.ocr) &&
        /ms/.test(evidence.tr),
      JSON.stringify(evidence).slice(0, 220),
    );
  }

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
  const url = (route) => `http://127.0.0.1:${started.port}/web-ai-showcase/${route}`;

  // Each cell gets a fresh process tree so released route workers cannot starve later CDP or WASM
  // work. The validator-owned profile persists only between cells, proving cached auto-init without
  // retaining renderer memory.
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
        cell.pass = passed - before === EXPECTED[rung];
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

const succeeded = checks === TOTAL && passed === checks && results.length === 10 &&
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
