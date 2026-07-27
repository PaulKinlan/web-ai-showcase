#!/usr/bin/env node
// Route-complete pitch-detection acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real are niobures/CREPE on every route and
// onnx-community/whisper-tiny.en on multi-model. Each cell owns a fresh Chrome process tree while
// the validator-owned cache profile persists between cells. Every wait has a hard deadline and
// model downloads emit incremental state logs.
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
const RUN_RECORD = join(repoRoot, "models/pitch-detection/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "pitch-detection-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const CREPE_ID = "niobures/CREPE";
const ASR_ID = "onnx-community/whisper-tiny.en";
const ROUTES = {
  overview: "models/pitch-detection/",
  basics: "models/pitch-detection/basics/",
  practical: "models/pitch-detection/practical/",
  wild: "models/pitch-detection/wild/",
  multimodel: "models/pitch-detection/multi-model/",
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

async function waitFor(cdp, sessionId, expression, deadlineMs, label, intervalMs = 1_000) {
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
        [...loader.querySelectorAll('button')].filter((item) => /Download|Retry|Re-download/i.test(item.textContent) && !item.disabled)
      );
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
  await setViewport(cdp, sid, viewport === "mobile" ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, rung === "multimodel" ? 2 : 1, `${viewport} ${rung}`);

  if (rung === "overview") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      120_000,
      `${viewport} overview inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      median:document.querySelector('#rMedian')?.textContent || '',
      frames:Number(document.querySelector('#rFrames')?.textContent || 0),
      pitchgram:document.querySelector('#pgram')?.width || 0,
      curve:document.querySelector('#pcurve')?.width || 0,
      table:document.querySelectorAll('#noteTable tbody tr').length
    })`,
    );
    check(
      `${viewport} overview: real ${CREPE_ID} output + see-inside`,
      /Hz/.test(evidence.median) && evidence.frames > 0 && evidence.pitchgram > 0 &&
        evidence.curve > 0 && evidence.table > 0,
      JSON.stringify(evidence),
    );
  } else if (rung === "basics") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#result')?.hidden === false`,
      120_000,
      `${viewport} basics inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ hz:document.querySelector('#rHz')?.textContent || '', note:document.querySelector('#rNote')?.textContent || '', confidence:document.querySelector('#rConf')?.textContent || '' })`,
    );
    check(
      `${viewport} basics: real note detection`,
      /Hz/.test(evidence.hz) && evidence.note.length > 1 && Number(evidence.confidence) > 0,
      JSON.stringify(evidence),
    );
  } else if (rung === "practical") {
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#samples button').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `!/^[–-]$/.test(document.querySelector('#rHz')?.textContent || '–')`,
      120_000,
      `${viewport} practical streaming inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ hz:document.querySelector('#rHz')?.textContent || '', confidence:document.querySelector('#rConf')?.textContent || '', backend:document.querySelector('#rBackend')?.textContent || '' })`,
    );
    check(
      `${viewport} practical: real live-tuner stream`,
      /Hz/.test(evidence.hz) && Number(evidence.confidence) > 0 && /wasm/i.test(evidence.backend),
      JSON.stringify(evidence),
    );
  } else if (rung === "wild") {
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      150_000,
      `${viewport} wild inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ notes:document.querySelectorAll('#notes .notechip').length, latency:document.querySelector('#rMs')?.textContent || '', backend:document.querySelector('#rBackend')?.textContent || '', curve:document.querySelector('#pcurve')?.width || 0 })`,
    );
    check(
      `${viewport} wild: real melody transcription`,
      evidence.notes > 0 && /ms$/.test(evidence.latency) && /wasm/i.test(evidence.backend) &&
        evidence.curve > 0,
      JSON.stringify(evidence),
    );
  } else {
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#run')?.disabled === false`,
      45_000,
      `${viewport} rights-cleared sample decode`,
    );
    await evaluate(cdp, sid, `(() => { document.querySelector('#run').click(); return true; })()`);
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      240_000,
      `${viewport} ${CREPE_ID} + ${ASR_ID}`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({ transcript:document.querySelector('#transcript')?.textContent || '', median:document.querySelector('#rMedian')?.textContent || '', voiced:document.querySelector('#rVoiced')?.textContent || '', backend:document.querySelector('#rBackend')?.textContent || '', sampleCount:document.querySelectorAll('#samples button').length })`,
    );
    check(
      `${viewport} multi-model: ${CREPE_ID} and ${ASR_ID} both ran`,
      evidence.transcript.trim().length > 8 && /Hz/.test(evidence.median) &&
        /%/.test(evidence.voiced) && /pitch.*asr/i.test(evidence.backend) &&
        evidence.sampleCount === 1,
      JSON.stringify(evidence),
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
  for (const viewport of ["desktop", "mobile"]) {
    for (const [rung, route] of Object.entries(ROUTES)) {
      const cell = { route, viewport, pass: false };
      results.push(cell);
      // One bounded fresh-process retry is allowed only when a transient CDP/startup failure occurs
      // before this cell records any checks. Never duplicate or hide a failed assertion.
      for (let attempt = 1; attempt <= 2 && !cell.pass; attempt++) {
        let cdp;
        let page;
        const before = passed;
        try {
          console.log(
            `\n=== ${viewport} × ${rung}: ${route}${
              attempt > 1 ? " (fresh-process retry)" : ""
            } ===`,
          );
          chrome = await launchChrome({
            userDataDir: PROFILE_DIR,
            resetProfile: false,
            removeProfileOnKill: false,
          });
          cdp = new CDP(chrome.ws);
          page = await openPage(cdp, url(route));
          await exercise(cdp, page, rung, viewport);
          cell.pass = passed - before === 3;
        } catch (error) {
          console.log(`FAIL  ${viewport} ${rung}: ${String(error.stack || error).slice(0, 500)}`);
          if (passed !== before) break;
        } finally {
          if (page) await closePage(cdp, page.targetId);
          if (chrome) await chrome.kill({ removeProfile: false });
          chrome = null;
          await sleep(1_000);
        }
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

const succeeded = checks === 30 && passed === checks && results.length === 10 &&
  results.every((item) => item.pass);
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" })
    .trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify({ commit, ranAt: new Date().toISOString(), exitCode: 0, results }, null, 2) +
      "\n",
  );
  console.log(`WROTE ${RUN_RECORD} for ${commit}`);
}
process.exit(succeeded ? 0 : 1);
