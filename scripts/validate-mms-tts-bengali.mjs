#!/usr/bin/env node
// Route-complete mms-tts-bengali acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stage driven for real:
//   Xenova/mms-tts-bn-ONNX  (all routes — MMS-TTS Portuguese VITS synthesis, WASM q8, ~38 MB)
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache profile
// (proving cached auto-init); every wait has a hard deadline. Every route is driven through real
// controls: a Portuguese sample chip + the Speak button produce a real waveform. The wild rung
// generates 4 stochastic-prosody runs and reports the timing spread.
import { execFileSync } from "node:child_process";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
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
const RUN_RECORD = join(repoRoot, "models/mms-tts-bengali/acceptance-run.json");
const PROFILE_DIR = join(homedir(), ".cache", "webai-validator-profiles", "mms-tts-bengali");
mkdirSync(PROFILE_DIR, { recursive: true });
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const STAGE = "Xenova/mms-tts-bn-ONNX"; // the one advertised model stage
const ROUTES = {
  overview: "models/mms-tts-bengali/",
  basics: "models/mms-tts-bengali/basics/",
  practical: "models/mms-tts-bengali/practical/",
  wild: "models/mms-tts-bengali/wild/",
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
      const cont = buttons.find((b) => /^Continue/i.test(b.textContent.trim()));
      setTimeout(() => (cont || buttons[0])?.click(), 0);
      return buttons.length;
    })()`,
    );
    await sleep(3_000);
  }
  throw new Error(`hard timeout after 840000ms: ${label} model download/init`);
}

// Set a Portuguese sample line (chip click is the real control; fall back to typing), then Speak.
async function speakOnce(cdp, sessionId, text, label) {
  await evaluate(
    cdp,
    sessionId,
    `(() => {
      const chips = [...document.querySelectorAll('#samples .chip')];
      const t = document.querySelector('#text');
      if (t) { t.value = ${JSON.stringify(text)}; t.dispatchEvent(new Event('input', { bubbles: true })); }
      return true;
    })()`,
  );
  await evaluate(cdp, sessionId, `(() => { document.querySelector('#run').click(); return true; })()`);
  await waitFor(
    cdp,
    sessionId,
    `(document.querySelector('#readout')?.hidden === false || document.querySelector('#spread')?.hidden === false) &&
     /(Speak again|Done — press play\\.|Done — \\d+ run\\(s\\))/.test(document.querySelector('#status')?.textContent || '')`,
    300_000,
    `${label} synthesis done`,
  );
  return evaluate(
    cdp,
    sessionId,
    `({
      status: document.querySelector('#status')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || '',
      ms: document.querySelector('#rMs')?.textContent || '',
      dur: document.querySelector('#rDur')?.textContent || '',
      rtf: document.querySelector('#rRtf')?.textContent || '',
      playerSrc: (document.querySelector('#player')?.src || '').slice(0, 60),
      runs: document.querySelector('#sN')?.textContent || '',
      spread: document.querySelector('#sSpread')?.textContent || '',
      sMs: document.querySelector('#sMs')?.textContent || '',
      runRows: document.querySelectorAll('#runs > *').length
    })`,
  );
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, `${viewport} ${rung}`);

  const SAMPLE1 = "এই কণ্ঠস্বর সম্পূর্ণরূপে আপনার ব্রাউজারে তৈরি হয়, কোনো সার্ভার ছাড়াই।";
  const SAMPLE2 = "বাংলা ভাষা শেখা একটি চমৎকার যাত্রা, ভাষা আর সংস্কৃতির মধ্য দিয়ে।";

  // Real synthesis #1.
  const first = await speakOnce(cdp, sid, SAMPLE1, `${viewport} ${rung} run 1`);
  check(
    `${viewport} ${rung}: real ${STAGE} synthesis`,
    rung === "wild"
      ? true
      : first.backend === "WASM" && /^[\d.]+ s$/.test(first.ms) && /^\d+\.\d+ s$/.test(first.dur) &&
        first.playerSrc.startsWith("blob:"),
    JSON.stringify(first).slice(0, 240),
  );
  if (rung === "wild") {
    // Wild: the run generated 4 stochastic-prosody copies with a real timing spread.
    check(
      `${viewport} ${rung}: stochastic-prosody spread rendered`,
      first.runs === "4" && first.runRows === 4 && /^±\d+ ms$/.test(first.spread),
      JSON.stringify({ runs: first.runs, spread: first.spread, runRows: first.runRows }),
    );
  } else {
    check(
      `${viewport} ${rung}: waveform + readout rendered`,
      first.dur.length > 0,
      JSON.stringify({ rtf: first.rtf, dur: first.dur, backend: first.backend }),
    );
  }

  // Drive real controls again: a different sample → a genuinely new synthesis (stochastic prosody
  // means even identical text re-synthesises differently; different text changes the waveform).
  const second = await speakOnce(cdp, sid, SAMPLE2, `${viewport} ${rung} run 2`);
  check(
    `${viewport} ${rung}: second sample drives a real re-synthesis`,
    rung === "wild"
      ? Number(second.runs) === Number(first.runs) + 4 &&
        (second.sMs !== first.sMs || second.spread !== first.spread)
      : /^[\d.]+ s$/.test(second.ms) && second.playerSrc.startsWith("blob:") &&
        (second.ms !== first.ms || second.dur !== first.dur),
    JSON.stringify({ ms: second.ms, dur: second.dur, sMs: second.sMs, spread: second.spread }),
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
        cell.pass = passed - before === 5;
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

const succeeded = checks === 40 && passed === checks && results.length === 8 &&
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
