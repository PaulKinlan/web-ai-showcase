#!/usr/bin/env node
// Route-complete Whisper acceptance: every published route, desktop + mobile, real local inference.
// Advertised stages driven for real:
//   onnx-community/whisper-base_timestamped (all routes, q8 WASM in the test browser)
//   onnx-community/Kokoro-82M-v1.0-ONNX (multi-model speak-back stage)
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
const RUN_RECORD = join(repoRoot, "models/whisper-speech-to-text/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "whisper-acceptance-"));
const ROUTES = {
  overview: "models/whisper-speech-to-text/",
  basics: "models/whisper-speech-to-text/basics/",
  practical: "models/whisper-speech-to-text/practical/",
  wild: "models/whisper-speech-to-text/wild/",
  multi: "models/whisper-speech-to-text/multi-model/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const cells = [];
let checks = 0;
let passed = 0;
let server;
let chrome;
let cdp;

function check(label, ok, detail = "") {
  checks++;
  if (ok) passed++;
  console.log(
    `${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${String(detail).slice(0, 220)}` : ""}`,
  );
  return ok;
}

async function evaluate(sessionId, expression, timeoutMs = 45_000) {
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

async function waitFor(sessionId, expression, timeoutMs, label, intervalMs = 1_000) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < timeoutMs) {
    try {
      if (await evaluate(sessionId, expression)) return;
    } catch { /* retry until the hard deadline */ }
    if (Date.now() >= nextLog) {
      const loader = await evaluate(
        sessionId,
        `JSON.stringify([...document.querySelectorAll('.model-loader')].map(x=>({state:x.dataset.state,status:x.querySelector('.status')?.textContent,buttons:[...x.querySelectorAll('button')].map(b=>b.textContent)})))`,
      ).catch(() => "unavailable");
      console.log(`  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${loader}`);
      nextLog = Date.now() + 10_000;
    }
    await sleep(intervalMs);
  }
  throw new Error(`hard timeout after ${timeoutMs}ms: ${label}`);
}

async function ensureModelsReady(sessionId, label) {
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 12 * 60_000) {
    const state = await evaluate(
      sessionId,
      `(() => {
        const loaders=[...document.querySelectorAll('.model-loader')];
        return {states:loaders.map(x=>x.dataset.state), buttons:loaders.flatMap(x=>[...x.querySelectorAll('button')].filter(b=>!b.disabled).map(b=>b.textContent.trim()))};
      })()`,
    );
    if (state.states.length > 0 && state.states.every((value) => value === "ready")) return;
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(state)}`,
      );
      nextLog = Date.now() + 10_000;
    }
    await evaluate(
      sessionId,
      `(() => {
        const buttons=[...document.querySelectorAll('.model-loader button')].filter(b=>/Download|Retry|Re-download|Continue/i.test(b.textContent)&&!b.disabled);
        buttons.forEach(b=>b.click()); return buttons.length;
      })()`,
    );
    await sleep(1_500);
  }
  throw new Error(`hard timeout after 720000ms: ${label} model preparation`);
}

async function chooseLibri(sessionId) {
  const count = await evaluate(
    sessionId,
    `(() => { const b=[...document.querySelectorAll('#samples button')].find(x=>/LibriSpeech/.test(x.textContent)); if(!b)return 0; b.click(); return 1; })()`,
  );
  if (!count) return false;
  await waitFor(
    sessionId,
    `/LibriSpeech/.test(document.getElementById('clipLabel')?.textContent || '') && !document.getElementById('run')?.disabled`,
    30_000,
    "LibriSpeech decode",
    250,
  );
  return true;
}

async function drive(rung, viewport) {
  const label = `${rung}@${viewport}`;
  const route = ROUTES[rung];
  const page = await openPage(cdp, `http://127.0.0.1:${server.port}/web-ai-showcase/${route}`);
  await setViewport(cdp, page.sessionId, VIEWPORTS[viewport]);
  let ok = true;
  const mark = (name, condition, detail = "") => {
    ok = check(`${label} ${name}`, condition, detail) && ok;
  };
  try {
    await ensureModelsReady(page.sessionId, label);
    const loaderFacts = await evaluate(
      page.sessionId,
      `(() => ({
        checks:[...document.querySelectorAll('.model-loader')].map(x=>Number(x.dataset.localCheckMs)),
        statuses:[...document.querySelectorAll('.model-loader .status')].map(x=>x.textContent.trim()),
        overflow:document.documentElement.scrollWidth-innerWidth,
        visibleStatus:!document.getElementById('status')?.hidden,
        credits:/credits and provenance/i.test(document.body.innerText)
      }))()`,
    );
    mark(
      "local-check-bounded",
      loaderFacts.checks.length > 0 &&
        loaderFacts.checks.every((ms) => Number.isFinite(ms) && ms <= 450),
      JSON.stringify(loaderFacts.checks),
    );
    mark("visible-progress-context", loaderFacts.visibleStatus === true);
    mark("no-horizontal-overflow", loaderFacts.overflow <= 1, `${loaderFacts.overflow}px`);
    mark("visible-audio-provenance", loaderFacts.credits === true);

    if (rung !== "multi") {
      mark("rights-safe-LibriSpeech-control", await chooseLibri(page.sessionId));
    }

    if (rung === "overview") {
      await evaluate(page.sessionId, `document.getElementById('run').click()`);
      await waitFor(
        page.sessionId,
        `document.querySelector('#stages [data-stage="output"]')?.dataset.state === 'done' && /wizard|curtain/i.test(document.getElementById('transcript')?.textContent || '')`,
        120_000,
        `${label} Whisper transcript`,
      );
      const facts = await evaluate(
        page.sessionId,
        `({text:document.getElementById('transcript').textContent,focused:document.activeElement===document.getElementById('transcript'),segments:document.querySelectorAll('#segRows tr').length})`,
      );
      mark("real-Whisper-transcript-visible", /wizard|curtain/i.test(facts.text), facts.text);
      mark("transcript-focused-for-discovery", facts.focused === true);
      mark("see-inside-timestamps", facts.segments > 0, `${facts.segments} segments`);
    } else if (rung === "basics") {
      await evaluate(page.sessionId, `document.getElementById('run').click()`);
      await waitFor(
        page.sessionId,
        `/wizard|curtain/i.test(document.getElementById('transcript')?.textContent || '')`,
        120_000,
        `${label} Whisper transcript`,
      );
      const words = await evaluate(
        page.sessionId,
        `document.querySelectorAll('#wordRows tr').length`,
      );
      mark("real-Whisper-word-timestamps", words > 3, `${words} rows`);
    } else if (rung === "practical") {
      await evaluate(page.sessionId, `document.getElementById('run').click()`);
      await waitFor(
        page.sessionId,
        `/wizard|curtain/i.test(document.getElementById('pad')?.value || '')`,
        120_000,
        `${label} dictation`,
      );
      const captions = await evaluate(
        page.sessionId,
        `document.querySelectorAll('#captions .caption').length`,
      );
      mark("real-Whisper-dictation-and-captions", captions > 0, `${captions} captions`);
    } else if (rung === "wild") {
      await evaluate(page.sessionId, `document.getElementById('run').click()`);
      await waitFor(
        page.sessionId,
        `document.querySelectorAll('#karaoke .w').length > 3`,
        120_000,
        `${label} karaoke`,
      );
      const words = await evaluate(
        page.sessionId,
        `document.querySelectorAll('#karaoke .w').length`,
      );
      mark("real-Whisper-karaoke-timings", words > 3, `${words} words`);
    } else {
      await waitFor(
        page.sessionId,
        `!document.getElementById('transcribe').disabled && !document.getElementById('speak').disabled`,
        30_000,
        `${label} controls`,
      );
      await evaluate(page.sessionId, `document.getElementById('transcribe').click()`);
      await waitFor(
        page.sessionId,
        `/country/i.test(document.getElementById('edit')?.value || '')`,
        120_000,
        `${label} Whisper stage`,
      );
      mark(
        "real-Whisper-stage",
        await evaluate(page.sessionId, `/country/i.test(document.getElementById('edit').value)`),
      );
      await evaluate(
        page.sessionId,
        `(() => { document.getElementById('edit').value='Hello from the private on-device voice loop.'; document.getElementById('speak').click(); return true; })()`,
      );
      await waitFor(
        page.sessionId,
        `document.getElementById('out')?.src.startsWith('blob:') && !document.getElementById('kRead').hidden`,
        180_000,
        `${label} Kokoro stage`,
      );
      const audio = await evaluate(
        page.sessionId,
        `({src:document.getElementById('out').src,ms:document.getElementById('kMs').textContent})`,
      );
      mark(
        "real-Kokoro-speak-back-stage",
        audio.src.startsWith("blob:") && /s/.test(audio.ms),
        audio.ms,
      );
    }

    mark("no-console-errors", page.errors.length === 0, page.errors.join(" | "));
    mark("no-network-failures", page.netFailures.length === 0, page.netFailures.join(" | "));
  } catch (error) {
    ok = false;
    check(`${label} completed`, false, error.message);
  } finally {
    cells.push({ route, viewport, pass: ok });
    await closePage(cdp, page.targetId).catch(() => {});
  }
}

try {
  const started = await startServer();
  server = started.server;
  server.port = started.port;
  chrome = await launchChrome({
    userDataDir: PROFILE_DIR,
    resetProfile: true,
    removeProfileOnKill: false,
  });
  cdp = new CDP(chrome.ws);
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) await drive(rung, viewport);
  }
} catch (error) {
  console.error(`FATAL ${String(error?.stack || error)}`);
} finally {
  if (chrome) await chrome.kill({ removeProfile: false });
  if (server) server.close();
  rmSync(PROFILE_DIR, { recursive: true, force: true });
}

const allPassed = cells.length === Object.keys(ROUTES).length * 2 &&
  cells.every((cell) => cell.pass) && checks === passed;
console.log(`\n${passed}/${checks} checks passed across ${cells.length}/10 route cells.`);
if (WRITE_RUN && allPassed) {
  const commit = execFileSync(
    "git",
    [
      "log",
      "-n1",
      "--format=%H",
      "HEAD",
      "--",
      "models/whisper-speech-to-text",
      "scripts/validate-whisper-speech-to-text.mjs",
    ],
    { cwd: repoRoot, encoding: "utf8" },
  ).trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify(
      {
        commit,
        ranAt: new Date().toISOString(),
        exitCode: 0,
        results: cells.map(({ route, viewport, pass }) => ({ route, viewport, pass })),
        notes:
          "Route-complete desktop/mobile browser run. Real Whisper ASR on the approved LibriSpeech fixture across overview/basics/practical/wild; real Whisper then Kokoro synthesis in multi-model. Visible progress/status, transcript focus, timestamps, provenance, overflow, console/network, and <=450 ms local-check cap asserted.",
      },
      null,
      2,
    ) + "\n",
  );
  console.log(`WROTE ${RUN_RECORD}`);
}
process.exit(allPassed ? 0 : 1);
