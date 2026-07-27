#!/usr/bin/env node
// Route-complete trocr-handwriting acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real:
//   Xenova/trocr-small-handwritten (all routes — TrOCR handwriting image-to-text, q8 WASM)
//   Xenova/m2m100_418M             (multi-model — stage-2 translation of the transcribed line)
// Deterministic OCR evidence (fresh-profile probes, WASM q8): sample-note.jpg via the exact
// crop.crop() PNG path reads "a few months were"; sample-manuscript.jpg reads "HelpLearn to edit".
// The Wild route's camera is exercised through a stubbed getUserMedia returning a canvas
// captureStream that renders a handwritten-style line into the reticle band — the crop + OCR the
// page then runs is the real pipeline on real camera-provided frames.
// The harness owns one fresh Chrome process tree per route cell while reusing its own cache
// profile; every wait has a hard deadline and long downloads emit incremental state logs.
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
const RUN_RECORD = join(repoRoot, "models/trocr-handwriting/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "trocr-handwriting-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const OCR_ID = "Xenova/trocr-small-handwritten";
const TR_ID = "Xenova/m2m100_418M";
const ROUTES = {
  overview: "models/trocr-handwriting/",
  basics: "models/trocr-handwriting/basics/",
  practical: "models/trocr-handwriting/practical/",
  wild: "models/trocr-handwriting/wild/",
  multimodel: "models/trocr-handwriting/multi-model/",
};
const EXPECTED = { overview: 3, basics: 4, practical: 3, wild: 3, multimodel: 3 };
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

// The crop canvas starts at the default 300x150 bitmap; CropCanvas resizes it to the scaled
// image dimensions once the sample's pixels actually load — that (not width>0) proves loadable.
const CROP_LOADED =
  `(() => { const c = document.querySelector('#canvas'); return c.width !== 300 || c.height !== 150; })()`;

// Draw two pen-like strokes on a canvas element via real mouse input (the DrawPad listens to
// pointer events, which Chrome synthesises from mouse input).
async function drawOnPad(cdp, sessionId, selector) {
  // The pad can sit below the fold once the loader panel expands; Input.dispatchMouseEvent only
  // reaches elements inside the viewport, so scroll it into view before measuring coordinates.
  await evaluate(
    cdp,
    sessionId,
    `(() => { document.querySelector('${selector}').scrollIntoView({ block: 'center' }); return true; })()`,
  );
  await sleep(400);
  const rect = await evaluate(
    cdp,
    sessionId,
    `(() => { const r = document.querySelector('${selector}').getBoundingClientRect(); return { x: r.left, y: r.top, w: r.width, h: r.height }; })()`,
  );
  const mouse = (type, x, y) =>
    cdp.send("Input.dispatchMouseEvent", {
      type,
      x: Math.round(x),
      y: Math.round(y),
      button: "left",
      buttons: type === "mouseReleased" ? 0 : 1,
      clickCount: type === "mousePressed" ? 1 : 0,
      pointerType: "mouse",
    }, sessionId);
  const midY = rect.y + rect.h * 0.55;
  // One long wavy "cursive" stroke across the pad.
  await mouse("mousePressed", rect.x + rect.w * 0.1, midY);
  for (let i = 1; i <= 16; i++) {
    await mouse(
      "mouseMoved",
      rect.x + rect.w * (0.1 + 0.7 * (i / 16)),
      midY + Math.sin(i * 1.4) * rect.h * 0.18,
    );
    await sleep(15);
  }
  await mouse("mouseReleased", rect.x + rect.w * 0.8, midY);
  // A short second stroke (like a dot/tittle).
  await mouse("mousePressed", rect.x + rect.w * 0.86, midY - rect.h * 0.2);
  await mouse("mouseMoved", rect.x + rect.w * 0.88, midY - rect.h * 0.14);
  await mouse("mouseReleased", rect.x + rect.w * 0.88, midY - rect.h * 0.14);
}

async function exercise(cdp, page, rung, viewport) {
  const sid = page.sessionId;
  const mobile = viewport === "mobile";
  await setViewport(cdp, sid, mobile ? MOBILE : DESKTOP);
  await ensureReady(cdp, sid, rung === "multimodel" ? 2 : 1, `${viewport} ${rung}`);

  if (rung === "overview") {
    // Pick the exercise-book sample (real control), wait for the crop image, then read it.
    await evaluate(
      cdp,
      sid,
      `(() => { [...document.querySelectorAll('#samples .sample-thumb')][0].click(); return true; })()`,
    );
    await waitFor(cdp, sid, CROP_LOADED, 30_000, `${viewport} overview sample load`);
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#readPhoto').click(); return true; })()`,
    );
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
      text: document.querySelector('#textOut')?.textContent || '',
      toks: document.querySelectorAll('#trace .tok').length,
      chars: document.querySelector('#iChars')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || '',
      rTok: document.querySelector('#rTok')?.textContent || ''
    })`,
    );
    check(
      `${viewport} overview: real ${OCR_ID} transcription + token trace inside`,
      evidence.text.includes("months") && evidence.toks >= 3 && Number(evidence.chars) >= 8 &&
        /WASM/.test(evidence.backend) && Number(evidence.rTok) >= 3,
      JSON.stringify(evidence),
    );
  } else if (rung === "basics") {
    // Example 1: draw real strokes on the pad and read them.
    await drawOnPad(cdp, sid, "#pad");
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#readPad').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `/Done|Failed/.test(document.querySelector('#s1')?.textContent || '')`,
      120_000,
      `${viewport} basics pad inference`,
    );
    const padEvidence = await evaluate(
      cdp,
      sid,
      `({ s: document.querySelector('#s1')?.textContent || '', out: document.querySelector('#out1')?.textContent || '' })`,
    );
    check(
      `${viewport} basics: real pad strokes read by ${OCR_ID}`,
      /Done — [1-9]\d* tokens/.test(padEvidence.s) && padEvidence.out.length >= 1 &&
        padEvidence.out !== "(nothing recognised)",
      JSON.stringify(padEvidence),
    );
    // Example 2: the preloaded public-domain manuscript line.
    await waitFor(cdp, sid, CROP_LOADED, 30_000, `${viewport} basics manuscript load`);
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#readLine').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `/Done|Failed/.test(document.querySelector('#s2')?.textContent || '')`,
      120_000,
      `${viewport} basics manuscript inference`,
    );
    const lineEvidence = await evaluate(
      cdp,
      sid,
      `({ s: document.querySelector('#s2')?.textContent || '', out: document.querySelector('#out2')?.textContent || '' })`,
    );
    check(
      `${viewport} basics: real manuscript line read by ${OCR_ID}`,
      /Done — [1-9]\d* tokens/.test(lineEvidence.s) && lineEvidence.out.includes("edit"),
      JSON.stringify(lineEvidence),
    );
  } else if (rung === "practical") {
    // The note sample is preloaded; read one boxed line into the editable list.
    await waitFor(cdp, sid, CROP_LOADED, 30_000, `${viewport} practical sample load`);
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#readLine').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelectorAll('#rows .row-line').length >= 1`,
      120_000,
      `${viewport} practical line read`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      rows: document.querySelectorAll('#rows .row-line').length,
      first: document.querySelector('#rows .row-line input')?.value || '',
      lines: document.querySelector('#rLines')?.textContent || '',
      copyHidden: document.querySelector('#copyAll')?.hidden,
      status: document.querySelector('#runStatus')?.textContent || ''
    })`,
    );
    check(
      `${viewport} practical: line digitised into editable list by ${OCR_ID}`,
      evidence.rows >= 1 && evidence.first.includes("months") && Number(evidence.lines) >= 1 &&
        evidence.copyHidden === false && /Line added/.test(evidence.status),
      JSON.stringify(evidence),
    );
  } else if (rung === "wild") {
    // Stub the camera with a canvas captureStream rendering a handwritten-style line exactly into
    // the reticle band; the page's own crop + ${OCR_ID} run is the real pipeline on the frames.
    await evaluate(
      cdp,
      sid,
      `(() => {
      const c = document.createElement('canvas'); c.width = 1280; c.height = 720;
      const x = c.getContext('2d');
      const paint = () => {
        x.fillStyle = '#fdfdf8'; x.fillRect(0, 0, 1280, 720);
        x.fillStyle = '#1a1a2e'; x.font = 'italic 60px "Comic Sans MS", cursive, serif';
        x.fillText('meet me at noon', 130, 388);
        x.fillStyle = '#bbb'; x.fillRect(1200, 20 + (Date.now() / 100) % 40, 10, 10);
      };
      paint(); setInterval(paint, 100);
      navigator.mediaDevices.getUserMedia = async () => c.captureStream(10);
      return true;
    })()`,
    );
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#camBtn').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#shotBtn')?.disabled === false`,
      30_000,
      `${viewport} wild camera start`,
    );
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#shotBtn').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      120_000,
      `${viewport} wild camera inference`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      text: document.querySelector('#textOut')?.textContent || '',
      status: document.querySelector('#runStatus')?.textContent || '',
      backend: document.querySelector('#rBackend')?.textContent || '',
      toks: document.querySelector('#rTok')?.textContent || ''
    })`,
    );
    check(
      `${viewport} wild: live camera frame read by ${OCR_ID}`,
      evidence.status === "Done." && evidence.text.length >= 2 &&
        !evidence.text.startsWith("(nothing read") && /WASM/.test(evidence.backend) &&
        Number(evidence.toks) >= 1,
      JSON.stringify(evidence),
    );
  } else {
    // Multi-model: TrOCR reads the sample note, then M2M100 translates the transcription.
    await evaluate(
      cdp,
      sid,
      `(() => { [...document.querySelectorAll('#samples .sample-thumb')][0].click(); return true; })()`,
    );
    await waitFor(cdp, sid, CROP_LOADED, 30_000, `${viewport} multimodel sample load`);
    await evaluate(
      cdp,
      sid,
      `(() => { document.querySelector('#runPhoto').click(); return true; })()`,
    );
    await waitFor(
      cdp,
      sid,
      `document.querySelector('#readout')?.hidden === false`,
      360_000,
      `${viewport} ${OCR_ID} → ${TR_ID}`,
    );
    const evidence = await evaluate(
      cdp,
      sid,
      `({
      ocr: document.querySelector('#textOut')?.textContent || '',
      tr: document.querySelector('#transOut')?.textContent || '',
      status: document.querySelector('#runStatus')?.textContent || '',
      ocrMs: document.querySelector('#rOcr')?.textContent || '',
      trMs: document.querySelector('#rTr')?.textContent || ''
    })`,
    );
    check(
      `${viewport} multi-model: ${OCR_ID} then ${TR_ID} both ran`,
      evidence.ocr.includes("months") && evidence.tr.length >= 3 &&
        evidence.status === "Done." && /ms/.test(evidence.ocrMs) && /ms/.test(evidence.trMs),
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

const totalChecks = Object.values(EXPECTED).reduce((a, b) => a + b, 0) * 2;
const succeeded = checks === totalChecks && passed === checks && results.length === 10 &&
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
