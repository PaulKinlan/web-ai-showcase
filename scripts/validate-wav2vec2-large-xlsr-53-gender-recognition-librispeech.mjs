#!/usr/bin/env node
// Route-complete wav2vec2-large-xlsr-53-gender-recognition-librispeech acceptance: real browser
// inference on every published route at desktop and mobile. Advertised stage driven for real:
//   Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech (all routes — binary acoustic
//   voice-label audio-classification, q8 WASM)
// The harness owns one fresh Chrome process tree for the whole run while reusing its own cache
// profile (first cell downloads ~319 MB once; later cells auto-init from cache); every wait has a
// hard deadline and long downloads/inference emit incremental state logs.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
const RUN_RECORD = join(
  repoRoot,
  "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/acceptance-run.json",
);
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "xlsr-voicelabel-acceptance-"));
const SCREEN_DIR = join(
  repoRoot,
  "reports/acceptance/wav2vec2-large-xlsr-53-gender-recognition-librispeech",
);
mkdirSync(SCREEN_DIR, { recursive: true });
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const ROUTES = {
  overview: "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/",
  basics: "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/basics/",
  practical: "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/practical/",
  wild: "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/wild/",
};
const VIEWPORTS = { desktop: DESKTOP, mobile: MOBILE };
const COMMON_ASSERTIONS = [
  "loader-ready",
  "no-horizontal-overflow",
  "console-clean",
  "network-clean",
];
const ROUTE_ASSERTIONS = {
  overview: [
    "real-binary-verdict",
    "raw-logits-shown",
    "backend-readout-wasm",
    "see-inside-logit-table",
    "honest-framing-panel",
    "genuine-worker-response",
    "dispose-releases-memory",
    "reinitialised-real-inference",
  ],
  basics: [
    "backend-readout-wasm",
    "raw-logits-shown",
    "abstention-threshold-control",
    "threshold-reset-real-rerun",
  ],
  practical: [
    "audit-5-real-worker-inferences",
    "audit-raw-logits-five-second",
    "aggregate-variation-summary",
    "truthful-segment-crossings",
    "non-operational-model-risk",
  ],
  wild: [
    "probe-7-real-worker-inferences",
    "observed-probability-deltas",
    "compound-resampling-limits",
    "conditional-crossing-summary",
    "probe-chart-rendered",
  ],
};
const EXPECTED_PER_CELL = Object.fromEntries(
  Object.entries(ROUTE_ASSERTIONS).map((
    [route, ids],
  ) => [route, [...COMMON_ASSERTIONS, ...ids].sort()]),
);
const EXPECTED_TOTAL = 76;
const near = (actual, expected, tolerance = 0.05) => Math.abs(actual - expected) <= tolerance;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const allRequests = [];
let checks = 0;
let passed = 0;
let server;
let chrome;
let cdp;
let clearLifecycleProven = false;

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
  status: (loader.querySelector('.status')?.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 180),
  buttons: [...loader.querySelectorAll('button')].map((button) => button.textContent.trim())
})))`;

async function ensureReady(cdp, sessionId, label) {
  // Fresh profile ⇒ model absent ⇒ the honest Download control appears; click it (the auto-init
  // policy correctly refuses to download silently), then poll the shared loader until ready.
  const started = Date.now();
  let nextLog = 0;
  while (Date.now() - started < 12 * 60_000) {
    const snapshot = JSON.parse(await evaluate(cdp, sessionId, loaderSnapshot));
    if (Date.now() >= nextLog) {
      console.log(
        `  [${label}] ${Math.round((Date.now() - started) / 1000)}s ${JSON.stringify(snapshot)}`,
      );
      nextLog = Date.now() + 10_000;
    }
    if (snapshot.length === 1 && snapshot.every((item) => item.state === "ready")) return snapshot;
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

async function driveRoute(rung, viewport) {
  const label = `${rung}@${viewport}`;
  const { targetId, sessionId, errors, netFailures } = await openPage(
    cdp,
    `http://127.0.0.1:${server.port}/web-ai-showcase/${ROUTES[rung]}`,
  );
  const httpErrors = [];
  cdp.on((message) => {
    if (message.sessionId !== sessionId || message.method !== "Network.responseReceived") return;
    const response = message.params.response;
    const record = {
      url: response.url,
      status: response.status,
      mimeType: response.mimeType,
      fromDiskCache: response.fromDiskCache,
      fromServiceWorker: response.fromServiceWorker,
      contentLength: Number(
        response.headers?.["content-length"] || response.headers?.["Content-Length"] ||
          response.headers?.["x-linked-size"] || response.headers?.["X-Linked-Size"] || 0,
      ),
    };
    allRequests.push(record);
    if (response.status >= 400) httpErrors.push(`${response.status} ${response.url}`);
  });
  await setViewport(cdp, sessionId, VIEWPORTS[viewport]);
  const cell = {
    rung,
    viewport,
    checks: 0,
    passed: 0,
    assertionIds: [],
    assertionEvidence: {},
    errors: [],
    netFailures: [],
    httpErrors,
  };
  const mark = (ok, what, detail) => {
    cell.checks++;
    cell.assertionIds.push(what);
    cell.assertionEvidence[what] = String(detail ?? "");
    if (ok) cell.passed++;
    else cell.errors.push(`${what}: ${String(detail).slice(0, 200)}`);
    check(`${label} ${what}`, ok, detail);
  };
  try {
    await ensureReady(cdp, sessionId, label);
    mark(
      true,
      "loader-ready",
      "inputs enabled (cold download for first cell; cached auto-init thereafter)",
    );
    await evaluate(
      cdp,
      sessionId,
      `(() => {
      window.__xlsrWorkerResults = [];
      window.addEventListener('xlsr-worker-result', event => window.__xlsrWorkerResults.push(event.detail));
      return true;
    })()`,
    );

    if (rung === "overview") {
      // sample auto-selected; wait for decode then run real inference
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('run').disabled`,
        120_000,
        `${label} run-enabled`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#scores .score-row').length === 2`,
        180_000,
        `${label} scores`,
      );
      const verdict = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#scores .verdict')?.textContent || ''`,
      );
      mark(
        /Acoustic voice label: “(female|male)” — margin \d+\.\d%/.test(verdict),
        "real-binary-verdict",
        verdict.slice(0, 120),
      );
      const logits = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('logitLine')?.textContent || ''`,
      );
      mark(
        /^logits \[-?\d+\.\d+, -?\d+\.\d+\] → softmax →/.test(logits),
        "raw-logits-shown",
        logits.slice(0, 120),
      );
      const backend = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('rBackend')?.textContent || ''`,
      );
      mark(backend === "WASM", "backend-readout-wasm", backend);
      const inside = await evaluate(
        cdp,
        sessionId,
        `document.querySelectorAll('#insideRows tr').length === 2 && ` +
          `/^-?\\d+\\.\\d{3}$/.test(document.querySelector('#insideRows tr td:nth-child(3)')?.textContent || '')`,
      );
      mark(inside === true, "see-inside-logit-table", "two labels with raw logits rendered");
      const ethics = await evaluate(
        cdp,
        sessionId,
        `/binary acoustic voice-label classifier|binary acoustic voice labels/i.test(document.querySelector('.ethics')?.textContent || '') && /never/i.test(document.querySelector('.ethics')?.textContent || '')`,
      );
      // Exercise the real upload control and deterministic microphone fallback; both select the
      // approved WAV and must flow through capture-ux's explicit "Use this" review state.
      const documentNode = await cdp.send(
        "DOM.getDocument",
        { depth: -1, pierce: true },
        sessionId,
      );
      const fileNode = await cdp.send("DOM.querySelector", {
        nodeId: documentNode.root.nodeId,
        selector: "#capFile input[type=file]",
      }, sessionId);
      await cdp.send("DOM.setFileInputFiles", {
        nodeId: fileNode.nodeId,
        files: [
          join(
            repoRoot,
            "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/libri-61-70968-0000-16k-mono.wav",
          ),
        ],
      }, sessionId);
      await waitFor(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#capFile button')].some(button=>button.textContent.includes('Use this'))`,
        30_000,
        `${label} upload-review`,
      );
      await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#capFile button')].find(button=>button.textContent.includes('Use this')).click()`,
      );
      await waitFor(
        cdp,
        sessionId,
        `/Audio ready/.test(document.getElementById('status').textContent)`,
        30_000,
        `${label} upload-use`,
      );
      await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#capMic .cap__fallback button').click()`,
      );
      await waitFor(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#capMic button')].some(button=>button.textContent.includes('Use this'))`,
        30_000,
        `${label} mic-fallback-review`,
      );
      await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#capMic button')].find(button=>button.textContent.includes('Use this')).click()`,
      );
      await waitFor(
        cdp,
        sessionId,
        `/Audio ready/.test(document.getElementById('status').textContent)`,
        30_000,
        `${label} mic-fallback-use`,
      );
      mark(
        ethics === true,
        "honest-framing-panel",
        "prohibited-use framing present; upload and deterministic mic fallback exercised",
      );
      const protocolResult = await evaluate(cdp, sessionId, `window.__xlsrWorkerResults.at(-1)`);
      mark(
        protocolResult?.logits?.length === 2 &&
          near(protocolResult.logits[0], -3.2610774) && near(protocolResult.logits[1], 3.3608596) &&
          protocolResult?.preprocessing?.modelSamples === 80000 &&
          protocolResult?.runtimeEvidence?.revision === "6bea1eddcfca9842add425123f4955d5b4f153f7",
        "genuine-worker-response",
        JSON.stringify(protocolResult).slice(0, 220),
      );
      // Release DURING work: deterministic dispose must reject the pending protocol promise, disable
      // page controls, and reach released without a stale completion re-enabling the page.
      await evaluate(
        cdp,
        sessionId,
        `(() => {
          document.getElementById('run').click();
          setTimeout(() => [...document.querySelectorAll('.model-loader button')].find((b) => /Release from memory/.test(b.textContent))?.click(), 10);
          return true;
        })()`,
      );
      await waitFor(
        cdp,
        sessionId,
        `document.querySelector('.model-loader')?.dataset.state === 'released'`,
        120_000,
        `${label} release`,
      );
      const releasedState = await evaluate(
        cdp,
        sessionId,
        `({disabled:document.getElementById('run').disabled,state:document.querySelector('.model-loader').dataset.state})`,
      );
      mark(
        releasedState.disabled && releasedState.state === "released",
        "dispose-releases-memory",
        "release during inference rejected pending work and left controls disabled",
      );
      await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('.model-loader button')].find((b) => /Load model into memory/.test(b.textContent))?.click()`,
      );
      await waitFor(
        cdp,
        sessionId,
        `document.querySelector('.model-loader')?.dataset.state === 'ready'`,
        300_000,
        `${label} reinit`,
      );
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('run').disabled`,
        120_000,
        `${label} reinit-run-enabled`,
      );
      const beforeReinitRun = await evaluate(cdp, sessionId, `window.__xlsrWorkerResults.length`);
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `window.__xlsrWorkerResults.length > ${beforeReinitRun}`,
        180_000,
        `${label} post-reinit`,
      );
      mark(true, "reinitialised-real-inference", "worker re-created; inference real after release");
    } else if (rung === "basics") {
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('run').disabled`,
        120_000,
        `${label} run-enabled`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#scores .score-row').length === 2`,
        180_000,
        `${label} scores`,
      );
      const backend = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('rBackend')?.textContent || ''`,
      );
      mark(backend === "WASM", "backend-readout-wasm", backend);
      const logits = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('logitLine')?.textContent || ''`,
      );
      mark(
        /^logits \[-?\d+\.\d+, -?\d+\.\d+\]/.test(logits),
        "raw-logits-shown",
        logits.slice(0, 90),
      );
      // abstention threshold control: 100% forces the honest review state
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('thresh'); s.value = '100'; s.dispatchEvent(new Event('input')); return s.value; })()`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `!!document.querySelector('#scores .verdict.abstain')`,
        180_000,
        `${label} abstain`,
      );
      const abstain = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#scores .verdict.abstain')?.textContent || ''`,
      );
      mark(
        /Below the display threshold/.test(abstain),
        "abstention-threshold-control",
        abstain.slice(0, 110),
      );
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('thresh'); s.value = '0'; s.dispatchEvent(new Event('input')); return s.value; })()`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#scores .score-row').length === 2 && ` +
          `!document.querySelector('#scores .verdict.abstain')`,
        180_000,
        `${label} unabstain`,
      );
      mark(
        true,
        "threshold-reset-real-rerun",
        "threshold back to 0 re-labelled with real inference",
      );
    } else if (rung === "practical") {
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('audit').disabled`,
        120_000,
        `${label} audit-enabled`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('audit').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#rows tr').length === 5`,
        360_000,
        `${label} audit`,
      );
      const audit = await evaluate(
        cdp,
        sessionId,
        `({
        rows:[...document.querySelectorAll('#rows tr')].map(row=>row.textContent),
        summary:document.getElementById('summary').textContent,
        worker:window.__xlsrWorkerResults
      })`,
      );
      mark(
        audit.rows.length === 5 && audit.worker.length === 5,
        "audit-5-real-worker-inferences",
        `${audit.worker.length} protocol results`,
      );
      mark(
        audit.worker.every((result) =>
          result.logits.length === 2 && result.preprocessing.modelSamples === 80000
        ),
        "audit-raw-logits-five-second",
        audit.worker.map((result) => result.logits.map((v) => v.toFixed(3)).join("/")).join(" | "),
      );
      mark(
        /maximum absolute delta/.test(audit.summary) &&
          /top-label crossings \d\/4/.test(audit.summary),
        "aggregate-variation-summary",
        audit.summary,
      );
      mark(
        audit.rows.every((row) => /zero-pad to 5\.000 s/.test(row)) &&
          audit.rows.every((row) => /baseline|yes|no/.test(row)),
        "truthful-segment-crossings",
        audit.rows.join(" | "),
      );
      const nonOperational = await evaluate(
        cdp,
        sessionId,
        `!document.querySelector('select') && !document.getElementById('csv') && /accepts no expected/.test(document.querySelector('.ethics').textContent) && /exports no person-level/.test(document.querySelector('.ethics').textContent)`,
      );
      mark(
        nonOperational === true,
        "non-operational-model-risk",
        "no expected labels, identity comparison, mismatch control, or export",
      );
    } else if (rung === "wild") {
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('sweep').disabled`,
        120_000,
        `${label} sweep-enabled`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('sweep').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#rows tr').length === 7`,
        480_000,
        `${label} sweep`,
      );
      const probe = await evaluate(
        cdp,
        sessionId,
        `({
        probs:[...document.querySelectorAll('#rows .p-female')].map(cell=>cell.textContent),
        rows:[...document.querySelectorAll('#rows tr')].map(row=>row.textContent),
        summary:document.getElementById('summary').textContent,
        worker:window.__xlsrWorkerResults,
        limits:document.querySelector('.ethics').textContent
      })`,
      );
      mark(
        probe.probs.length === 7 && probe.worker.length === 7,
        "probe-7-real-worker-inferences",
        probe.probs.join(" | "),
      );
      mark(
        new Set(probe.probs).size >= 2 &&
          probe.worker.every((result) => result.logits.length === 2) &&
          near(probe.worker[3].logits[0], -3.2610774) &&
          near(probe.worker[3].logits[1], 3.3608596) &&
          near(probe.worker[4].logits[0], 3.146918) &&
          near(probe.worker[4].logits[1], -3.257751),
        "observed-probability-deltas",
        `${new Set(probe.probs).size} distinct values; raw logits ${
          probe.worker.map((result) => result.logits.map((value) => value.toFixed(6)).join("/"))
            .join(" | ")
        }`,
      );
      mark(
        /pitch, spectrum\/formants, tempo, duration, and temporal structure together/.test(
          probe.limits,
        ) && /isolates neither pitch nor timbre/.test(probe.limits),
        "compound-resampling-limits",
        probe.limits.slice(0, 180),
      );
      mark(
        /top-label crossings \d\/6/.test(probe.summary) &&
          (/No top-label crossing was measured/.test(probe.summary) ||
            /At least one crossing was measured/.test(probe.summary)),
        "conditional-crossing-summary",
        probe.summary,
      );
      const chartDrawn = await evaluate(
        cdp,
        sessionId,
        `(() => { const c=document.getElementById('chart'),x=c.getContext('2d'),d=x.getImageData(0,0,c.width,c.height).data; for(let i=3;i<d.length;i+=4) if(d[i]) return true; return false; })()`,
      );
      mark(chartDrawn === true, "probe-chart-rendered", "canvas has drawn pixels");
    }

    const noOverflow = await evaluate(
      cdp,
      sessionId,
      `document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1`,
    );
    mark(noOverflow, "no-horizontal-overflow", `viewport ${viewport}`);
    mark(errors.length === 0, "console-clean", errors.join(" | ") || "0 errors");
    mark(
      netFailures.length === 0 && httpErrors.length === 0,
      "network-clean",
      [...netFailures, ...httpErrors].join(" | ") || "0 failed requests / HTTP 4xx/5xx",
    );
  } catch (error) {
    mark(false, "route-drive", error.message);
  }
  cell.browserConsole = [...errors];
  cell.networkFailures = [...netFailures];
  const actualIds = [...cell.assertionIds].sort();
  if (JSON.stringify(actualIds) !== JSON.stringify(EXPECTED_PER_CELL[rung])) {
    cell.errors.push(
      `frozen assertion-id drift: expected ${EXPECTED_PER_CELL[rung].join(",")} got ${
        actualIds.join(",")
      }`,
    );
  }
  const screenshotPath =
    `reports/acceptance/wav2vec2-large-xlsr-53-gender-recognition-librispeech/${rung}-${viewport}.png`;
  try {
    const metrics = await cdp.send("Page.getLayoutMetrics", {}, sessionId);
    const width = Math.ceil(metrics.cssContentSize?.width || VIEWPORTS[viewport].width);
    const height = Math.ceil(metrics.cssContentSize?.height || VIEWPORTS[viewport].height);
    const shot = await cdp.send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
      clip: { x: 0, y: 0, width, height, scale: 1 },
    }, sessionId);
    writeFileSync(join(repoRoot, screenshotPath), Buffer.from(shot.data, "base64"));
    cell.screenshot = screenshotPath;
  } catch (error) {
    cell.errors.push(`screenshot: ${error.message}`);
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
  // Dedicated-worker requests live on child CDP targets. Auto-attach and enable Network there so
  // the run can prove the actual Transformers.js request used the full immutable revision URL.
  await cdp.send("Target.setAutoAttach", {
    autoAttach: true,
    waitForDebuggerOnStart: false,
    flatten: true,
  });
  cdp.on((message) => {
    if (message.method === "Target.attachedToTarget" && message.params.sessionId) {
      cdp.send("Network.enable", {}, message.params.sessionId).catch(() => {});
      return;
    }
    if (message.method !== "Network.responseReceived") return;
    const response = message.params.response;
    if (!response.url.includes("huggingface.co") && !response.url.includes("hf.co")) return;
    allRequests.push({
      url: response.url,
      status: response.status,
      mimeType: response.mimeType,
      fromDiskCache: response.fromDiskCache,
      fromServiceWorker: response.fromServiceWorker,
      contentLength: Number(
        response.headers?.["content-length"] || response.headers?.["Content-Length"] ||
          response.headers?.["x-linked-size"] || response.headers?.["X-Linked-Size"] || 0,
      ),
    });
  });
  for (const rung of Object.keys(ROUTES)) {
    for (const viewport of Object.keys(VIEWPORTS)) {
      await driveRoute(rung, viewport);
    }
  }
  // Focused shared-loader regression: after all route cells, warm-init once, clear while a live
  // instance exists, and require synchronized download-required + disabled route controls.
  const clearPage = await openPage(
    cdp,
    `http://127.0.0.1:${server.port}/web-ai-showcase/models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/`,
  );
  try {
    await ensureReady(cdp, clearPage.sessionId, "clear-cache-regression");
    await evaluate(
      cdp,
      clearPage.sessionId,
      `[...document.querySelectorAll('.model-loader button')].find(button=>/Clear cached model/.test(button.textContent))?.click()`,
    );
    await waitFor(
      cdp,
      clearPage.sessionId,
      `document.querySelector('.model-loader')?.dataset.state === 'download-required'`,
      120_000,
      "clear-cache-dispose",
    );
    clearLifecycleProven = await evaluate(
      cdp,
      clearPage.sessionId,
      `document.getElementById('run').disabled && [...document.querySelectorAll('.model-loader button')].some(button=>/Download model/.test(button.textContent))`,
    );
    console.log(
      `${clearLifecycleProven ? "PASS" : "FAIL"}  clear-cache disposes live instance before assets`,
    );
  } finally {
    await closePage(cdp, clearPage.targetId);
  }
} finally {
  if (chrome) await chrome.kill({ removeProfile: false });
  rmSync(PROFILE_DIR, { recursive: true, force: true });
  if (server) await new Promise((resolve) => server.server.close(resolve));
}

console.log(
  `\n${passed}/${checks} checks passed across ${results.length} route cells (frozen denominator ${EXPECTED_TOTAL})`,
);
const q8Requests = allRequests.filter((request) =>
  request.url.includes(
    "/resolve/6bea1eddcfca9842add425123f4955d5b4f153f7/onnx/model_quantized.onnx",
  )
);
const pinnedNetworkProven = q8Requests.some((request) =>
  request.status === 200 && request.contentLength === 318834205
);
console.log(
  `${pinnedNetworkProven ? "PASS" : "FAIL"}  pinned q8 network request — ${
    JSON.stringify(q8Requests)
  }`,
);
const succeeded = checks === EXPECTED_TOTAL && passed === EXPECTED_TOTAL && pinnedNetworkProven &&
  clearLifecycleProven && results.length === 8 &&
  results.every((c) =>
    c.checks === EXPECTED_PER_CELL[c.rung].length && c.checks === c.passed && c.errors.length === 0
  );
if (WRITE_RUN && succeeded) {
  const commit = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: repoRoot,
    encoding: "utf8",
  }).trim();
  writeFileSync(
    RUN_RECORD,
    JSON.stringify(
      {
        schemaVersion: 2,
        validator: "scripts/validate-wav2vec2-large-xlsr-53-gender-recognition-librispeech.mjs",
        commit,
        ranAt: new Date().toISOString(),
        exitCode: 0,
        checks: `${EXPECTED_TOTAL}/${EXPECTED_TOTAL}`,
        frozenDenominator: {
          total: EXPECTED_TOTAL,
          perCell: Object.fromEntries(
            Object.entries(EXPECTED_PER_CELL).map((
              [route, ids],
            ) => [route, { count: ids.length, assertionIds: ids }]),
          ),
        },
        pinnedNetworkEvidence: q8Requests,
        clearCacheLifecycle: {
          pass: clearLifecycleProven,
          evidence:
            "warm live instance disposed; route run control disabled; loader transitioned to download-required after asset removal",
        },
        blockedManual: [
          "Physical low-memory phone viability cannot be established by desktop Chrome viewport emulation; the product discloses RAM/storage limits and support remains needs-review.",
          "A real microphone grant/recording is environment-specific; shared capture-ux deterministic denied/fallback behavior is covered by its component contract and remains a manual device check, not counted as pass here.",
        ],
        results: results.map((cell) => ({
          route: ROUTES[cell.rung],
          viewport: cell.viewport,
          checks: `${cell.passed}/${cell.checks}`,
          assertionIds: cell.assertionIds,
          assertionEvidence: cell.assertionEvidence,
          assertionFailures: cell.errors,
          console: cell.browserConsole,
          networkFailures: cell.networkFailures,
          httpErrors: cell.httpErrors,
          screenshot: cell.screenshot,
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
