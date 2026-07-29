#!/usr/bin/env node
// Route-complete wav2vec2-large-xlsr-53-gender-recognition-librispeech acceptance: real browser
// inference on every published route at desktop and mobile. Advertised stage driven for real:
//   Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech (all routes — binary acoustic
//   voice-label audio-classification, q8 WASM)
// The harness owns one fresh Chrome process tree for the whole run while reusing its own cache
// profile (first cell downloads ~319 MB once; later cells auto-init from cache); every wait has a
// hard deadline and long downloads/inference emit incremental state logs.
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
const RUN_RECORD = join(
  repoRoot,
  "models/wav2vec2-large-xlsr-53-gender-recognition-librispeech/acceptance-run.json",
);
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "xlsr-voicelabel-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const MODEL_ID = "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech";
const SLUG = "wav2vec2-large-xlsr-53-gender-recognition-librispeech";
const ROUTES = {
  overview: `models/${SLUG}/`,
  basics: `models/${SLUG}/basics/`,
  practical: `models/${SLUG}/practical/`,
  wild: `models/${SLUG}/wild/`,
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
      mark(
        ethics === true,
        "honest-framing-panel",
        "binary-acoustic framing + prohibited uses present",
      );
      // chip control swaps the clip and re-runs real inference
      await evaluate(cdp, sessionId, `document.querySelectorAll('#samples .chip')[1].click()`);
      await waitFor(
        cdp,
        sessionId,
        `/Sample ready/.test(document.getElementById('status')?.textContent || '')`,
        60_000,
        `${label} chip-decode`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#scores .score-row').length === 2 && ` +
          `/Acoustic voice label/.test(document.querySelector('#scores .verdict')?.textContent || '')`,
        180_000,
        `${label} rerun`,
      );
      mark(true, "chip-rerun-real-inference", "sample chip drove a second real classification");
      // disposal + reinitialisation through the shared loader
      await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('.model-loader button')].find((b) => /Release from memory/.test(b.textContent))?.click()`,
      );
      await waitFor(
        cdp,
        sessionId,
        `document.querySelector('.model-loader')?.dataset.state === 'released'`,
        120_000,
        `${label} release`,
      );
      mark(true, "dispose-releases-memory", "loader reached released state");
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
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#scores .score-row').length === 2`,
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
        /Below the review threshold/.test(abstain),
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
        `!document.getElementById('batch').disabled`,
        120_000,
        `${label} batch-enabled`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('batch').click()`);
      await waitFor(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#rows .m-label')].filter((c) => /“(female|male)”/.test(c.textContent)).length === 4`,
        300_000,
        `${label} batch`,
      );
      const margins = await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#rows .m-margin')].map((c) => c.textContent)`,
      );
      mark(
        margins.length === 4 && margins.every((m) => /^\d+\.\d%$/.test(m)),
        "batch-4-real-classifications",
        margins.join(" | "),
      );
      const lat = await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#rows .m-ms')].map((c) => c.textContent)`,
      );
      mark(lat.every((m) => /^\d+ ms$/.test(m)), "batch-latencies-real", lat.join(" | "));
      // expectation control drives a real mismatch flag (clip A measures "male"; expect "female")
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('exp-0'); s.value = 'female'; s.dispatchEvent(new Event('change')); return s.value; })()`,
      );
      const flag = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#row-0 .m-flag')?.textContent || ''`,
      );
      mark(/REVIEW —/.test(flag) && /MISMATCH/.test(flag), "expectation-mismatch-flag", flag);
      // low-margin review flag via threshold at 100%
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('threshP'); s.value = '100'; s.dispatchEvent(new Event('input')); return s.value; })()`,
      );
      const lowFlags = await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#rows .m-flag')].filter((c) => /LOW MARGIN/.test(c.textContent)).length`,
      );
      mark(lowFlags === 4, "low-margin-escalation", `${lowFlags}/4 flagged at 100% bar`);
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('threshP'); s.value = '50'; s.dispatchEvent(new Event('input')); return s.value; })()`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('csv').click()`);
      const csv = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('csvPre')?.textContent || ''`,
      );
      mark(
        csv.split("\n").length === 5 && csv.includes("model_acoustic_label"),
        "csv-export-real",
        csv.split("\n")[0] + " (+4 rows)",
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
        `document.querySelectorAll('#sweepRows tr').length === 7`,
        420_000,
        `${label} sweep`,
      );
      const probs = await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#sweepRows .p-female')].map((c) => c.textContent)`,
      );
      mark(
        probs.length === 7 && probs.every((p) => /^\d+\.\d%$/.test(p)),
        "sweep-7-real-inferences",
        probs.join(" | "),
      );
      const distinct = new Set(probs).size;
      mark(
        distinct >= 2,
        "sweep-probabilities-vary",
        `${distinct} distinct P(“female”) values across rates`,
      );
      const logits = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#sweepRows tr td:nth-child(6)')?.textContent || ''`,
      );
      mark(/^\[-?\d+\.\d+, -?\d+\.\d+\]$/.test(logits), "sweep-raw-logits", logits);
      const chartDrawn = await evaluate(
        cdp,
        sessionId,
        `(() => { const c = document.getElementById('chart'); const x = c.getContext('2d');
          const d = x.getImageData(0, 0, c.width, c.height).data; let s = 0;
          for (let i = 3; i < d.length; i += 4) s += d[i]; return s > 0; })()`,
      );
      mark(chartDrawn === true, "sweep-chart-rendered", "canvas has drawn pixels");
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
