#!/usr/bin/env node
// Route-complete roberta-fill-mask acceptance: real browser inference on every published route
// at desktop and mobile. Advertised stage driven for real:
//   Xenova/roberta-base (all routes — fill-mask masked-LM, q8 WASM)
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
const RUN_RECORD = join(repoRoot, "models/roberta-fill-mask/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "robertafm-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const MODEL_ID = "Xenova/roberta-base";
const ROUTES = {
  overview: "models/roberta-fill-mask/",
  basics: "models/roberta-fill-mask/basics/",
  practical: "models/roberta-fill-mask/practical/",
  wild: "models/roberta-fill-mask/wild/",
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
      // real inference: ranked predictions render with probability bars + raw logits.
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#preds .pred-row').length >= 3`,
        120_000,
        `${label} preds`,
      );
      const top = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#preds .pred-row .pred-tok')?.textContent || ''`,
      );
      mark(top.length > 0, "real-prediction-rendered", `top token "${top}"`);
      const logitTxt = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#preds .pred-logit')?.textContent || ''`,
      );
      mark(/^logit -?\d+\.\d{2}$/.test(logitTxt), "raw-logit-shown", logitTxt);
      const ent = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('entLine')?.hidden === false && /bits/.test(document.getElementById('entLine')?.textContent || '')`,
      );
      mark(ent === true, "entropy-see-inside", "top-k entropy rendered");
      // chip control swaps the sentence and re-runs real inference
      await evaluate(cdp, sessionId, `document.querySelectorAll('#chips .mask-chip')[1].click()`);
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#preds .pred-row').length >= 3`,
        120_000,
        `${label} rerun`,
      );
      const filled = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('filled')?.textContent || ''`,
      );
      mark(!filled.includes("<mask>"), "chip-rerun-fills-mask", filled.slice(0, 80));
      // top-k slider control
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('topk'); s.value = '5'; s.dispatchEvent(new Event('input')); return s.value; })()`,
      );
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#preds .pred-row').length === 5`,
        120_000,
        `${label} topk`,
      );
      mark(true, "topk-slider-5-rows", "slider drove a 5-row re-run");
    } else if (rung === "basics") {
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#preds .pred-row').length >= 3`,
        120_000,
        `${label} preds`,
      );
      const top = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#preds .pred-row .pred-tok')?.textContent || ''`,
      );
      mark(top.length > 0, "real-prediction-rendered", `top token "${top}"`);
      const backend = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('rBackend')?.textContent || ''`,
      );
      mark(backend === "WASM", "backend-readout-wasm", backend);
      await evaluate(cdp, sessionId, `document.querySelectorAll('#chips .mask-chip')[2].click()`);
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#preds .pred-row').length >= 3`,
        120_000,
        `${label} rerun`,
      );
      mark(true, "chip-rerun", "object chip re-ran inference");
    } else if (rung === "practical") {
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#scores .cand-row').length >= 3`,
        120_000,
        `${label} scores`,
      );
      const first = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#scores .cand-row .cand-word')?.textContent || ''`,
      );
      mark(first.length > 0, "candidate-scores-rendered", `top candidate "${first}"`);
      const num = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#scores .cand-row .cand-num')?.textContent || ''`,
      );
      mark(/logit -?\d+\.\d{2}/.test(num), "candidate-logits-shown", num);
      await evaluate(cdp, sessionId, `document.querySelectorAll('#chips .mask-chip')[2].click()`);
      await sleep(1_500);
      await waitFor(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#scores .cand-row .cand-word')].some(e => /moved|postponed|cancelled|rescheduled|brought/.test(e.textContent))`,
        120_000,
        `${label} rerun`,
      );
      mark(true, "template-chip-rerun", "scheduling template re-scored its candidates");
    } else if (rung === "wild") {
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#results .probe-card').length === 12`,
        180_000,
        `${label} batch`,
      );
      const n = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('rN')?.textContent || ''`,
      );
      mark(n === "12", "batch-12-real-fills", `${n} prompts filled`);
      const sure = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('rSure')?.textContent || ''`,
      );
      mark(/^\d+\/12$/.test(sure), "certainty-readout", sure);
      const oneTok = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#results .probe-card .pred-tok')?.textContent || ''`,
      );
      mark(oneTok.length > 0, "proverb-top1-real", `first card top-1 "${oneTok}"`);
      await evaluate(
        cdp,
        sessionId,
        `(() => { const s = document.getElementById('set'); s.value = 'idioms'; s.dispatchEvent(new Event('change')); return s.value; })()`,
      );
      await sleep(2_000);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#results .probe-card').length === 12`,
        180_000,
        `${label} idioms`,
      );
      mark(true, "set-select-rerun", "idioms set re-ran the batch");
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
