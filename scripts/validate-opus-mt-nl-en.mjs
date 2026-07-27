#!/usr/bin/env node
// Route-complete opus-mt-nl-en acceptance: real browser inference on every published route at
// desktop and mobile. Advertised stages driven for real:
//   Xenova/opus-mt-nl-en  (all routes — Dutch→English translation, q8 WASM)
//   Xenova/opus-mt-en-nl  (wild route — reverse pair for the round-trip, downloaded on demand)
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
const RUN_RECORD = join(repoRoot, "models/opus-mt-nl-en/acceptance-run.json");
const PROFILE_DIR = mkdtempSync(join(tmpdir(), "opusmtnlen-acceptance-"));
if (WRITE_RUN) rmSync(RUN_RECORD, { force: true });

const MODEL_ID = "Xenova/opus-mt-nl-en";
const ROUTES = {
  overview: "models/opus-mt-nl-en/",
  basics: "models/opus-mt-nl-en/basics/",
  practical: "models/opus-mt-nl-en/practical/",
  wild: "models/opus-mt-nl-en/wild/",
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
      // real inference: chip fills the textarea, Translate runs the model, English streams out.
      await evaluate(cdp, sessionId, `document.querySelectorAll('.chip')[1].click()`);
      const src = await evaluate(cdp, sessionId, `document.getElementById('text').value`);
      mark(/treinstation/i.test(src), "chip-fills-textarea", src.slice(0, 60));
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('readout').hidden && document.getElementById('out').textContent.trim().length > 3`,
        180_000,
        `${label} translation`,
      );
      const out1 = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('out').textContent.trim()`,
      );
      mark(
        /station/i.test(out1) && !/treinstation/i.test(out1),
        "real-translation-rendered",
        out1.slice(0, 90),
      );
      const tok = await evaluate(cdp, sessionId, `document.getElementById('rTok').textContent`);
      mark(/^\d+ → \d+$/.test(tok), "token-readout", tok);
      const inTok = await evaluate(cdp, sessionId, `document.getElementById('inTok').textContent`);
      mark(Number(inTok) > 3, "see-inside-tokens", `in=${inTok}`);
      // edit → rerun changes the output
      await evaluate(
        cdp,
        sessionId,
        `(()=>{const t=document.getElementById('text'); t.value='De kat zit op de mat.'; t.dispatchEvent(new Event('input'));})()`,
      );
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `/cat|mat/i.test(document.getElementById('out').textContent) && !document.getElementById('readout').hidden`,
        180_000,
        `${label} rerun`,
      );
      const out2 = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('out').textContent.trim()`,
      );
      mark(/cat/i.test(out2), "rerun-after-edit", out2.slice(0, 90));
    } else if (rung === "basics") {
      await evaluate(cdp, sessionId, `document.querySelectorAll('.chip')[0].click()`);
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `!document.getElementById('readout').hidden && document.getElementById('out').textContent.trim().length > 3`,
        180_000,
        `${label} translation`,
      );
      const out = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('out').textContent.trim()`,
      );
      mark(
        /coffee|morning/i.test(out),
        "real-translation-rendered",
        out.slice(0, 90),
      );
      const pair = await evaluate(cdp, sessionId, `document.getElementById('rPair').textContent`);
      mark(/Dutch → English/.test(pair), "pair-readout", pair);
      // chip control switches the input and reruns
      await evaluate(cdp, sessionId, `document.querySelectorAll('.chip')[1].click()`);
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `/thank/i.test(document.getElementById('out').textContent)`,
        180_000,
        `${label} rerun`,
      );
      const out2 = await evaluate(
        cdp,
        sessionId,
        `document.getElementById('out').textContent.trim()`,
      );
      mark(/thank/i.test(out2), "chip-rerun", out2.slice(0, 90));
    } else if (rung === "practical") {
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#results tbody tr').length === 6`,
        240_000,
        `${label} batch`,
      );
      const rows = await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#results tbody tr')].map((tr) => tr.cells[1].textContent.trim())`,
      );
      mark(/save/i.test(rows[0]), "batch-row-1", rows[0].slice(0, 60));
      mark(
        /session/i.test(rows[2]),
        "batch-row-3",
        rows[2].slice(0, 60),
      );
      const n = await evaluate(cdp, sessionId, `document.getElementById('rN').textContent`);
      const ms = await evaluate(cdp, sessionId, `document.getElementById('rMs').textContent`);
      mark(n === "6" && /s$/.test(ms), "batch-readout", `${n} strings in ${ms}`);
    } else if (rung === "wild") {
      // Round-trip: primary pair (already cached) + reverse pair Xenova/opus-mt-en-nl downloaded
      // on demand by the worker — the honest pair-loading status appears while it transfers.
      await evaluate(cdp, sessionId, `document.querySelectorAll('.chip')[3].click()`); // plain sentence
      await evaluate(cdp, sessionId, `document.getElementById('run').click()`);
      await waitFor(
        cdp,
        sessionId,
        `document.querySelectorAll('#trip .rt-card.back').length >= 2`,
        12 * 60_000,
        `${label} round-trip (downloads reverse pair on first run)`,
        5_000,
      );
      const cards = await evaluate(
        cdp,
        sessionId,
        `[...document.querySelectorAll('#trip .rt-card')].map((c) => c.querySelector('.meta').textContent)`,
      );
      mark(
        cards.some((m) => /Dutch → English/.test(m)) && cards.some((m) => /English → Dutch/.test(m)),
        "both-hops-rendered",
        cards.join(" | "),
      );
      const sim = await evaluate(cdp, sessionId, `document.getElementById('rSim').textContent`);
      mark(/^\d+%$/.test(sim), "drift-similarity", sim);
      const hop = await evaluate(
        cdp,
        sessionId,
        `document.querySelector('#trip .rt-card.hop div:last-child')?.textContent || ''`,
      );
      mark(/bike|work|cycle|center|centre/i.test(hop), "forward-hop-real", hop.slice(0, 90));
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
        model: MODEL_ID,
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
