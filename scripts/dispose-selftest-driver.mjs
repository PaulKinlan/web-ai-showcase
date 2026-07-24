#!/usr/bin/env node
// Verification driver for the shared loader's memory-release contract (lib/model-loader.js).
// Reuses scripts/browser.mjs (CDP + static server) to drive REAL headless Chrome through the whole
// Download → ready → Release → released → reload cycle on lib/__dispose-selftest__, asserting the real
// state machine and the dispose/onReady contract. No model is downloaded (synthetic load). Standalone
// evidence generator (not wired into the default gates), exit 0 = all pass.

import {
  BASE,
  CDP,
  closePage,
  evalValue,
  launchChrome,
  openPage,
  startServer,
} from "./browser.mjs";

const URL_PATH = BASE + "lib/__dispose-selftest__/index.html";
const tests = [];
const record = (name, pass, detail) => tests.push({ name, pass: !!pass, detail });

async function waitFor(cdp, sessionId, expr, { tries = 40, gap = 150 } = {}) {
  for (let i = 0; i < tries; i++) {
    if (await evalValue(cdp, sessionId, expr) === true) return true;
    await new Promise((r) => setTimeout(r, gap));
  }
  return false;
}
const state = (cdp, s) => evalValue(cdp, s, "window.__dispose.state()");
const labels = (cdp, s) =>
  evalValue(cdp, s, "JSON.stringify(window.__dispose.actionLabels())").then(JSON.parse);
const counters = (cdp, s) => evalValue(cdp, s, "window.__dispose.counters()");
const runEnabled = (cdp, s) => evalValue(cdp, s, "window.__dispose.runEnabled()");
const click = (cdp, s, text) =>
  evalValue(cdp, s, `window.__dispose.click(${JSON.stringify(text)})`);
const hasLabel = (arr, t) => arr.some((l) => l.toLowerCase().includes(t.toLowerCase()));

const { server, port } = await startServer();
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);
const url = `http://127.0.0.1:${port}${URL_PATH}`;
const { targetId, sessionId, errors, netFailures } = await openPage(cdp, url);

await waitFor(cdp, sessionId, "window.__disposeReady === true");

// 1) Initial state: fixture is uncached → real Download button, run disabled, nothing disposed.
await waitFor(cdp, sessionId, "window.__dispose.state() === 'download-required'");
{
  const s = await state(cdp, sessionId);
  const l = await labels(cdp, sessionId);
  const c = await counters(cdp, sessionId);
  const run = await runEnabled(cdp, sessionId);
  record(
    "initial download-required with Download button, run disabled, nothing loaded",
    s === "download-required" && hasLabel(l, "download") && !run && c.loadCalls === 0 &&
      c.disposeCalls === 0,
    `state=${s} labels=${JSON.stringify(l)} run=${run} loadCalls=${c.loadCalls}`,
  );
}

// 2) Download → ready: synthetic load resolves, onReady enables run, Release + Clear appear.
await click(cdp, sessionId, "download");
await waitFor(cdp, sessionId, "window.__dispose.state() === 'ready'");
{
  const l = await labels(cdp, sessionId);
  const c = await counters(cdp, sessionId);
  const run = await runEnabled(cdp, sessionId);
  record(
    "after Download: ready, Release+Clear controls, run enabled, one load, not disposed",
    hasLabel(l, "release from memory") && hasLabel(l, "clear cached") && run &&
      c.loadCalls === 1 && c.readyCalls === 1 && c.disposeCalls === 0 && c.instanceLive === true,
    `labels=${JSON.stringify(l)} run=${run} counters=${JSON.stringify(c)}`,
  );
}

// 3) Release from memory → released: dispose + onDispose fire once, run disabled, reload control appears.
await click(cdp, sessionId, "release from memory");
await waitFor(cdp, sessionId, "window.__dispose.state() === 'released'");
{
  const s = await state(cdp, sessionId);
  const l = await labels(cdp, sessionId);
  const c = await counters(cdp, sessionId);
  const run = await runEnabled(cdp, sessionId);
  record(
    "after Release: released state, dispose+onDispose once, instance gone, run disabled",
    s === "released" && c.disposeCalls === 1 && c.onDisposeCalls === 1 &&
      c.instanceLive === false && !run,
    `state=${s} counters=${JSON.stringify(c)} run=${run}`,
  );
  record(
    "released state offers 'Load model into memory' + keeps 'Clear cached model'",
    hasLabel(l, "load model into memory") && hasLabel(l, "clear cached"),
    `labels=${JSON.stringify(l)}`,
  );
}

// 4) Reload from cache → ready again: init (not a fresh dispose), run re-enabled, Release returns.
await click(cdp, sessionId, "load model into memory");
await waitFor(cdp, sessionId, "window.__dispose.state() === 'ready'");
{
  const l = await labels(cdp, sessionId);
  const c = await counters(cdp, sessionId);
  const run = await runEnabled(cdp, sessionId);
  record(
    "reload: ready again, second load, still exactly one dispose, run re-enabled, Release returns",
    c.loadCalls === 2 && c.readyCalls === 2 && c.disposeCalls === 1 && c.instanceLive === true &&
      run &&
      hasLabel(l, "release from memory"),
    `labels=${JSON.stringify(l)} run=${run} counters=${JSON.stringify(c)}`,
  );
}

record("no console errors", errors.length === 0, JSON.stringify(errors));
record("no network failures", netFailures.length === 0, JSON.stringify(netFailures));

await closePage(cdp, targetId);
chrome.kill();
server.close();

console.log("\n=== DISPOSE SELF-TEST ===");
for (const t of tests) console.log(`${t.pass ? "PASS" : "FAIL"}  ${t.name}  — ${t.detail}`);
const allPass = tests.every((t) => t.pass);
console.log(
  `\n${allPass ? "ALL PASS" : "FAILURES PRESENT"} (${
    tests.filter((t) => t.pass).length
  }/${tests.length})`,
);
process.exit(allPass ? 0 : 1);
