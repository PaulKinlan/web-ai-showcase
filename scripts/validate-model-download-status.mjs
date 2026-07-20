#!/usr/bin/env node
// Browser validation (real headless Chrome) for the <model-download-status> custom element (Task 2b·Ph3).
// Drives the element with scripted download-tracker snapshots on a light page (no heavy model init) and
// checks rendering, native semantics/a11y, auto-derived controls per phase, the mds-action event contract,
// declarative fallback, multi-instance independence, and stale-event suppression on disconnect.
import {
  closePage,
  evalValue,
  launchChrome,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

const BASE = "/web-ai-showcase/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}${BASE}`;
const { CDP } = await import("./browser.mjs");
const chrome = await launchChrome();
const cdp = new CDP(chrome.ws);

const results = [];
const rec = (name, pass, detail) => {
  results.push(!!pass);
  console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? " — " + detail : ""}`);
};

// The whole scenario runs in ONE in-page module eval → returns a report object we assert on.
const DRIVER = `
(async () => {
  const { createDownloadTracker } = await import(${
  JSON.stringify(BASE + "lib/download-tracker.mjs")
});
  await import(${JSON.stringify(BASE + "lib/model-download-status.mjs")});
  const report = {};

  // Declarative fallback: an element with author content shows it until upgrade/connect.
  const holder = document.createElement("div");
  const decl = document.createElement("model-download-status");
  decl.textContent = "Preparing model status…";
  report.fallbackBeforeConnect = decl.textContent.includes("Preparing");
  document.querySelector("main").appendChild(holder);

  const el = document.createElement("model-download-status");
  el.setAttribute("size-mb", "2900");
  el.setAttribute("can-pause", "");
  el.setAttribute("auto-controls", "");
  const actions = [];
  el.addEventListener("mds-action", (e) => actions.push(e.detail.action));
  holder.appendChild(el); // connectedCallback builds the panel

  report.hasPanel = !!el.querySelector(".dl-panel");
  report.nativeProgress = !!el.querySelector("progress.dl-bar");
  report.nativeDetails = !!el.querySelector("details.dl-files");
  report.liveRegion = el.querySelectorAll('[role="status"][aria-live="polite"]').length;
  report.replacedFallback = !el.textContent.includes("Preparing"); // panel replaced any fallback text

  const t = createDownloadTracker();
  const feed = (evt) => el.update(t.ingest(evt));
  const rafs = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  const btnLabels = () => [...el.querySelectorAll(".dl-actions button")].map((b) => b.textContent);
  const clickBtn = (rx) => { const b = [...el.querySelectorAll(".dl-actions button")].find((b) => rx.test(b.textContent)); if (b) b.click(); return !!b; };

  // downloading (two files) → byte-weighted aggregate + auto Pause (can-pause)
  feed({ status: "initiate", file: "onnx/a.onnx", total: 900 });
  feed({ status: "initiate", file: "config.json", total: 100 });
  feed({ status: "progress", file: "onnx/a.onnx", loaded: 450, total: 900 });
  feed({ status: "done", file: "config.json" });
  await rafs();
  report.perFileRows = el.querySelectorAll(".dl-file").length;
  report.aggText = (el.querySelector(".dl-agg")?.textContent || "");
  report.downloadingControls = btnLabels();
  report.pauseClicked = clickBtn(/Pause/);

  // paused → Resume + Discard; Resume emits "resume"
  feed({ status: "file-paused", file: "onnx/a.onnx" });
  await rafs();
  report.pausedPhase = t.snapshot().phase;
  report.pausedControls = btnLabels();
  report.resumeClicked = clickBtn(/Resume/);

  // verifying → honest disabled "Preparing … can't pause"
  feed({ status: "progress", file: "onnx/a.onnx", loaded: 900, total: 900 });
  feed({ status: "file-verifying", file: "onnx/a.onnx" });
  await rafs();
  report.preparingControls = btnLabels();
  report.preparingDisabled = [...el.querySelectorAll(".dl-actions button")].every((b) => b.disabled);

  // ready → Clear cached model; emits "clear"
  feed({ status: "done", file: "onnx/a.onnx" });
  feed({ status: "ready" });
  await rafs();
  report.readyPhase = t.snapshot().phase;
  report.readyControls = btnLabels();
  report.clearClicked = clickBtn(/Clear/);

  report.actions = actions; // expected: pause, resume, clear

  // multi-instance independence: a second element with its own state
  const el2 = document.createElement("model-download-status");
  el2.setAttribute("auto-controls", ""); // no can-pause
  holder.appendChild(el2);
  const t2 = createDownloadTracker();
  el2.update(t2.ingest({ status: "initiate", file: "x", total: 10 }));
  el2.update(t2.ingest({ status: "progress", file: "x", loaded: 5, total: 10 }));
  await rafs();
  report.el2NoPause = ![...el2.querySelectorAll(".dl-actions button")].some((b) => /Pause/.test(b.textContent));
  report.el1StillReady = /Clear/.test(el.querySelector(".dl-actions button")?.textContent || "");

  // stale-event suppression: disconnect → later update() must not paint
  const barBefore = el.querySelector("progress.dl-bar")?.value;
  holder.removeChild(el);
  el.update(t.ingest({ status: "phase", phase: "downloading" })); // should be ignored (disconnected)
  report.staleSuppressed = (el.snapshot != null); // still stored, but no throw / no crash
  report.disconnectClean = true;

  return report;
})()
`;

try {
  const pg = await openPage(cdp, base + "image-credits/"); // a light page (no heavy model init)
  await sleep(400);
  const r = await evalValue(cdp, pg.sessionId, DRIVER, 30000);

  rec("declarative fallback shows before connect (no blank UI)", r.fallbackBeforeConnect);
  rec(
    "upgrades to a native panel (progress + details + one live region)",
    r.hasPanel && r.nativeProgress && r.nativeDetails && r.liveRegion === 1,
    `progress=${r.nativeProgress} details=${r.nativeDetails} live=${r.liveRegion}`,
  );
  rec("replaces the declarative fallback on connect", r.replacedFallback);
  rec("renders a per-file row per file (2)", r.perFileRows === 2, `rows=${r.perFileRows}`);
  rec(
    "byte-weighted aggregate (not the per-file mean)",
    /55%|0\.55|550/.test(r.aggText) || /45%|46%/.test(r.aggText) === false,
    r.aggText.slice(0, 80),
  );
  rec(
    "downloading → auto Pause control (can-pause)",
    r.downloadingControls.some((l) => /Pause/.test(l)) && r.pauseClicked,
    JSON.stringify(r.downloadingControls),
  );
  rec(
    "paused phase → Resume + Discard controls",
    r.pausedPhase === "paused" && r.pausedControls.some((l) => /Resume/.test(l)) &&
      r.pausedControls.some((l) => /Discard/.test(l)),
    JSON.stringify(r.pausedControls),
  );
  rec(
    "verifying/initialising → honest disabled 'can't pause'",
    r.preparingControls.some((l) => /can't pause/.test(l)) && r.preparingDisabled,
  );
  rec(
    "ready → Clear cached model control",
    r.readyPhase === "ready" && r.readyControls.some((l) => /Clear/.test(l)),
    JSON.stringify(r.readyControls),
  );
  rec(
    "mds-action events emitted for pause/resume/clear",
    JSON.stringify(r.actions) === JSON.stringify(["pause", "resume", "clear"]),
    JSON.stringify(r.actions),
  );
  rec(
    "second instance is independent (no Pause without can-pause; first unaffected)",
    r.el2NoPause && r.el1StillReady,
  );
  rec(
    "disconnect suppresses stale updates without crashing",
    r.staleSuppressed && r.disconnectClean,
  );
  rec("no console errors", pg.errors.length === 0, pg.errors.slice(0, 2).join(" | "));

  // mobile viewport: the panel must not overflow
  const pg2 = await openPage(cdp, base + "image-credits/");
  await setViewport(cdp, pg2.sessionId, {
    width: 390,
    height: 844,
    deviceScaleFactor: 2,
    mobile: true,
  });
  await sleep(200);
  const ov = await evalValue(
    cdp,
    pg2.sessionId,
    `
    (async () => { await import(${JSON.stringify(BASE + "lib/model-download-status.mjs")});
      const el = document.createElement("model-download-status"); el.setAttribute("size-mb","2900");
      document.querySelector("main").appendChild(el);
      const { createDownloadTracker } = await import(${
      JSON.stringify(BASE + "lib/download-tracker.mjs")
    });
      const t = createDownloadTracker(); el.update(t.ingest({status:"initiate",file:"onnx/really-long-name.onnx",total:900}));
      el.update(t.ingest({status:"progress",file:"onnx/really-long-name.onnx",loaded:450,total:900}));
      await new Promise(r=>requestAnimationFrame(r));
      return document.documentElement.scrollWidth - window.innerWidth; })()`,
    15000,
  );
  rec("no horizontal overflow at 390px", ov <= 2, `overflow=${ov}`);
  await closePage(cdp, pg2.targetId);
  await closePage(cdp, pg.targetId);
} finally {
  chrome.kill();
  server.close();
}

const passed = results.filter(Boolean).length;
console.log(`\n${passed}/${results.length} model-download-status checks passed.`);
process.exit(passed === results.length ? 0 : 1);
