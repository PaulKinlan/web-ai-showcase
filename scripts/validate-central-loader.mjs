#!/usr/bin/env node
// Central-adoption validation (Task 2b · Phase 4): the shared createModelLoader now routes every runtime
// family's onProgress through the adapters → download-tracker → <model-download-status>, preserving
// auto-init. Driven with mock `load` functions (family-shaped onProgress) on a light page — no real model
// download — proving the central wiring works for Transformers.js, WebLLM, and MediaPipe without the demo
// pages changing.
import { closePage, evalValue, launchChrome, openPage, startServer } from "./browser.mjs";

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

const DRIVER = `
(async () => {
  const { createModelLoader } = await import(${JSON.stringify(BASE + "lib/model-loader.js")});
  const main = document.querySelector("main");
  const rafs = () => new Promise((r)=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  const sleep = (ms)=>new Promise(r=>setTimeout(r,ms));
  const uid = () => "probe/" + Math.floor(performance.now()) + "-" + Math.floor(performance.now()*7%9999);
  const report = {};

  async function run(runtime, emitDuring, emitFinal) {
    const mount = document.createElement("div"); main.appendChild(mount);
    let emit = null, resolveLoad = null;
    const done = new Promise((r)=>{ resolveLoad = r; });
    createModelLoader({
      mount,
      model: { modelId: uid(), runtime, dtype: "q8", sizeMB: 100, requiresWebGPU: false },
      load: async (onProgress) => { emit = onProgress; emitDuring(onProgress); await done; emitFinal(onProgress); return {}; },
      onReady: () => {}, onError: () => {},
    });
    // start() → inspectModel (absent for a fresh id) → Download button
    for (let i=0;i<20 && !emit;i++){ const b=[...mount.querySelectorAll(".loader-actions button")].find(x=>/Download/.test(x.textContent)); if(b){ b.click(); } await sleep(60); }
    await rafs();
    const comp = mount.querySelector("model-download-status");
    const status = mount.querySelector(".status");
    const during = {
      hasComp: !!comp,
      rows: comp ? comp.querySelectorAll(".dl-file").length : -1,
      agg: comp ? (comp.querySelector(".dl-agg")?.textContent||"") : "",
      runtimeOwned: comp?.snapshot?.aggregate?.runtimeOwned,
      statusHidden: status ? status.hidden : null,
      compVisible: comp ? !comp.hidden : false,
    };
    resolveLoad();
    // poll for the ready manage-controls (recordValidated hits IndexedDB; cold-start can be slow)
    let clearBtn = false;
    for (let i=0;i<30 && !clearBtn;i++){ await sleep(50); clearBtn = [...mount.querySelectorAll(".loader-actions button")].some(b=>/Clear/.test(b.textContent)); }
    await rafs();
    const after = { phase: comp?.snapshot?.phase, clearBtn };
    mount.remove();
    return { during, after };
  }

  // Transformers.js: real per-file byte events → per-file rows + byte-weighted aggregate
  report.tjs = await run("transformers.js",
    (e)=>{ e({status:"initiate",file:"onnx/a.onnx",total:900}); e({status:"initiate",file:"config.json",total:100}); e({status:"progress",file:"onnx/a.onnx",loaded:450,total:900}); e({status:"done",file:"config.json"}); },
    (e)=>{ e({status:"progress",file:"onnx/a.onnx",loaded:900,total:900}); e({status:"done",file:"onnx/a.onnx"}); });

  // WebLLM: overall fraction → runtime-owned aggregate (no fabricated per-file bytes)
  report.webllm = await run("webllm",
    (e)=>{ e({text:"Fetching param cache[1/20]",progress:0}); e({text:"…",progress:0.4}); },
    (e)=>{ e({text:"Loading model into memory",progress:1}); });

  // MediaPipe: file initiate/ready vocabulary → a single file → ready
  report.mediapipe = await run("mediapipe",
    (e)=>{ e({status:"initiate",file:"blaze_face_short_range.tflite"}); },
    (e)=>{ e({status:"ready"}); });

  return report;
})()
`;

try {
  const pg = await openPage(cdp, base + "image-credits/");
  await sleep(400);
  const r = await evalValue(cdp, pg.sessionId, DRIVER, 30000);

  // Transformers.js
  rec("TJS: createModelLoader renders the <model-download-status> panel", r.tjs.during.hasComp);
  rec(
    "TJS: routes real per-file byte events → 2 per-file rows",
    r.tjs.during.rows === 2,
    `rows=${r.tjs.during.rows}`,
  );
  rec(
    "TJS: byte-weighted aggregate (55%, not the per-file mean 75%)",
    /55%|550/.test(r.tjs.during.agg),
    r.tjs.during.agg.slice(0, 70),
  );
  rec(
    "TJS: status line hidden + panel visible during download (one live region)",
    r.tjs.during.statusHidden === true && r.tjs.during.compVisible,
    `statusHidden=${r.tjs.during.statusHidden} compVisible=${r.tjs.during.compVisible}`,
  );
  rec(
    "TJS: reaches ready + offers Clear cached model",
    r.tjs.after.phase === "ready" && r.tjs.after.clearBtn,
    JSON.stringify(r.tjs.after),
  );

  // WebLLM
  rec(
    "WebLLM: routes the overall fraction as a runtime-owned aggregate",
    r.webllm.during.runtimeOwned === true && r.webllm.during.rows === 0,
    `runtimeOwned=${r.webllm.during.runtimeOwned} rows=${r.webllm.during.rows}`,
  );
  rec(
    "WebLLM: no fabricated per-file bytes in the aggregate",
    !/\bMB\b|\bGB\b|\bKB\b/.test(r.webllm.during.agg) || r.webllm.during.rows === 0,
    r.webllm.during.agg.slice(0, 60),
  );
  rec("WebLLM: reaches ready + Clear", r.webllm.after.phase === "ready" && r.webllm.after.clearBtn);

  // MediaPipe
  rec(
    "MediaPipe: routes initiate/ready vocabulary → panel + ready",
    r.mediapipe.during.hasComp && r.mediapipe.after.phase === "ready",
    JSON.stringify(r.mediapipe.after),
  );
  rec("MediaPipe: Clear cached model offered after ready", r.mediapipe.after.clearBtn);

  rec(
    "no console errors across all three families",
    pg.errors.length === 0,
    pg.errors.slice(0, 2).join(" | "),
  );
  await closePage(cdp, pg.targetId);
} finally {
  chrome.kill();
  server.close();
}

const passed = results.filter(Boolean).length;
console.log(`\n${passed}/${results.length} central-loader checks passed.`);
process.exit(passed === results.length ? 0 : 1);
