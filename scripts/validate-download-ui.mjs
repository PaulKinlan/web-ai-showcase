#!/usr/bin/env node
// Real-browser validation of the multi-file download UI (headless Chrome via the repo harness).
// Drives lib/download-tracker.mjs + lib/download-ui.mjs with a SCRIPTED multi-file event stream (the
// 2.9 GB PaliGemma shape, without the 2.9 GB) and asserts the rendered DOM: phases, byte-weighted
// aggregate, per-file rows, cached mix, a config-100% that does NOT read as model-ready, pause/resume
// controls, and stable layout. Also confirms the PaliGemma page + routes load clean with the new loader.

import { closePage, evalValue, launchChrome, openPage, startServer } from "./browser.mjs";
const BASE = "/web-ai-showcase/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const rec = (n, p, d) => {
  results.push({ n, p: !!p, d });
  console.log(`${p ? "PASS" : "FAIL"}  ${n}${d ? " — " + d : ""}`);
};

const { server, port } = await startServer();
const base = `http://127.0.0.1:${port}${BASE}`;
const chrome = await launchChrome();
const { CDP } = await import("./browser.mjs");
const cdp = new CDP(chrome.ws);

// The in-page driver: imports the real modules, feeds a scripted sequence, returns rendered state.
const DRIVER = (steps) => `
(async () => {
  const { createDownloadTracker } = await import(${
  JSON.stringify(BASE + "lib/download-tracker.mjs")
});
  const { createDownloadUI } = await import(${JSON.stringify(BASE + "lib/download-ui.mjs")});
  const mount = document.createElement('div');
  mount.style.cssText = 'container-type:inline-size'; document.body.append(mount);
  const ui = createDownloadUI({ mount, sizeMB: 2900 });
  const tracker = createDownloadTracker();
  window.__t = tracker; window.__ui = ui; window.__mount = mount;
  for (const s of ${JSON.stringify(steps)}) ui.update(tracker.ingest(s));
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))); // let the throttled paint run
  const q = (sel) => mount.querySelector(sel);
  return {
    phase: mount.querySelector('.dl-panel').dataset.phase,
    phaseText: q('.dl-phase').textContent,
    barValue: q('.dl-bar').hasAttribute('value') ? Number(q('.dl-bar').value) : null,
    agg: q('.dl-agg').textContent,
    rows: [...mount.querySelectorAll('.dl-file')].map(r => ({
      name: r.querySelector('.dl-fname').textContent,
      state: r.querySelector('.dl-fstate').textContent,
      bytes: r.querySelector('.dl-fbytes').textContent,
    })),
    liveRegions: mount.querySelectorAll('[role=status][aria-live]').length,
  };
})()`;

try {
  const pg = await openPage(cdp, base); // any same-origin page; we drive modules in-page
  await sleep(600);

  // 1) Small config reaches 100% first, big weights barely started → must NOT read as ready/100%.
  let st = await evalValue(
    cdp,
    pg.sessionId,
    DRIVER([
      { status: "initiate", file: "config.json", total: 500 },
      { status: "progress", file: "config.json", loaded: 500, total: 500 },
      { status: "done", file: "config.json" },
    ]),
  );
  rec(
    "config-100% is not model-ready",
    st.phase !== "ready" && st.barValue !== 100,
    JSON.stringify({ phase: st.phase, bar: st.barValue }),
  );
  rec("one live region only", st.liveRegions === 1, `${st.liveRegions}`);

  // 2) Full multi-file shape: cached embed + downloading vision + queued decoder → byte-weighted agg.
  const big = [
    { status: "initiate", file: "config.json", total: 500 },
    { status: "done", file: "config.json" },
    { status: "initiate", file: "onnx/embed_tokens_quantized.onnx", total: 600000000 },
    { status: "done", file: "onnx/embed_tokens_quantized.onnx", cached: true }, // cache hit, no bytes
    { status: "initiate", file: "onnx/vision_encoder_fp16.onnx", total: 800000000 },
    {
      status: "progress",
      file: "onnx/vision_encoder_fp16.onnx",
      loaded: 400000000,
      total: 800000000,
    },
    { status: "initiate", file: "onnx/decoder_model_merged_q4f16.onnx", total: 1400000000 },
    {
      status: "progress",
      file: "onnx/decoder_model_merged_q4f16.onnx",
      loaded: 0,
      total: 1400000000,
    },
  ];
  st = await evalValue(cdp, pg.sessionId, DRIVER(big));
  rec("phase is downloading", st.phase === "downloading", st.phase);
  rec("per-file rows for every file", st.rows.length === 4, `${st.rows.length}`);
  const embed = st.rows.find((r) => r.name.includes("embed"));
  rec("cached file shown as cached", embed && embed.state === "cached", JSON.stringify(embed));
  // byte-weighted: loaded = cached embed 600M + vision 400M ≈ 1.0G of 2.8G ≈ 36% — and crucially NOT
  // the per-file arithmetic mean (100+100+50+0)/4 = 62.5%.
  rec(
    "byte-weighted aggregate (~36%, not the 62% mean)",
    /3[0-9]%/.test(st.agg) && !/6[0-9]%/.test(st.agg),
    st.agg,
  );
  rec(
    "aggregate shows readable bytes + file count",
    /GB|MB/.test(st.agg) && /\/4 files/.test(st.agg),
    st.agg,
  );

  // 3) All done + ready → 100% + ready phase.
  st = await evalValue(
    cdp,
    pg.sessionId,
    DRIVER([
      ...big,
      {
        status: "progress",
        file: "onnx/vision_encoder_fp16.onnx",
        loaded: 800000000,
        total: 800000000,
      },
      { status: "done", file: "onnx/vision_encoder_fp16.onnx" },
      {
        status: "progress",
        file: "onnx/decoder_model_merged_q4f16.onnx",
        loaded: 1400000000,
        total: 1400000000,
      },
      { status: "done", file: "onnx/decoder_model_merged_q4f16.onnx" },
      { status: "ready" },
    ]),
  );
  rec(
    "ready → phase ready + bar 100",
    st.phase === "ready" && st.barValue === 100,
    JSON.stringify({ phase: st.phase, bar: st.barValue }),
  );

  // 4) Error surfaces on the page.
  st = await evalValue(
    cdp,
    pg.sessionId,
    DRIVER([
      { status: "initiate", file: "onnx/w.onnx", total: 1000 },
      { status: "error", file: "onnx/w.onnx", message: "network dropped" },
    ]),
  );
  const failed = st.rows.find((r) => r.state === "failed");
  rec(
    "file error surfaces in the UI",
    !!failed && /failed/.test(st.agg),
    JSON.stringify({ failed, agg: st.agg }),
  );

  // 5) Pause/Resume controls are real buttons (label honest: Resume, not a fake restart).
  const ctl = await evalValue(
    cdp,
    pg.sessionId,
    `(() => {
    window.__ui.setActions([{ label: "Pause download", onClick(){} }]);
    const before = [...window.__mount.querySelectorAll('.dl-actions button')].map(b=>b.textContent);
    window.__ui.setActions([{ label: "Resume download", onClick(){} }, { label: "Discard partial downloads", onClick(){} }]);
    const after = [...window.__mount.querySelectorAll('.dl-actions button')].map(b=>b.textContent);
    return { before, after };
  })()`,
  );
  rec(
    "pause→resume controls swap correctly (honest labels)",
    ctl.before.includes("Pause download") && ctl.after.includes("Resume download") &&
      ctl.after.includes("Discard partial downloads"),
    JSON.stringify(ctl),
  );

  await closePage(cdp, pg.targetId);

  // 6) PaliGemma page + routes load clean with the new resumable loader (fresh profile → Download state).
  for (
    const route of [
      "models/paligemma/",
      "models/paligemma/basics/",
      "models/paligemma/multi-model/",
    ]
  ) {
    const p = await openPage(cdp, base + route);
    await sleep(1600);
    const info = await evalValue(
      cdp,
      p.sessionId,
      `(() => {
      const panel = document.querySelector('.dl-panel');
      const attr = document.querySelector('.attribution');
      const overflow = document.documentElement.scrollWidth - window.innerWidth;
      return { hasPanel: !!panel, phase: panel?.dataset.phase, hasAttribution: !!attr, attrLinks: attr ? attr.querySelectorAll('a').length : 0, overflow };
    })()`,
    );
    rec(
      `${route} loads with resumable panel + near-top attribution`,
      info.hasPanel && info.hasAttribution && info.attrLinks >= 2 && info.overflow <= 1 &&
        p.errors.length === 0,
      JSON.stringify({ ...info, errors: p.errors.slice(0, 2) }),
    );
    await closePage(cdp, p.targetId);
  }
} finally {
  chrome.kill();
  server.close();
}

const passed = results.filter((r) => r.p).length;
console.log(`\n${passed}/${results.length} checks passed.`);
process.exit(passed === results.length ? 0 : 1);
