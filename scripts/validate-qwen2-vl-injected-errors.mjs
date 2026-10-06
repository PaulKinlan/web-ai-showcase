#!/usr/bin/env node
// INJECTED worker-error-path acceptance for qwen2-vl (web-ai-showcase-oow).
//
// EVIDENCE CLASS: INJECTED. A real headless Chrome drives the real published pages, but
// models/qwen2-vl/worker.js is replaced (CDP Fetch) by a stub that answers probe/load/run and then
// posts the error message we choose. This proves the PAGE's error path (classifier -> labelled
// message -> demoted raw detail -> honest answer region) end to end. It does NOT prove the real ORT
// WebGPU [Concat] crash was reproduced on hardware; that needs a real shader-f16 GPU and is covered
// by scripts/validate-qwen2-vl-degradation.mjs when such hardware exists. No 2.7 GB download occurs.
//
// Discriminating: with the product fixes present every cell passes; with models/qwen2-vl/ reverted
// to the pre-fix files the labelled-message assertions FAIL (raw kernel text reaches the page).
// Cells: generation error on overview/basics/practical/wild x desktop/mobile; initialisation error on
// all five rungs x desktop/mobile. multi-model generation is NOT injected (stage 1 is a different model).
import {
  CDP,
  closePage,
  DESKTOP,
  evalValue,
  launchChrome,
  MOBILE,
  openPage,
  setViewport,
  startServer,
} from "./browser.mjs";

const RAW =
  'INJECTED [WebGPU] Kernel "[Concat] /model/layers.0/self_attn/Concat_7" failed. Error: Failed to generate kernel\'s output[0] with dims [1,2,286,128].';
const STUB = (mode) => `
const RAW = ${JSON.stringify(RAW)}; const MODE = ${JSON.stringify(mode)};
self.addEventListener("message", (e) => {
  const { type, id } = e.data;
  if (type === "probe") self.postMessage({ type: "probe-result", gpu: { ok: true, shaderF16: true } });
  else if (type === "load") {
    if (MODE === "init") self.postMessage({ type: "error", message: RAW });
    else self.postMessage({ type: "ready", device: "webgpu" });
  } else if (type === "run") {
    self.postMessage({ type: "prompt", id, template: "injected" });
    self.postMessage({ type: "error", id, message: RAW });
  }
});`;

const RUNGS = {
  overview: "models/qwen2-vl/",
  basics: "models/qwen2-vl/basics/",
  practical: "models/qwen2-vl/practical/",
  wild: "models/qwen2-vl/wild/",
  multimodel: "models/qwen2-vl/multi-model/",
};
let checks = 0, passed = 0, mode = "run";
const check = (l, c, d = "") => {
  checks++;
  if (c) passed++;
  console.log(`${c ? "PASS" : "FAIL"}  ${l}${d ? " — " + String(d).slice(0, 200) : ""}`);
  return c;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const waitFor = async (cdp, s, expr, ms) => {
  const t = Date.now() + ms;
  let v;
  while (Date.now() < t) {
    v = await evalValue(cdp, s, expr);
    if (v) return v;
    await sleep(500);
  }
  return v;
};
const LEAK = /\[WebGPU\]|\/model\/layers|\[Concat\]|Failed to generate kernel|INJECTED/;

const server = await startServer();
const chrome = await launchChrome({ webgpu: true });
const cdp = new CDP(chrome.ws);
let stubFired = 0;
cdp.on(async (msg) => {
  if (msg.method !== "Fetch.requestPaused") return;
  const { requestId } = msg.params;
  stubFired++;
  await cdp.send("Fetch.fulfillRequest", {
    requestId,
    responseCode: 200,
    responseHeaders: [{ name: "content-type", value: "text/javascript" }],
    body: Buffer.from(STUB(mode)).toString("base64"),
  }, msg.sessionId);
});
try {
  for (const [rung, route] of Object.entries(RUNGS)) {
    if (process.env.ONLY && !process.env.ONLY.split(",").includes(rung)) continue;
    for (const [vpName, vp] of [["desktop", DESKTOP], ["mobile", MOBILE]]) {
      for (const m of rung === "multimodel" ? ["init"] : ["run", "init"]) {
        mode = m;
        const cell = `${rung}@${vpName}/${
          m === "init" ? "init-error" : "generation-error"
        } [INJECTED]`;
        console.log(`\n=== ${cell} ===`);
        const page = await openPage(cdp, "about:blank");
        try {
          await cdp.send("Network.setBypassServiceWorker", { bypass: true }, page.sessionId);
          await cdp.send("Network.setCacheDisabled", { cacheDisabled: true }, page.sessionId);
          await cdp.send("Fetch.enable", {
            patterns: [{ urlPattern: "*qwen2-vl/worker.js*", requestStage: "Request" }],
          }, page.sessionId);
          await setViewport(cdp, page.sessionId, vp);
          await cdp.send("Page.navigate", {
            url: `http://127.0.0.1:${server.port}/web-ai-showcase/${route}`,
          }, page.sessionId);
          const L = "document.querySelector('#model-loader .model-loader')";
          await waitFor(cdp, page.sessionId, `${L}?.dataset.state`, 20000);
          await sleep(1500);
          const state = await evalValue(cdp, page.sessionId, `${L}?.dataset.state`);
          if (state === "download-required" || state === "partial") {
            await evalValue(
              cdp,
              page.sessionId,
              "document.querySelector('#model-loader .loader-actions button')?.click()",
            );
          }
          if (m === "init") {
            const st = await waitFor(
              cdp,
              page.sessionId,
              `(()=>{const s=document.querySelector('#model-loader .status')?.textContent||'';return /initialisation failed/i.test(s)?s:null})()`,
              30000,
            );
            const detail = await evalValue(
              cdp,
              page.sessionId,
              "document.querySelector('#model-loader .status')?.nextElementSibling?.matches('details.err-detail') ? document.querySelector('#model-loader .status').nextElementSibling.querySelector('pre')?.textContent : null",
            );
            check(
              `${cell}: labelled init failure shown`,
              /WebGPU backend crashed/i.test(st ?? ""),
              st,
            );
            check(`${cell}: no raw kernel text in user-facing message`, !!st && !LEAK.test(st), st);
            check(`${cell}: advice is phase-neutral`, !/model loaded/i.test(st ?? ""), st);
            check(`${cell}: raw text demoted to details`, /INJECTED/.test(detail ?? ""), detail);
          } else {
            const ready = await waitFor(
              cdp,
              page.sessionId,
              `${L}?.dataset.state==='ready'?'y':null`,
              30000,
            );
            check(`${cell}: stubbed model reaches ready`, ready === "y", state);
            await evalValue(
              cdp,
              page.sessionId,
              "(()=>{const p=document.querySelector('#prompt');if(p&&!p.value)p.value='Describe this image.';document.querySelector('#run')?.click();return 1})()",
            );
            const st = await waitFor(
              cdp,
              page.sessionId,
              "(()=>{const s=document.querySelector('#status')?.textContent||'';return /failed/i.test(s)?s:null})()",
              30000,
            );
            const detail = await evalValue(
              cdp,
              page.sessionId,
              "document.querySelector('#status + details.err-detail pre')?.textContent||null",
            );
            const ans = await evalValue(
              cdp,
              page.sessionId,
              "(document.querySelector('#answer')??document.querySelector('#transcript')??document.querySelector('#streamA')??document.querySelector('#chat .turn.a'))?.textContent?.trim()||null",
            );
            check(
              `${cell}: labelled degradation is the message`,
              /WebGPU backend crashed/i.test(st ?? ""),
              st,
            );
            check(`${cell}: no raw kernel text in user-facing message`, !!st && !LEAK.test(st), st);
            check(`${cell}: raw text demoted to details`, /INJECTED/.test(detail ?? ""), detail);
            check(`${cell}: answer region honest`, /failed/i.test(ans ?? ""), ans);
          }
        } finally {
          await closePage(cdp, page.targetId);
        }
      }
    }
  }
  check("worker stub actually intercepted requests", stubFired > 0, `${stubFired}`);
} finally {
  chrome.kill();
  server.server.close();
}
console.log(`\n${passed}/${checks} checks passed [INJECTED — not a real-hardware run]`);
process.exit(passed === checks ? 0 : 1);
