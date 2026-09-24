#!/usr/bin/env node
// Probe (bead web-ai-showcase-fb9): does an fp16-family dtype actually EXECUTE on the WASM EP?
//
// Why this exists: lib/webai.js `pickDevice()` silently degrades webgpu -> wasm while passing the
// caller's `dtype` through unchanged. AGENTS.md records a measured precedent (jina-embeddings-v3)
// where "fp16 aborts at execution on the WASM EP [WebGPU-only fp16 compute]". If that generalises,
// every route that declares an fp16-family dtype and can reach the wasm path is a latent runtime
// abort on a no-WebGPU device. But bart-zero-shot's header claims a MEASURED q4f16-on-WASM success.
// Those two claims cannot both be general. This probe settles it with evidence instead of assertion.
//
// Method: headless Chrome (repo's own harness), a module worker, real transformers.js at the shared
// pin, device:"wasm" forced, one small model that genuinely ships q8 + fp16 + q4f16 builds. We RUN
// inference, not just session creation — an abort typically surfaces at execution, not load.
//
// Deterministic and honest: a failure is recorded as a failure with its error text. Downloads are
// real (that is the point), so it is bounded to one small model and a hard timeout per case.

import { rmSync, writeFileSync } from "node:fs";
import { CDP, closePage, launchChrome, openPage, startServer } from "./browser.mjs";

const MODEL = "Xenova/all-MiniLM-L6-v2"; // ships model_quantized(q8) / model_fp16 / model_q4f16
const TASK = "feature-extraction";
const TJS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";
const CASES = ["q8", "fp16", "q4f16"];
const TIMEOUT_MS = 180000;

const workerTpl = (dtype) => `
import { pipeline, env } from "${TJS}";
self.onmessage = async () => {
  const t0 = performance.now();
  try {
    env.allowLocalModels = false;
    const pipe = await pipeline("${TASK}", "${MODEL}", { device: "wasm", dtype: "${dtype}" });
    const loadMs = Math.round(performance.now() - t0);
    const t1 = performance.now();
    const out = await pipe("The quick brown fox jumps over the lazy dog.", { pooling: "mean", normalize: true });
    const inferMs = Math.round(performance.now() - t1);
    const data = Array.from(out.data.slice(0, 4));
    const finite = data.every((v) => Number.isFinite(v));
    const nonZero = data.some((v) => Math.abs(v) > 1e-6);
    self.postMessage({ ok: true, dtype: "${dtype}", loadMs, inferMs, dims: out.dims, sample: data, finite, nonZero });
  } catch (e) {
    self.postMessage({ ok: false, dtype: "${dtype}", err: String((e && e.message) || e).slice(0, 400) });
  }
};`;

const chrome = await launchChrome();
const { server, port } = await startServer();
const cdp = new CDP(chrome.ws);
const ctx = await openPage(cdp, `http://127.0.0.1:${port}/web-ai-showcase/`);
const probePath = new URL("../__probe_dtype_worker.js", import.meta.url);
const results = [];

for (const dtype of CASES) {
  writeFileSync(probePath, workerTpl(dtype));
  const expr =
    `await new Promise((resolve)=>{const w=new Worker('/web-ai-showcase/__probe_dtype_worker.js?d=${dtype}',{type:'module'});` +
    `const to=setTimeout(()=>{w.terminate();resolve({ok:false,dtype:'${dtype}',err:'timeout'});},${TIMEOUT_MS});` +
    `w.onmessage=(e)=>{clearTimeout(to);w.terminate();resolve(e.data);};` +
    `w.onerror=(e)=>{clearTimeout(to);resolve({ok:false,dtype:'${dtype}',err:'worker.onerror '+(e.message||'')});};w.postMessage(1);})`;
  const { result } = await cdp.send(
    "Runtime.evaluate",
    {
      expression:
        `(async()=>{try{return (${expr});}catch(e){return {ok:false,dtype:'${dtype}',err:String(e)};}})()`,
      awaitPromise: true,
      returnByValue: true,
    },
    ctx.sessionId,
    TIMEOUT_MS + 20000,
  );
  const r = result?.value ?? { ok: false, dtype, err: "no result" };
  results.push(r);
  console.log(dtype, "=>", JSON.stringify(r));
}

try {
  rmSync(probePath, { force: true });
} catch {}
await closePage(cdp, ctx.targetId);
chrome.kill();
server.close();

console.log("\n=== SUMMARY (device:wasm, transformers.js 3.7.5, " + MODEL + ") ===");
for (const r of results) {
  console.log(
    `  ${r.dtype.padEnd(7)} ${r.ok ? "RUNS" : "FAILS"}` +
      (r.ok
        ? ` load=${r.loadMs}ms infer=${r.inferMs}ms finite=${r.finite} nonZero=${r.nonZero}`
        : ` — ${r.err}`),
  );
}
writeFileSync(
  new URL("../reports/wasm-dtype-probe.json", import.meta.url),
  JSON.stringify(
    {
      generated: new Date().toISOString(),
      model: MODEL,
      task: TASK,
      transformers: "3.7.5",
      device: "wasm",
      results,
    },
    null,
    2,
  ) + "\n",
);
