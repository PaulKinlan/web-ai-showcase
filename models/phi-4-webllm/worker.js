// Phi-4-mini-instruct (WebLLM) worker — chat generation off the main thread via MLC's WebGPU engine.
// Runtime is WebLLM, not Transformers.js: we import the shared helpers from lib/webllm.js
// (createEngine / streamChat) so every WebLLM page shares one verified code path. WebLLM is
// WebGPU-ONLY — the page gates on webGPUAdapterAvailable() before it ever asks us to load.
// Model: Phi-4-mini-instruct-q4f16_1-MLC (Microsoft Phi-4 family, 3.8B params, instruction-tuned).

import { createEngine, streamChat } from "/web-ai-showcase/lib/webllm.js";

const MODEL_ID = "Phi-4-mini-instruct-q4f16_1-MLC";
let engine = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (engine) return;
  engine = await createEngine({
    model: MODEL_ID,
    onProgress: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready" });
}

async function run(id, req) {
  await ensureLoaded();
  const t0 = performance.now();
  let ttft = null;
  let chunks = 0;

  const text = await streamChat(engine, req, (delta) => {
    if (ttft === null) {
      ttft = performance.now() - t0;
      post({ type: "first", id, t: Math.round(ttft) });
    }
    chunks++;
    post({ type: "token", id, delta });
  });

  const ms = Math.round(performance.now() - t0);
  // Real WebLLM runtime stats (authoritative prefill/decode tokens-per-second), if exposed.
  let stats = null;
  try {
    if (typeof engine.runtimeStatsText === "function") {
      stats = await engine.runtimeStatsText();
    }
  } catch {
    stats = null;
  }
  post({
    type: "done",
    id,
    text,
    ms,
    ttft: ttft === null ? ms : Math.round(ttft),
    chunks,
    stats,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.req);
    } else if (type === "stop") {
      // WebLLM cooperatively interrupts the decode loop; the streamChat iterator then ends.
      if (engine && typeof engine.interruptGenerate === "function") {
        engine.interruptGenerate();
      }
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
