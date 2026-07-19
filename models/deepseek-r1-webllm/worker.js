// DeepSeek-R1-Distill (WebLLM) worker — reasoning generation off the main thread via MLC's WebGPU
// engine. Runtime is WebLLM, not Transformers.js: we import the shared helpers from lib/webllm.js
// (createEngine / streamChat) so every WebLLM page shares one verified code path. WebLLM is
// WebGPU-ONLY — the page gates on webGPUAdapterAvailable() before it ever asks us to load.
//
// The model id is passed in the `load`/`run` message so one worker file can drive any MLC build
// (the multi-model page spins up a second worker for a different model with the same code).

import { createEngine, streamChat } from "/web-ai-showcase/lib/webllm.js";

const DEFAULT_MODEL = "DeepSeek-R1-Distill-Llama-8B-q4f16_1-MLC";
let engine = null;
let loadedModel = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded(modelId) {
  const id = modelId || DEFAULT_MODEL;
  if (engine && loadedModel === id) return;
  loadedModel = id;
  engine = await createEngine({
    model: id,
    onProgress: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready" });
}

async function run(id, req, modelId) {
  await ensureLoaded(modelId);
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
      await ensureLoaded(e.data.modelId);
    } else if (type === "run") {
      await run(e.data.id, e.data.req, e.data.modelId);
    } else if (type === "stop") {
      if (engine && typeof engine.interruptGenerate === "function") {
        engine.interruptGenerate();
      }
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
