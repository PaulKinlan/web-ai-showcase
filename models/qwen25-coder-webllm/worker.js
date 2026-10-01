// Qwen2.5-Coder (WebLLM) worker — code generation off the main thread via MLC's WebGPU engine.
// Runtime is WebLLM, not Transformers.js: we import the shared helpers from lib/webllm.js
// (createEngine / streamChat) so every WebLLM page shares one verified code path. WebLLM is
// WebGPU-ONLY — the page gates on webGPUAdapterAvailable() before it ever asks us to load.
//
// The model id is passed in the `load`/`run` message so one worker file can drive any MLC build
// (the multi-model page spins up a second worker for a different model with the same code).

import { createEngine, retryPlan, streamChat } from "/web-ai-showcase/lib/webllm.js";

const DEFAULT_MODEL = "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC";
let engine = null;
let loadedModel = null;
let loadRaceRetries = 0;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded(modelId) {
  const id = modelId || DEFAULT_MODEL;
  if (engine && loadedModel === id) return;
  loadedModel = id;
  // The MLC weight-load path can lose the WebGPU buffer-mapping race (web-ai-showcase-w03); retry it
  // through the narrow, tested policy and keep the attempt count visible.
  const ATTEMPTS = 3;
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    try {
      engine = await createEngine({
        model: id,
        onProgress: (p) => post({ type: "progress", p }),
      });
      break;
    } catch (err) {
      const plan = retryPlan(err, attempt, { attempts: ATTEMPTS });
      if (!plan.retry) throw err;
      loadRaceRetries += 1;
      post({ type: "progress", p: { text: `engine load lost a GPU race — retry ${loadRaceRetries} of ${ATTEMPTS - 1}` } });
      await new Promise((resolve) => setTimeout(resolve, 500 * attempt));
    }
  }
  post({ type: "ready" });
}

async function run(id, req, modelId) {
  await ensureLoaded(modelId);
  const t0 = performance.now();
  let ttft = null;
  let chunks = 0;

  // The completion's OWN token count (web-ai-showcase-0ly): deltas are visible chunks, so the readouts
  // use this while `chunks` stays for the live rate.
  let tokens = null;
  let generationRaceRetries = 0;

  const text = await streamChat(engine, req, (delta) => {
    if (ttft === null) {
      ttft = performance.now() - t0;
      post({ type: "first", id, t: Math.round(ttft) });
    }
    chunks++;
    post({ type: "token", id, delta });
  }, (usage) => {
    if (typeof usage?.completion_tokens === "number") tokens = usage.completion_tokens;
  }, (retries) => {
    generationRaceRetries = retries;
    post({ type: "progress", p: { text: `generation lost a GPU race — retry ${retries} (the failed attempt streamed nothing)` } });
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
    tokens,
    stats,
    loadRaceRetries,
    generationRaceRetries,
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
      // WebLLM cooperatively interrupts the decode loop; the streamChat iterator then ends.
      if (engine && typeof engine.interruptGenerate === "function") {
        engine.interruptGenerate();
      }
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
