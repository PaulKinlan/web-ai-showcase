// OLMo-2 1B Instruct (WebLLM) worker — streaming chat generation off the main thread via MLC's
// WebGPU engine. Runtime is WebLLM, not Transformers.js: engine creation goes through the shared
// helper in lib/webllm.js (createEngine) so every WebLLM page shares one verified load path. WebLLM
// is WebGPU-ONLY — the page gates on webGPUAdapterAvailable() before it ever asks us to load.
//
// Model: OLMo-2-0425-1B-Instruct-q4f16_1-MLC (Ai2 / Allen Institute for AI, 1B params,
// instruction-tuned). OLMo-2 is the DISTINCT "fully open" LLM in the catalogue: Ai2 releases not just
// the weights but the full training data (Dolma / Dolmino), the training code (OLMo-core), the
// intermediate checkpoints, and the training logs — a reproducible recipe, not a black box.
//
// The streaming loop posts each answer delta plus a running character count so the page can render the
// answer as it forms and show real tok/s + time-to-first-token. Nothing is ever faked: on a device
// without a WebGPU adapter the page's gate stops us before load, and no token is synthesised.

import { createEngine } from "/web-ai-showcase/lib/webllm.js";

const MODEL_ID = "OLMo-2-0425-1B-Instruct-q4f16_1-MLC";
let engine = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (engine) return;
  console.log(`[olmo-2 worker] creating MLC engine for ${MODEL_ID}`);
  engine = await createEngine({
    model: MODEL_ID,
    onProgress: (p) => post({ type: "progress", p }),
  });
  console.log(`[olmo-2 worker] engine ready`);
  post({ type: "ready" });
}

async function run(id, req) {
  await ensureLoaded();
  const { messages, temperature, top_p, max_tokens } = req;
  const t0 = performance.now();
  let ttft = null;
  let chunks = 0;

  const stream = await engine.chat.completions.create({
    messages,
    temperature,
    top_p,
    max_tokens,
    stream: true,
  });

  for await (const chunk of stream) {
    const delta = chunk.choices?.[0]?.delta?.content ?? "";
    if (!delta) continue;
    if (ttft === null) {
      ttft = performance.now() - t0;
      post({ type: "first", id, t: Math.round(ttft) });
    }
    chunks++;
    post({ type: "token", id, delta });
  }

  const ms = Math.round(performance.now() - t0);
  let stats = null;
  try {
    if (typeof engine.runtimeStatsText === "function") stats = await engine.runtimeStatsText();
  } catch {
    stats = null;
  }
  post({
    type: "done",
    id,
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
      if (engine && typeof engine.interruptGenerate === "function") engine.interruptGenerate();
    }
  } catch (err) {
    console.error("[olmo-2 worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
