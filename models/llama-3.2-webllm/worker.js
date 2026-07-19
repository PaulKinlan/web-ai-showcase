// Llama 3.2 3B Instruct (WebLLM) worker — chat generation off the main thread via MLC's WebGPU
// engine. Runtime is WebLLM, not Transformers.js: we import the shared helpers from lib/webllm.js
// (createEngine / streamChat) so every WebLLM page shares one verified code path. WebLLM is
// WebGPU-ONLY — the page gates on a real GPU adapter (via createModelLoader) before it ever asks us
// to load, and shows an honest needs-WebGPU state otherwise (never a faked reply).
//
// Model: Llama-3.2-3B-Instruct-q4f16_1-MLC (Meta Llama 3.2, 3.21B params, instruction-tuned) — the
// BIGGER sibling of the built Llama-3.2-1B page. Same runtime, distinctly more reasoning headroom for
// tool-routing / structured output / persona consistency, at a ~1.9 GB download (cached after first
// load) needing ~2.3 GB of GPU memory. Verified present in webllm.prebuiltAppConfig (2026-07-19).
// Native Llama-3 chat template with a real `system` role — no system-prompt folding.

import { createEngine, streamChat } from "/web-ai-showcase/lib/webllm.js";

const MODEL_ID = "Llama-3.2-3B-Instruct-q4f16_1-MLC";
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
