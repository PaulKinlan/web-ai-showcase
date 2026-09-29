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
import { retryPlan } from "/web-ai-showcase/lib/webllm-race-policy.mjs";

let engine = null;
let loadRaceRetries = 0;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (engine) return;
  console.log(`[olmo-2 worker] creating MLC engine for OLMo-2-0425-1B-Instruct-q4f16_1-MLC`);
  // The MLC weight-load path can lose a WebGPU buffer-mapping race
  // ('mapAsync … Buffer was unmapped before mapping was resolved') and that REJECTS the whole engine
  // load, leaving the page at "Failed." — a browser-level race above this page (web-ai-showcase-w03,
  // ~50% of loads for the RAG rung, seen across families). Retry the creation a bounded number of
  // times; every other error propagates unchanged, so a real load failure is still a load failure.
  const ATTEMPTS = 3;
  for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
    try {
      engine = await createEngine({
        model: "OLMo-2-0425-1B-Instruct-q4f16_1-MLC",
        onProgress: (p) => post({ type: "progress", p }),
      });
      break;
    } catch (err) {
      const plan = retryPlan(err, attempt, { attempts: ATTEMPTS });
      if (!plan.retry) throw err;
      loadRaceRetries += 1;
      post({
        type: "progress",
        p: { text: `engine load lost a GPU race — retry ${loadRaceRetries} of ${ATTEMPTS - 1}` },
      });
      await new Promise((resolve) => setTimeout(resolve, 500 * attempt));
    }
  }
  console.log(`[olmo-2 worker] engine ready`);
  post({ type: "ready" });
}

async function run(id, req) {
  await ensureLoaded();
  const { messages, temperature, top_p, max_tokens } = req;
  const GENERATION_ATTEMPTS = 3;
  let generationRaceRetries = 0;
  let ttft = null;
  let chunks = 0;
  // The completion's OWN token count (web-ai-showcase-0ly): deltas are visible chunks, so the
  // readouts use this while `chunks` stays for the live rate. WebLLM is OpenAI-compatible and emits
  // a final chunk carrying `usage` when stream_options.include_usage is requested.
  let tokens = null;
  let t0 = performance.now();

  for (let attempt = 1;; attempt++) {
    ttft = null;
    chunks = 0;
    tokens = null;
    t0 = performance.now();
    try {
      const stream = await engine.chat.completions.create({
        messages,
        temperature,
        top_p,
        max_tokens,
        stream: true,
        stream_options: { include_usage: true },
      });

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content ?? "";
        if (chunk.usage && typeof chunk.usage.completion_tokens === "number") {
          tokens = chunk.usage.completion_tokens;
        }
        if (!delta) continue;
        if (ttft === null) {
          ttft = performance.now() - t0;
          post({ type: "first", id, t: Math.round(ttft) });
        }
        chunks++;
        post({ type: "token", id, delta });
      }
      break;
    } catch (err) {
      // Only the known GPU race, only when this attempt streamed NOTHING (the policy enforces both):
      // a retry after partial output would duplicate text the reader has already seen, and any other
      // error must fail on the first attempt rather than being swallowed.
      const plan = retryPlan(err, attempt, { attempts: GENERATION_ATTEMPTS, emitted: chunks > 0 });
      if (!plan.retry) {
        post({ type: "error", id, message: String(err?.message ?? err) });
        return;
      }
      generationRaceRetries += 1;
      // A wedged engine does not heal by re-calling it: the race leaves its buffers unusable, so the
      // retry gets a REBUILT engine (fresh MLC instance, fresh GPU buffers) rather than the same one.
      try {
        await engine?.unload?.();
      } catch {
        // unloading is best-effort; a failed unload must not block the rebuild
      }
      engine = null;
      post({
        type: "progress",
        p: {
          text:
            `generation lost a GPU race — rebuilding the engine and retrying (${generationRaceRetries} of ${GENERATION_ATTEMPTS - 1})`,
        },
      });
      await ensureLoaded();
      await new Promise((resolve) => setTimeout(resolve, 500 * generationRaceRetries));
    }
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
    tokens,
    stats,
    // VISIBLE retries: a cell that only passed after losing races must be legible as such in the run
    // evidence (coord, w03), never indistinguishable from a clean first-try pass.
    loadRaceRetries,
    generationRaceRetries,
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
