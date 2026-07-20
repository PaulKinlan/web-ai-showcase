// BLOOMZ-560m — multilingual instruction text-generation off the main thread via Transformers.js.
// Model: Xenova/bloomz-560m (task: text-generation).
//
// BLOOMZ is BigScience's multitask-finetuned BLOOM: a distinct decoder architecture (ALiBi positions,
// a 250k-token multilingual/multicode BPE vocab) trained to FOLLOW INSTRUCTIONS across 46 human
// languages and 13 programming languages, zero-shot, from a plain prompt (no chat template). It is a
// different model family from every LLM already in this showcase (SmolLM/Llama/Qwen/Gemma/Phi/…).
//
// LOAD NOTE (verified): this repo ships the LEGACY split-decoder ONNX layout (onnx/decoder_model_merged*
// with an external .onnx_data), NOT the modern unified model.onnx. So we pass
// `model_file_name: "decoder_model_merged"` — with dtype "q8" Transformers.js loads
// onnx/decoder_model_merged_quantized.onnx (~350 MB). Verified in headless Chrome (WASM): real,
// coherent, non-degenerate multilingual output — e.g. "Translate to French: The weather is nice today."
// → "Aujourd'hui, le temps s'améliore bien." At 560M it is a SMALL model, so facts can be wrong
// (disclosed on the page) — but the output is a real on-device run, never a canned reply.
//
// dtype: q8 runs on the UNIVERSAL WebAssembly/CPU path (no GPU). WebGPU is attempted as an optional
// accelerator with an honest fallback to the verified WASM q8 path.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/bloomz-560m";
const LOAD_OPTS = { dtype: "q8", model_file_name: "decoder_model_merged" };
let generator = null;
let mod = null;
let stopper = null;
let DEVICE = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// Workers expose navigator.gpu; probe for a REAL adapter (existence alone is not enough — headless
// returns null). We use q8 on both paths; WebGPU is optional with an honest fallback to WASM q8.
async function hasWebGPU() {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      return !!(await navigator.gpu.requestAdapter());
    } catch { /* fall through */ }
  }
  return false;
}

async function makePipeline(device) {
  const { pipeline } = mod;
  return pipeline("text-generation", MODEL_ID, {
    device,
    ...LOAD_OPTS,
    progress_callback: (p) => post({ type: "progress", p }),
  });
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  if (await hasWebGPU()) {
    try {
      DEVICE = "webgpu";
      console.log(`[bloomz worker] trying ${MODEL_ID} on webgpu (q8)`);
      generator = await makePipeline("webgpu");
    } catch (err) {
      console.warn("[bloomz worker] webgpu failed, falling back to WASM q8", err);
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      generator = null;
    }
  }
  if (!generator) {
    DEVICE = "wasm";
    console.log(`[bloomz worker] loading ${MODEL_ID} on wasm (q8)`);
    generator = await makePipeline("wasm");
  }
  console.log(`[bloomz worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

// Stream a completion for a PLAIN prompt string (BLOOMZ has no chat template — you instruct it in the
// prompt itself, in any of its 46 languages).
async function complete(id, prompt, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  post({ type: "prompt", id, template: prompt });

  let count = 0;
  const t0 = performance.now();
  let firstTokenMs = null;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      count++;
      if (firstTokenMs === null) firstTokenMs = performance.now() - t0;
      post({ type: "token", id, token, t: performance.now() - t0 });
    },
  });

  stopper = mod.InterruptableStoppingCriteria ? new mod.InterruptableStoppingCriteria() : null;
  const doSample = opts?.doSample ?? false;
  const genOpts = {
    max_new_tokens: Math.max(1, Math.min(256, opts?.maxTokens ?? 64)),
    do_sample: doSample,
    repetition_penalty: opts?.repetitionPenalty ?? 1.3,
    streamer,
    return_full_text: false,
    ...(doSample
      ? { temperature: opts?.temperature ?? 0.7, top_p: opts?.topP ?? 0.9, top_k: opts?.topK ?? 50 }
      : {}),
    ...(stopper ? { stopping_criteria: stopper } : {}),
  };

  const out = await generator(prompt, genOpts);
  const ms = Math.round(performance.now() - t0);
  const full = out?.[0]?.generated_text;
  const text = typeof full === "string" ? full.slice(prompt.length) : String(full ?? "");
  post({
    type: "done",
    id,
    ms,
    tokens: count,
    text,
    device: DEVICE,
    firstTokenMs: firstTokenMs === null ? null : Math.round(firstTokenMs),
  });
}

// Top-k next-token distribution: one real forward pass over the prompt → last-position logits →
// softmax → top-k. Honest "see inside" — real probabilities computed on-device.
async function topk(id, prompt, k) {
  await ensureLoaded();
  const inputs = generator.tokenizer(prompt);
  const out = await generator.model(inputs);
  const logits = out.logits; // [1, seq, vocab]
  const dims = logits.dims;
  const vocab = dims.at(-1);
  const seq = dims.at(-2);
  const data = logits.data;
  const start = (seq - 1) * vocab;

  let max = -Infinity;
  for (let i = 0; i < vocab; i++) {
    const v = data[start + i];
    if (v > max) max = v;
  }
  let sum = 0;
  const probs = new Float64Array(vocab);
  for (let i = 0; i < vocab; i++) {
    const e = Math.exp(data[start + i] - max);
    probs[i] = e;
    sum += e;
  }
  const idx = Array.from({ length: vocab }, (_, i) => i);
  idx.sort((a, b) => probs[b] - probs[a]);
  const top = idx.slice(0, k).map((i) => ({
    id: i,
    token: generator.tokenizer.decode([i]),
    prob: probs[i] / sum,
    logit: data[start + i],
  }));
  post({ type: "topk-result", id, prompt, tokens: top });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "complete") await complete(e.data.id, e.data.prompt, e.data.opts);
    else if (type === "topk") await topk(e.data.id, e.data.prompt, e.data.k ?? 12);
    else if (type === "stop") stopper?.interrupt?.();
  } catch (err) {
    console.error("[bloomz worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
