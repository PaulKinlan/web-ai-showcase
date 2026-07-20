// Granite-4.0-350M — text generation off the main thread via Transformers.js.
// Model: onnx-community/granite-4.0-350m-ONNX-web (task: text-generation).
//
// IBM Granite 4.0 is a DISTINCT, NEWER model family from the Granite-3.0 already in this showcase: a
// HYBRID architecture (model_type "granitemoehybrid" — interleaved Mamba-2 state-space layers with a
// few softmax-attention layers), designed for long context at a small memory footprint. The 350M
// "micro"-class checkpoint is an instruct/chat model. It is distinct in BOTH generation (Granite-4 vs
// Granite-3) AND architecture (hybrid Mamba vs transformer) from every LLM already built here.
//
// dtype NOTE (verified in headless Chrome, WASM): the q4 build (onnx/model_q4.onnx + _data, ~576 MB)
// runs real, coherent, non-degenerate output — e.g. "Name three primary colors." → "The primary colors
// are red, blue, and yellow." The hybrid Mamba ops run fine on the WASM EP (no abort). q4f16/fp16 are
// WebGPU-only paths. WebGPU is used as an optional accelerator with an honest fallback to the verified
// WASM q4 path. Either way, a real on-device run — never a canned reply.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/granite-4.0-350m-ONNX-web";
const DTYPE = "q4";
let generator = null;
let mod = null;
let stopper = null;
let DEVICE = "wasm";

function post(msg) {
  self.postMessage(msg);
}

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
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  if (await hasWebGPU()) {
    try {
      DEVICE = "webgpu";
      console.log(`[granite-4 worker] trying ${MODEL_ID} on webgpu (q4)`);
      generator = await makePipeline("webgpu");
    } catch (err) {
      console.warn("[granite-4 worker] webgpu failed, falling back to WASM q4", err);
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      generator = null;
    }
  }
  if (!generator) {
    DEVICE = "wasm";
    console.log(`[granite-4 worker] loading ${MODEL_ID} on wasm (q4)`);
    generator = await makePipeline("wasm");
  }
  console.log(`[granite-4 worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

async function chat(id, messages, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const template = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  post({ type: "prompt", id, template });

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
  const doSample = opts?.doSample ?? true;
  const genOpts = {
    max_new_tokens: Math.max(1, Math.min(1024, opts?.maxTokens ?? 256)),
    do_sample: doSample,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
    ...(doSample
      ? { temperature: opts?.temperature ?? 0.7, top_p: opts?.topP ?? 0.9, top_k: opts?.topK ?? 50 }
      : {}),
    ...(stopper ? { stopping_criteria: stopper } : {}),
  };
  const out = await generator(messages, genOpts);
  const ms = Math.round(performance.now() - t0);
  const full = out?.[0]?.generated_text;
  const text = Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "");
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

async function topk(id, messages, k) {
  await ensureLoaded();
  const prompt = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  const inputs = generator.tokenizer(prompt);
  const out = await generator.model(inputs);
  const logits = out.logits,
    dims = logits.dims,
    vocab = dims.at(-1),
    seq = dims.at(-2),
    data = logits.data;
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
    else if (type === "chat") await chat(e.data.id, e.data.messages, e.data.opts);
    else if (type === "topk") await topk(e.data.id, e.data.messages, e.data.k ?? 12);
    else if (type === "stop") stopper?.interrupt?.();
  } catch (err) {
    console.error("[granite-4 worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
