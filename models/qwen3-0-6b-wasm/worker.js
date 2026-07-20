// Qwen3-0.6B — text generation off the main thread via Transformers.js.
// Model: onnx-community/Qwen3-0.6B-ONNX (task: text-generation).
//
// Qwen3-0.6B is the SMALLEST member of Alibaba's Qwen3 family (~0.6B params, 28 layers, hidden 1024).
// This showcase already has Qwen3-1.7B — but ONLY via WebLLM, which is WebGPU-ONLY. This demo is
// DISTINCT: the smallest Qwen3, running on the UNIVERSAL WebAssembly/CPU path with NO GPU, so it reaches
// almost any device. Qwen3 is a HYBRID REASONING model: the same weights run in a "thinking" mode
// (emits a <think>…</think> chain before the answer) or a fast "non-thinking" mode — toggled in the chat
// template. We surface both.
//
// dtype NOTE (verified in headless Chrome, WASM): the q8 build (onnx/model_quantized.onnx, ~618 MB) runs
// real, coherent, non-degenerate output — e.g. "Name three primary colors." → "<think>… the primary
// colors are red, blue, and yellow …". The q4 build ABORTS on the WASM EP (its GatherBlockQuantized op
// has no wasm kernel), and q4f16/fp16 are WebGPU-only paths — so we ship q8 on WASM. WebGPU is used as an
// optional accelerator with an honest fallback to the verified WASM q8 path. Either way, a real
// on-device run — never a canned reply.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Qwen3-0.6B-ONNX";
const DTYPE = "q8";
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
      console.log(`[qwen3-0.6b worker] trying ${MODEL_ID} on webgpu (q8)`);
      generator = await makePipeline("webgpu");
    } catch (err) {
      console.warn("[qwen3-0.6b worker] webgpu failed, falling back to WASM q8", err);
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      generator = null;
    }
  }
  if (!generator) {
    DEVICE = "wasm";
    console.log(`[qwen3-0.6b worker] loading ${MODEL_ID} on wasm (q8)`);
    generator = await makePipeline("wasm");
  }
  console.log(`[qwen3-0.6b worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

function buildInputs(messages, thinking) {
  // Qwen3 chat template supports enable_thinking. Return the templated string for "see inside".
  return generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
    enable_thinking: thinking !== false,
  });
}

async function chat(id, messages, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const thinking = opts?.thinking !== false;
  const template = buildInputs(messages, thinking);
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
      ? { temperature: opts?.temperature ?? 0.7, top_p: opts?.topP ?? 0.9, top_k: opts?.topK ?? 20 }
      : {}),
    ...(stopper ? { stopping_criteria: stopper } : {}),
  };
  // Pass enable_thinking through the tokenizer's chat template when the pipeline templates internally.
  const out = await generator(messages, {
    ...genOpts,
    chat_template_options: { enable_thinking: thinking },
  });
  const ms = Math.round(performance.now() - t0);
  const full = out?.[0]?.generated_text;
  const text = Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "");
  // Split reasoning (<think>…</think>) from the final answer for the "see inside" surface.
  let think = null, answer = text;
  const m = text.match(/^([\s\S]*?)<\/think>/);
  if (m) {
    think = m[1].replace(/^<think>/, "").trim();
    answer = text.slice(m[0].length).trim();
  }
  post({
    type: "done",
    id,
    ms,
    tokens: count,
    text,
    think,
    answer,
    device: DEVICE,
    firstTokenMs: firstTokenMs === null ? null : Math.round(firstTokenMs),
  });
}

// Real top-k next-token distribution over the current context (one forward pass) → honest "see inside".
async function topk(id, messages, k, thinking) {
  await ensureLoaded();
  const prompt = buildInputs(messages, thinking);
  const inputs = generator.tokenizer(prompt);
  const out = await generator.model(inputs);
  const logits = out.logits;
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
    else if (type === "chat") await chat(e.data.id, e.data.messages, e.data.opts);
    else if (type === "topk") {
      await topk(e.data.id, e.data.messages, e.data.k ?? 12, e.data.thinking);
    } else if (type === "stop") stopper?.interrupt?.();
  } catch (err) {
    console.error("[qwen3-0.6b worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
