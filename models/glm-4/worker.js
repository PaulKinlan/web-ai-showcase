// GLM-Edge-1.5B-Chat worker — text generation off the main thread, with honest WebGPU gating.
// Model: onnx-community/glm-edge-1.5b-chat-ONNX (task: text-generation).
//
// This is Zhipu AI's (Z.ai / THUDM) GLM family — the SAME "glm" architecture as GLM-4 (Transformers
// registers it as model_type "glm" → GlmForCausalLM), in the browser-feasible GLM-Edge 1.5B size.
// It is genuinely BILINGUAL (English + Chinese), which the demos lean on. Larger GLM-4 members
// (GLM-4-9B and up) have no browser-feasible ONNX — 9B q4 is ~5 GB — so GLM-Edge-1.5B is the member
// of the family that actually runs in a tab today. Nothing here is faked: if there's no WebGPU
// adapter, we surface the honest needs-WebGPU state (a WASM opt-in is offered — real, but slow).
//
// Primary path: WebGPU + dtype q4f16 (4-bit weights, fp16 compute, ~1 GB). WASM q4 (~1.3 GB) is an
// honest, clearly-slow, opt-in fallback. Uses the canonical Transformers.js text-generation pipeline
// + TextStreamer for streaming, and a raw model forward pass for the top-k next-token surface.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/glm-edge-1.5b-chat-ONNX";
let generator = null;
let mod = null;
let loadedDevice = null;
let stopper = null;

function post(msg) {
  self.postMessage(msg);
}

// Real capability check — navigator.gpu existing is NOT enough; the adapter must actually resolve.
async function probeGPU() {
  if (!("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
  if (!adapter) return { ok: false, reason: "no-adapter" };
  const shaderF16 = adapter.features?.has?.("shader-f16") ?? false;
  return { ok: true, shaderF16 };
}

async function ensureLoaded(device, dtype) {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  const dev = device === "wasm" ? "wasm" : "webgpu";
  const dt = dtype ?? (dev === "wasm" ? "q4" : "q4f16");
  console.log(`[glm worker] loading ${MODEL_ID} on ${dev} (${dt})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device: dev,
    dtype: dt,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  loadedDevice = dev;
  console.log(`[glm worker] ready on ${dev}`);
  post({ type: "ready", device: dev });
}

async function chat(id, messages, opts) {
  await ensureLoaded(opts?.device, opts?.dtype);
  const { TextStreamer } = mod;

  // "See inside" — the exact templated prompt the model receives (GLM chat markup + roles).
  const template = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  post({ type: "prompt", id, template });

  let count = 0;
  const t0 = performance.now();
  let ttft = null;
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      if (ttft === null) {
        ttft = performance.now() - t0;
        post({ type: "first", id, t: ttft });
      }
      count++;
      post({ type: "token", id, token, t: performance.now() - t0 });
    },
  });

  // Interruptable stop (canonical in the Transformers.js WebGPU chat demos).
  stopper = mod.InterruptableStoppingCriteria ? new mod.InterruptableStoppingCriteria() : null;

  const doSample = opts?.doSample ?? true;
  const genOpts = {
    max_new_tokens: Math.max(1, Math.min(1024, opts?.maxTokens ?? 256)),
    do_sample: doSample,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
    ...(doSample
      ? {
        temperature: opts?.temperature ?? 0.7,
        top_p: opts?.topP ?? 0.9,
        top_k: opts?.topK ?? 50,
      }
      : {}),
    ...(stopper ? { stopping_criteria: stopper } : {}),
  };

  const out = await generator(messages, genOpts);
  const ms = Math.round(performance.now() - t0);
  const full = out?.[0]?.generated_text;
  const text = Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "");
  post({ type: "done", id, ms, tokens: count, ttft: ttft === null ? ms : Math.round(ttft), text });
}

// Top-k next-token distribution: one real forward pass over the current context → last-position
// logits → softmax → top-k. This is the honest "see inside" — real probabilities, not a mock.
async function topk(id, messages, k) {
  await ensureLoaded();
  const prompt = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  const inputs = generator.tokenizer(prompt);
  const out = await generator.model(inputs);
  const logits = out.logits; // Tensor [1, seq, vocab]
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
  post({ type: "topk-result", id, prompt, tokens: top, vocab });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "probe") {
      post({ type: "probe-result", gpu: await probeGPU() });
    } else if (type === "load") {
      await ensureLoaded(e.data.device, e.data.dtype);
    } else if (type === "chat") {
      await chat(e.data.id, e.data.messages, e.data.opts);
    } else if (type === "topk") {
      await topk(e.data.id, e.data.messages, e.data.k ?? 12);
    } else if (type === "stop") {
      stopper?.interrupt?.();
    }
  } catch (err) {
    console.error("[glm worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
