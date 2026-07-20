// ERNIE-4.5-0.3B — text generation off the main thread via Transformers.js.
// Model: onnx-community/ERNIE-4.5-0.3B-ONNX (task: text-generation).
//
// ERNIE 4.5 is Baidu's 2025 open family; the 0.3B is its smallest post-trained chat model — a compact,
// strongly BILINGUAL (English + Chinese) instruct LLM at a size that runs anywhere. This ONNX build has
// the `ernie4_5` architecture, whose model class landed in Transformers.js **4.2.0** — newer than the
// repo's shared 3.7.5 pin — so this ONE worker imports 4.2.0 LOCALLY (the isolated version-pin escape
// hatch; the shared lib/webai.js and every other page stay on 3.7.5, and lib/model-cache.js is
// version-agnostic so auto-init still works). At ~0.36B params it runs on the UNIVERSAL WebAssembly/CPU
// path (dtype q8, onnx/model_quantized.onnx, ~430 MB); when a real WebGPU adapter is present the worker
// uses the WebGPU fast path (dtype q4f16, onnx/model_q4f16.onnx) instead. Either way the output is a
// real, on-device run — never a canned reply.

// Isolated version pin: ernie4_5 requires Transformers.js >= 4.2.0 (absent from the shared 3.7.5).
// Scoped to THIS worker only — do not bump shared lib/webai.js (precedent: sam2-segmentation/worker.js).
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "onnx-community/ERNIE-4.5-0.3B-ONNX";
let generator = null;
let mod = null;
let stopper = null;
let DEVICE = "wasm";
let DTYPE = "q8";

function post(msg) {
  self.postMessage(msg);
}

// Workers expose navigator.gpu; probe for a REAL adapter (existence alone is not enough — headless
// returns null). WebGPU → q4f16 fast path; otherwise the universal WASM/CPU path with q8.
async function pickBackend() {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return { device: "webgpu", dtype: "q4f16" };
    } catch { /* fall through to WASM */ }
  }
  return { device: "wasm", dtype: "q8" };
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  const picked = await pickBackend();
  DEVICE = picked.device;
  DTYPE = picked.dtype;
  console.log(
    `[ernie worker] loading ${MODEL_ID} on ${DEVICE} (${DTYPE}) via transformers.js 4.2.0`,
  );
  generator = await pipeline("text-generation", MODEL_ID, {
    device: DEVICE,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[ernie worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

async function chat(id, messages, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;

  // "See inside" — the exact templated prompt the model receives (ERNIE chat markup + roles).
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
    max_new_tokens: Math.max(1, Math.min(1024, opts?.maxTokens ?? 200)),
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

// Top-k next-token distribution: one real forward pass over the current context → last-position
// logits → softmax → top-k. Honest "see inside" — real probabilities computed on-device.
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
  post({ type: "topk-result", id, prompt, tokens: top });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "chat") {
      await chat(e.data.id, e.data.messages, e.data.opts);
    } else if (type === "topk") {
      await topk(e.data.id, e.data.messages, e.data.k ?? 12);
    } else if (type === "stop") {
      stopper?.interrupt?.();
    }
  } catch (err) {
    console.error("[ernie worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
