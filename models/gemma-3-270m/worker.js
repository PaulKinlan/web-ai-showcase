// Gemma 3 270M (instruction-tuned) — text generation off the main thread via Transformers.js.
// Model: onnx-community/gemma-3-270m-it-ONNX (task: text-generation).
//
// Gemma 3 270M is Google's SMALLEST Gemma 3 (2025) — a ~270M-parameter instruction-tuned chat model,
// the "runs anywhere" tier of the Gemma 3 family. This ONNX build has the `gemma3_text` architecture,
// whose model class landed in Transformers.js **4.2.0** — newer than the repo's shared 3.7.5 pin — so
// this ONE worker imports 4.2.0 LOCALLY (the isolated version-pin escape hatch; the shared lib/webai.js
// and every other page stay on 3.7.5, and lib/model-cache.js is version-agnostic so auto-init still
// works). Precedent: models/sam2-segmentation/worker.js and models/ernie-4-5-0-3b/worker.js.
//
// DTYPE / BACKEND (verified in headless Chrome):
//   • WebGPU present → q4f16 (onnx/model_q4f16.onnx, ~200 MB) — the fast path.
//   • WASM/CPU (no adapter) → **fp32** (onnx/model.onnx, ~1.05 GB). This is deliberate and honest: the
//     quantized exports (q4, q8) use the `GatherBlockQuantized` embedding op, which ORT-Web only
//     implements on the WebGPU EP — on the WASM EP they abort at session creation ("Could not find an
//     implementation for GatherBlockQuantized"). fp16/q4f16 likewise need WebGPU's fp16 kernels. So on
//     the universal CPU path fp32 is the only export that runs — larger, but real, coherent output.
// Either way the output is a real, on-device run — never a canned reply.

// Isolated version pin: gemma3_text requires Transformers.js >= 4.2.0 (absent from the shared 3.7.5).
// Scoped to THIS worker only — do not bump shared lib/webai.js.
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "onnx-community/gemma-3-270m-it-ONNX";
let generator = null;
let mod = null;
let stopper = null;
let DEVICE = "wasm";
let DTYPE = "fp32";

function post(msg) {
  self.postMessage(msg);
}

// Workers expose navigator.gpu; probe for a REAL adapter (existence alone is not enough — headless
// returns null). WebGPU → q4f16 fast path; otherwise the universal WASM/CPU path with fp32 (the only
// export whose ops all run on the WASM EP — the quantized builds need WebGPU's GatherBlockQuantized).
async function pickBackend() {
  if (typeof navigator !== "undefined" && "gpu" in navigator) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) return { device: "webgpu", dtype: "q4f16" };
    } catch { /* fall through to WASM */ }
  }
  return { device: "wasm", dtype: "fp32" };
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  const picked = await pickBackend();
  DEVICE = picked.device;
  DTYPE = picked.dtype;
  console.log(
    `[gemma3-270m worker] loading ${MODEL_ID} on ${DEVICE} (${DTYPE}) via transformers.js 4.2.0`,
  );
  generator = await pipeline("text-generation", MODEL_ID, {
    device: DEVICE,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[gemma3-270m worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

async function chat(id, messages, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;

  // "See inside" — the exact templated prompt the model receives (Gemma chat markup + roles). Gemma's
  // template folds a leading system message into the first user turn, so passing system is safe.
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
    console.error("[gemma3-270m worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
