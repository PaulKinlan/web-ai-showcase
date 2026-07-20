// Apertus-v1.1-1.5B-Instruct (INT4-QAD) — text generation off the main thread.
// Model: onnx-community/Apertus-v1.1-1.5B-Instruct-QAD-INT4-ONNX (task: text-generation).
//
// Apertus is the fully-open, compliant, multilingual model family from the Swiss AI Initiative
// (SNAI — ETH Zurich + EPFL). This is the v1.1 1.5B instruct model, distilled + quantization-aware
// trained to INT4 (QAD) for constrained hardware — it runs here on the WebAssembly CPU backend.
//
// VERSION PIN (isolated escape hatch): the `apertus` model class lands in @huggingface/transformers
// 4.x — it is ABSENT from the shared 3.7.5 pin (loading on 3.7.5 throws
// "Unsupported model type: apertus"). Per the repo's isolated version-pin policy (precedent:
// models/sam2-segmentation/worker.js pins 4.2.0), we pin 4.2.0 LOCALLY in THIS worker only — the
// shared lib/webai.js stays 3.7.5, and lib/model-cache.js is version-agnostic so auto-init still works.
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "onnx-community/Apertus-v1.1-1.5B-Instruct-QAD-INT4-ONNX";
const DEVICE = "wasm";
const DTYPE = "q4"; // the QAD INT4 web build (onnx/model_q4.onnx). No q8 export exists in this repo.
let generator = null;
let mod = null;
let stopper = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  console.log(`[apertus worker] loading ${MODEL_ID} on ${DEVICE} (${DTYPE})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device: DEVICE,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[apertus worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

async function chat(id, messages, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;

  // "See inside" — the exact templated prompt the model receives (chat markup + roles).
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
    // Apertus authors recommend temperature 0.8, top_p 0.9.
    ...(doSample
      ? {
        temperature: opts?.temperature ?? 0.8,
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
    firstTokenMs: firstTokenMs === null ? null : Math.round(firstTokenMs),
  });
}

// Top-k next-token distribution: one real forward pass over the current context → last-position
// logits → softmax → top-k. Honest "see inside" — real probabilities, computed on the CPU.
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
    console.error("[apertus worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
