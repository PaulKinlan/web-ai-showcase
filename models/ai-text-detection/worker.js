// AI-text detection worker — the GPT-2 output detector, off the main thread so as-you-type scoring never
// janks the UI. Model: onnx-community/roberta-base-openai-detector-ONNX (task: text-classification), a
// RoBERTa-base (125M) sequence classifier fine-tuned by OpenAI on GPT-2 1.5B outputs vs WebText, exported
// to ONNX (q8), WASM.
//
// The head is a single-label softmax over 2 classes from the real config.json: "Fake" (id 0 — reads
// machine-generated) and "Real" (id 1 — reads human-written). OpenAI report ~95% accuracy ON 1.5B GPT-2
// outputs at 510 tokens — NOT on ChatGPT/modern-LLM text, short snippets, or non-English prose. This is a
// triage SIGNAL, never a verdict, and must never be used to accuse anyone of misconduct.
//
// Operations:
//   run   → score one text, return P(Fake) + P(Real) (a real 2-class softmax, sums to 1).
//   batch → score many texts in one padded forward pass (for the triage queue + the wild gauntlet).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "text-classification",
    model: "onnx-community/roberta-base-openai-detector-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Recover P(Fake) — "reads machine-generated" — from a top_k:2 result (row order can vary; softmax, so
// the pair sums to 1).
function aiProb(rows) {
  const arr = Array.isArray(rows) ? rows : [rows];
  const fake = arr.find((r) => r.label === "Fake");
  if (fake) return fake.score;
  const real = arr.find((r) => r.label === "Real");
  return real ? 1 - real.score : 0.5;
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 2 });
  const ai = aiProb(out);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    ai,
    real: 1 - ai,
    label: ai >= 0.5 ? "Fake" : "Real",
    scores: out,
    ms,
    device,
  });
}

async function classifyBatch(id, texts) {
  await ensureLoaded();
  if (!texts.length) {
    post({ type: "batch", id, texts: [], results: [], ms: 0, device });
    return;
  }
  const t0 = performance.now();
  const out = await pipe(texts, { top_k: 2 });
  const results = texts.map((_, i) => {
    const ai = aiProb(out[i]);
    return { ai, real: 1 - ai, label: ai >= 0.5 ? "Fake" : "Real" };
  });
  const ms = Math.round(performance.now() - t0);
  post({ type: "batch", id, texts, results, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "batch") await classifyBatch(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
