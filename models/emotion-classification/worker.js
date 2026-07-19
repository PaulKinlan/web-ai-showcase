// Emotion (7-class Ekman) worker — single-label emotion classification off the main thread so the
// "as you type" UI stays smooth. Model: onnx-community/emotion-english-distilroberta-base-ONNX
// (task: text-classification), WASM, q8. Weights are j-hartmann/emotion-english-distilroberta-base
// exported to ONNX by onnx-community.
//
// Unlike the 28-class GoEmotions head (multi-label sigmoid, several fire at once), this DistilRoBERTa
// head is SINGLE-LABEL: its config has the default problem_type, so Transformers.js applies a SOFTMAX
// across the 7 Ekman emotions (anger, disgust, fear, joy, neutral, sadness, surprise). The 7 scores
// COMPETE and sum to ~1.0 — there is exactly one winner. That's the whole distinction we show.
//
// Operations:
//   run   → score one text, return all 7 class scores (sorted high→low, a real softmax distribution).
//   batch → score many texts in one padded forward pass (for the line-by-line arc + the triage queue).

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
    model: "onnx-community/emotion-english-distilroberta-base-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// top_k = 7 (the full label set) returns EVERY class score, sorted high→low. Because the head is
// single-label, these are one softmax distribution — the real, unaltered model output.
async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const scores = await pipe(text, { top_k: 7 });
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, scores, ms, device });
}

async function classifyBatch(id, texts) {
  await ensureLoaded();
  if (!texts.length) {
    post({ type: "batch", id, texts: [], results: [], ms: 0, device });
    return;
  }
  const t0 = performance.now();
  // One padded forward pass over the whole batch — returns one 7-score softmax row per input.
  const results = await pipe(texts, { top_k: 7 });
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
