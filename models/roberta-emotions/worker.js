// RoBERTa GoEmotions worker — multi-label emotion classification off the main thread so the UI stays
// smooth. Model: SamLowe/roberta-base-go_emotions-onnx (task: text-classification), WASM, q8.
//
// Unlike a binary sentiment head, this is MULTI-LABEL: the model's config declares
// problem_type = "multi_label_classification", so Transformers.js applies a per-class SIGMOID (not a
// softmax). Every one of the 28 GoEmotions classes gets its OWN independent 0–1 score, and several can
// fire at once (a message can be both "gratitude" and "relief"). The scores do NOT sum to 1.
//
// Operations:
//   run   → score one text, return all 28 class scores (sorted, each an independent sigmoid).
//   batch → score many texts in a single padded forward pass (for the sentence-by-sentence arc and
//           the support-ticket router). Same shape, one array of 28-score rows per input.

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
    model: "SamLowe/roberta-base-go_emotions-onnx",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// top_k = 28 (the full label set) returns EVERY class score, sorted high→low. Because the head is
// multi-label, each score is an independent sigmoid — this is the real, unaltered model output.
async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const scores = await pipe(text, { top_k: 28 });
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
  // One padded forward pass over the whole batch — returns one 28-score row per input.
  const results = await pipe(texts, { top_k: 28 });
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
