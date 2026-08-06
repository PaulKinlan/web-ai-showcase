// DistilBERT SST-2 sentiment worker — inference off the main thread so the "as you type" UI stays smooth.
// Model: Xenova/distilbert-base-uncased-finetuned-sst-2-english (task: text-classification), WASM, q8.
//
// Two operations:
//   run       → classify one text, return POSITIVE and NEGATIVE probabilities.
//   attribute → OCCLUSION attribution: re-score the text with each word removed, and report how much
//               each word moved the POSITIVE probability. Removing a word that was pushing "positive"
//               drops the score (positive attribution); removing a negative word raises it. This is a
//               real, model-grounded saliency — no gradients, just N+1 forward passes, batched.

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
    model: "Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Normalize a text-classification result row to the POSITIVE probability. Note: with topk the pipeline
// may return only the winning label, so we recover POSITIVE from NEGATIVE when needed.
function posProb(row) {
  const arr = Array.isArray(row) ? row : [row];
  const p = arr.find((r) => r.label === "POSITIVE");
  if (p) return p.score;
  const n = arr.find((r) => r.label === "NEGATIVE");
  return n ? 1 - n.score : 0.5;
}

// Log-odds of the POSITIVE class. DistilBERT is near-saturated (p ≈ 0.999) on clear text, so raw
// probability deltas from occlusion round to ~0. Working in log-odds makes each word's pull legible
// while staying a real, monotonic transform of the model's own confidence.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { topk: 2 });
  const pos = posProb(out);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    pos,
    neg: 1 - pos,
    label: pos >= 0.5 ? "POSITIVE" : "NEGATIVE",
    ms,
    device,
  });
}

// Split into human-readable words but keep them re-joinable. Trailing punctuation rides with the word.
function tokenize(text) {
  return text.split(/(\s+)/).filter((t) => t.trim().length > 0);
}

async function attribute(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const words = tokenize(text);
  if (words.length === 0) {
    post({ type: "attr", id, text, words: [], attributions: [], pos: 0.5, label: "NEUTRAL", ms: 0, device });
    return;
  }
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [text, ...words.map((_, i) => words.filter((_, j) => j !== i).join(" "))];
  const out = await pipe(variants, { topk: 2 });
  const fullPos = posProb(out[0]);
  const fullLogit = logit(fullPos);
  // attribution_i = logit_pos(full) - logit_pos(without word i), in log-odds so near-saturated scores
  // still reveal each word's pull: positive → removing the word dropped POSITIVE, so it pushed POSITIVE.
  const attributions = words.map((_, i) => fullLogit - logit(posProb(out[i + 1])));
  const ms = Math.round(performance.now() - t0);
  post({
    type: "attr",
    id,
    text,
    words,
    attributions,
    pos: fullPos,
    label: fullPos >= 0.5 ? "POSITIVE" : "NEGATIVE",
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "attribute") await attribute(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
