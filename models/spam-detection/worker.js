// Spam / ham text-classification worker — inference off the main thread so the triage UI stays smooth.
// Model: onnx-community/tanaos-spam-detection-v1-ONNX (task: text-classification), WASM, q8.
// Labels: not_spam / spam (binary softmax over 2 logits). We report the SPAM probability; ham = 1 - spam.
//
// Operations:
//   run       → classify one message → spam + ham probabilities + label.
//   batch     → classify an array of messages (inbox triage); posts each result as it lands so the
//               board fills in progressively — no blocking loop on the caller's main thread.
//   attribute → OCCLUSION attribution: re-score with each word removed and report how much each word
//               moved the SPAM probability (in log-odds, since the model is near-saturated). Removing a
//               word that pushed "spam" drops the score → that word gets a spam (red) weight. Real,
//               model-grounded saliency — no gradients, just N+1 forward passes, batched.

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
    model: "onnx-community/tanaos-spam-detection-v1-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Recover the SPAM probability from a text-classification result row. top_k:2 returns both classes;
// if only the winner comes back we reconstruct spam from not_spam (binary softmax ⇒ they sum to 1).
function spamProb(row) {
  const arr = Array.isArray(row) ? row : [row];
  const s = arr.find((r) => r.label === "spam" || r.label === "LABEL_1");
  if (s) return s.score;
  const h = arr.find((r) => r.label === "not_spam" || r.label === "LABEL_0");
  return h ? 1 - h.score : 0.5;
}

// Log-odds of the SPAM class. The model is near-saturated (p ≈ 0.999) on clear cases, so raw
// probability deltas from occlusion round to ~0. Log-odds makes each word's pull legible while staying
// a real, monotonic transform of the model's own confidence.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

function verdict(spam) {
  return {
    spam,
    ham: 1 - spam,
    label: spam >= 0.5 ? "SPAM" : "HAM",
  };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 2 });
  const v = verdict(spamProb(out));
  post({ type: "result", id, text, ...v, ms: Math.round(performance.now() - t0), device });
}

async function batch(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  // Classify one message at a time and post each as it lands so the board fills progressively.
  // The heavy work is here in the worker; the page's main thread only renders small DOM updates.
  for (let i = 0; i < texts.length; i++) {
    const out = await pipe(texts[i], { top_k: 2 });
    const v = verdict(spamProb(out));
    const item = { index: i, text: texts[i], ...v };
    results.push(item);
    post({ type: "batch-item", id, item });
  }
  post({ type: "batch", id, results, ms: Math.round(performance.now() - t0), device });
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
    post({
      type: "attr",
      id,
      text,
      words: [],
      attributions: [],
      spam: 0.5,
      label: "HAM",
      ms: 0,
      device,
    });
    return;
  }
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [text, ...words.map((_, i) => words.filter((_, j) => j !== i).join(" "))];
  const out = await pipe(variants, { top_k: 2 });
  const fullSpam = spamProb(out[0]);
  const fullLogit = logit(fullSpam);
  // attribution_i = logit_spam(full) - logit_spam(without word i): positive → removing the word dropped
  // SPAM, so that word was pushing the message toward spam.
  const attributions = words.map((_, i) => fullLogit - logit(spamProb(out[i + 1])));
  post({
    type: "attr",
    id,
    text,
    words,
    attributions,
    spam: fullSpam,
    label: fullSpam >= 0.5 ? "SPAM" : "HAM",
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "batch") await batch(e.data.id, e.data.texts);
    else if (type === "attribute") await attribute(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
