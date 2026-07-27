// Phishing email / URL detection worker — inference off the main thread so the triage UI stays smooth.
// Model: onnx-community/phishing-email-detection-distilbert_v2.4.1-ONNX (task: text-classification),
// WASM, q8. DistilBERT fine-tuned by cybersectony on PhishingEmailDetectionv2.0 (apache-2.0).
// The checkpoint emits FOUR logits (id2label LABEL_0..3); the model card maps them as
//   0 = legitimate_email · 1 = phishing · 2 = legitimate_url · 3 = phishing_url_alt
// We report the THREAT probability = P(class 1) + P(class 3) and surface all four class scores.
//
// Operations:
//   run       → classify one message → threat + safe probabilities + label + the 4 class scores.
//   batch     → classify an array of messages (queue triage); posts each result as it lands so the
//               board fills in progressively — no blocking loop on the caller's main thread.
//   attribute → OCCLUSION attribution: re-score with each word removed and report how much each word
//               moved the THREAT probability (in log-odds, since the model is near-saturated). Removing
//               a word that pushed "phishing" drops the score → that word gets a threat (red) weight.
//               Real, model-grounded saliency — no gradients, just N+1 forward passes, batched.

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
    model: "onnx-community/phishing-email-detection-distilbert_v2.4.1-ONNX",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Recover the four class probabilities from a text-classification result row. top_k:4 returns every
// class; if a row is missing we treat it as 0 and rely on the softmax having summed to ~1.
// Card mapping: 0 legitimate_email, 1 phishing, 2 legitimate_url, 3 phishing_url_alt.
function classProbs(row) {
  const arr = Array.isArray(row) ? row : [row];
  const get = (i) => arr.find((r) => r.label === `LABEL_${i}`)?.score ?? 0;
  return { legitEmail: get(0), phish: get(1), legitUrl: get(2), phishUrl: get(3) };
}

function threatProb(row) {
  const p = classProbs(row);
  return p.phish + p.phishUrl;
}

// Log-odds of the THREAT classes. The model is near-saturated (p ≈ 0.999) on clear cases, so raw
// probability deltas from occlusion round to ~0. Log-odds makes each word's pull legible while staying
// a real, monotonic transform of the model's own confidence.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

function verdict(row) {
  const threat = threatProb(row);
  const p = classProbs(row);
  return {
    threat,
    safe: 1 - threat,
    label: threat >= 0.5 ? "PHISHING" : "LEGITIMATE",
    classes: [
      ["legitimate email", p.legitEmail],
      ["phishing", p.phish],
      ["legitimate url", p.legitUrl],
      ["phishing url", p.phishUrl],
    ],
  };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 4 });
  const v = verdict(out);
  post({ type: "result", id, text, ...v, ms: Math.round(performance.now() - t0), device });
}

async function batch(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  // Classify one message at a time and post each as it lands so the board fills progressively.
  // The heavy work is here in the worker; the page's main thread only renders small DOM updates.
  for (let i = 0; i < texts.length; i++) {
    const out = await pipe(texts[i], { top_k: 4 });
    const v = verdict(out);
    const item = { index: i, text: texts[i], threat: v.threat, safe: v.safe, label: v.label };
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
      threat: 0.5,
      label: "LEGITIMATE",
      ms: 0,
      device,
    });
    return;
  }
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [text, ...words.map((_, i) => words.filter((_, j) => j !== i).join(" "))];
  const out = await pipe(variants, { top_k: 4 });
  const fullThreat = threatProb(out[0]);
  const fullLogit = logit(fullThreat);
  // attribution_i = logit_threat(full) - logit_threat(without word i): positive → removing the word
  // dropped THREAT, so that word was pushing the message toward phishing.
  const attributions = words.map((_, i) => fullLogit - logit(threatProb(out[i + 1])));
  post({
    type: "attr",
    id,
    text,
    words,
    attributions,
    threat: fullThreat,
    label: fullThreat >= 0.5 ? "PHISHING" : "LEGITIMATE",
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
