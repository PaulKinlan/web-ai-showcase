// toxic-bert content-moderation worker — all inference off the main thread so the control UI stays smooth.
// Model: Xenova/toxic-bert (BertForSequenceClassification, task: text-classification), WASM, q8.
// This is MULTI-LABEL classification: the model emits 6 INDEPENDENT logits (toxic, severe_toxic, obscene,
// threat, insult, identity_hate). Each goes through its OWN sigmoid — the scores do NOT sum to 1, and a
// single comment can be high on several labels at once (a threat that is also an insult). That is why we
// run the LOW-LEVEL forward (tokenizer → model → logits) instead of the convenience pipeline: one pass
// yields the raw logits AND the per-label sigmoid probabilities the "see inside" surface needs.
//
// This is defensive content-safety tooling: it scores text so a moderation queue can triage it. It never
// leaves the device — the point is private, client-side moderation with no comment sent to a server.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/toxic-bert";
let tokenizer = null;
let model = null;
let device = "wasm";
let id2label = {};

function post(msg) {
  self.postMessage(msg);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  id2label = model.config.id2label || {};
  post({ type: "ready", device, labels: id2label });
}

// Score one text → per-label {label, score(sigmoid), logit}. Sorted high→low by score.
async function scoreText(text) {
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const raw = Array.from(logits.data); // [6] independent logits, pre-sigmoid
  const labels = raw.map((v, i) => ({
    label: id2label[i] ?? `label ${i}`,
    logit: v,
    score: sigmoid(v),
  }));
  labels.sort((a, b) => b.score - a.score);
  return labels;
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const labels = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, labels, ms, device });
}

// Batch triage: score many texts at once (still one at a time through the model, but off-main-thread and
// reported as a single job). Returns per-item top label + score so a queue can be sorted/triaged.
async function triage(id, texts, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const items = [];
  for (const text of texts) {
    const labels = await scoreText(text);
    const top = labels[0];
    const flagged = labels.filter((l) => l.score >= threshold);
    items.push({
      text,
      top,
      labels,
      flagged: flagged.map((l) => l.label),
      flag: flagged.length > 0,
    });
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "triage", id, items, ms, device });
}

// Moderation GATE for multi-model composition: is this text clean (below threshold on every label)?
async function gate(id, text, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const labels = await scoreText(text);
  const maxScore = Math.max(...labels.map((l) => l.score));
  const ms = Math.round(performance.now() - t0);
  post({
    type: "gate",
    id,
    text,
    clean: maxScore < threshold,
    maxScore,
    labels,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "triage") await triage(e.data.id, e.data.texts, e.data.threshold);
    else if (type === "gate") await gate(e.data.id, e.data.text, e.data.threshold);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
