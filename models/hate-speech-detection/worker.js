// Hate-speech moderation worker — all inference off the main thread so the control UI stays smooth.
// Model: onnx-community/twitter-roberta-large-hate-latest-ONNX
// (cardiffnlp/twitter-roberta-large-hate-latest, RobertaForSequenceClassification, task:
// text-classification), WASM, q8.
//
// This is SINGLE-LABEL (softmax) classification over EIGHT classes: seven kinds of hate keyed to a
// PROTECTED CHARACTERISTIC plus a not_hate class:
//   hate_gender · hate_race · hate_sexuality · hate_religion · hate_origin · hate_disability · hate_age
//   · not_hate
// The eight scores pass through ONE softmax, so they SUM TO 1. "Hate vs not-hate" is therefore
// 1 − P(not_hate); when it fires, the argmax names WHICH protected group is targeted — a strictly
// narrower task than toxicity or general offensiveness.
//
// This is defensive content-safety tooling and an IMPERFECT signal: the model reads surface text, has no
// speaker model, and inherits its training biases (it can flag reclaimed language and miss coded hate).
// Nothing leaves the device — the point is private, client-side moderation with no comment sent anywhere.
//
// NOTE ON SIZE: this is the LARGE (355M-param) Cardiff model — the only real browser-loadable ONNX export
// of a protected-group hate classifier (the base variant and facebook/CNERG models ship no ONNX). The q8
// download is ~358 MB; the shared loader shows the size and never silently re-downloads it.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/twitter-roberta-large-hate-latest-ONNX";

// Human-friendly names for the model's id2label (hate_gender → "gender", etc.), index === class index.
const RAW_LABELS = [
  "hate_gender",
  "hate_race",
  "hate_sexuality",
  "hate_religion",
  "hate_origin",
  "hate_disability",
  "hate_age",
  "not_hate",
];
const NOT_HATE = "not_hate";

let tokenizer = null;
let model = null;
let device = "wasm";
let id2label = {};

function post(msg) {
  self.postMessage(msg);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
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
  post({ type: "ready", device, labels: RAW_LABELS });
}

// Score one text → per-class {label, score(softmax), logit} sorted high→low, plus the aggregate
// hateScore = 1 − P(not_hate) and the argmax (which may be not_hate).
async function scoreText(text) {
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const raw = Array.from(logits.data); // [8] logits
  const probs = softmax(raw);
  const labels = raw.map((v, i) => ({
    label: id2label[i] ?? RAW_LABELS[i] ?? `label ${i}`,
    logit: v,
    score: probs[i],
    index: i,
  }));
  const notHate = labels.find((l) => l.label === NOT_HATE);
  const hateScore = 1 - (notHate ? notHate.score : 0);
  const argmax = labels.reduce((a, b) => (b.score > a.score ? b : a));
  labels.sort((a, b) => b.score - a.score);
  return { labels, hateScore, argmax };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, ...r, ms, device });
}

// Batch triage: score many texts (one at a time through the model, off-main-thread, reported once).
async function triage(id, texts, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const items = [];
  for (const text of texts) {
    const r = await scoreText(text);
    items.push({
      text,
      argmax: r.argmax.label,
      hateScore: r.hateScore,
      labels: r.labels,
      flag: r.hateScore >= threshold,
    });
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "triage", id, items, ms, device });
}

// Moderation GATE for multi-model composition: is this text clean (hate probability below threshold)?
async function gate(id, text, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "gate",
    id,
    text,
    clean: r.hateScore < threshold,
    hateScore: r.hateScore,
    argmax: r.argmax.label,
    labels: r.labels,
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
