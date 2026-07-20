// Multilingual toxicity / content-moderation worker — all inference off the main thread so the control
// UI stays smooth while typing. Model: onnx-community/bert-multilingual-toxicity-classifier-ONNX
// (BertForSequenceClassification, task: text-classification), WASM, q8.
//
// This is BINARY, SINGLE-LABEL classification: the model emits TWO logits and we apply a SOFTMAX, so
// the two probabilities SUM TO 1 — { neutral (idx 0), toxic (idx 1) }. (Contrast the English
// Xenova/toxic-bert demo, which is multi-LABEL: six independent sigmoids.) We run the low-level forward
// (tokenizer → model → logits) rather than the convenience pipeline so the "see inside" surface can
// show the raw logits next to the softmax probabilities and apply the visitor's own threshold.
//
// The model is bert-base-multilingual-cased fine-tuned on the TextDetox 2025 multilingual dataset, so
// ONE checkpoint moderates 15 languages (Arabic, Hindi, Chinese, Japanese, Hebrew, Russian, German,
// Spanish, French, …). It is defensive content-safety tooling: it scores text so a moderation queue
// can triage it, entirely on-device — no comment is ever sent to a server.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/bert-multilingual-toxicity-classifier-ONNX";
let tokenizer = null;
let model = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// Numerically-stable softmax over the two class logits.
function softmax2(logits) {
  const m = Math.max(...logits);
  const e = logits.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the service worker.
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}

// Score one text → { logits:[neutral, toxic], pNeutral, pToxic }.
async function scoreText(text) {
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const raw = Array.from(logits.data); // [2] logits: idx 0 neutral, idx 1 toxic
  const [pNeutral, pToxic] = softmax2(raw);
  return { logits: raw, pNeutral, pToxic };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const s = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    logits: s.logits,
    pNeutral: s.pNeutral,
    pToxic: s.pToxic,
    label: s.pToxic >= 0.5 ? "toxic" : "neutral",
    ms,
    device,
  });
}

// Batch triage: score many texts (one at a time through the model, but off-thread, reported as one job)
// so a moderation queue can be sorted highest-toxicity first.
async function triage(id, texts, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const items = [];
  for (const text of texts) {
    const s = await scoreText(text);
    items.push({ text, pToxic: s.pToxic, pNeutral: s.pNeutral, flag: s.pToxic >= threshold });
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "triage", id, items, ms, device });
}

// Moderation GATE for multi-model composition: is this text clean (toxic probability below threshold)?
async function gate(id, text, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const s = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({ type: "gate", id, text, clean: s.pToxic < threshold, pToxic: s.pToxic, ms, device });
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
