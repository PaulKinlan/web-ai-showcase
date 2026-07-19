// Offensive-language moderation worker — all inference off the main thread so the control UI stays smooth.
// Model: onnx-community/hate_speech_en-ONNX (IMSyPP/hate_speech_en, BertForSequenceClassification,
// task: text-classification), WASM, q8.
//
// This is SINGLE-LABEL (softmax) classification over an ORDERED severity ladder of four classes:
//   0 acceptable · 1 inappropriate · 2 offensive · 3 violent
// The four scores pass through ONE softmax, so they SUM TO 1 — the model commits to a distribution over
// how offensive the text is, unlike toxic-bert's independent per-label sigmoids. That contrast is the
// point of the "see inside" surface. The IMSyPP config only ships generic LABEL_0..3 ids, so we map them
// to the model card's documented names here (acceptable/inappropriate/offensive/violent).
//
// This is defensive content-safety tooling: it scores text so a moderation queue can triage it. Nothing
// leaves the device — the point is private, client-side moderation with no comment sent to a server.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/hate_speech_en-ONNX";

// The model card's documented label meaning for LABEL_0..3 (config.json only carries generic ids).
const LABEL_NAMES = ["acceptable", "inappropriate", "offensive", "violent"];
// "Offensive-or-worse" = offensive + violent. These two classes are what a queue actually flags.
const FLAG_CLASSES = new Set(["offensive", "violent"]);

let tokenizer = null;
let model = null;
let device = "wasm";

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
  post({ type: "ready", device, labels: LABEL_NAMES });
}

// Score one text → per-class {label, score(softmax), logit} sorted high→low, plus the aggregate
// "offensiveScore" = P(offensive)+P(violent) and the argmax severity label.
async function scoreText(text) {
  const inputs = await tokenizer(text);
  const { logits } = await model(inputs);
  const raw = Array.from(logits.data); // [4] logits over the severity ladder
  const probs = softmax(raw);
  const labels = raw.map((v, i) => ({
    label: LABEL_NAMES[i] ?? `label ${i}`,
    logit: v,
    score: probs[i],
    index: i,
  }));
  const offensiveScore = labels
    .filter((l) => FLAG_CLASSES.has(l.label))
    .reduce((a, l) => a + l.score, 0);
  const argmax = labels.reduce((a, b) => (b.score > a.score ? b : a));
  labels.sort((a, b) => b.score - a.score);
  return { labels, offensiveScore, argmax };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, ...r, ms, device });
}

// Batch triage: score many texts (one at a time through the model, but off-main-thread and reported as a
// single job). Returns per-item severity + offensiveScore so a queue can be sorted/triaged.
async function triage(id, texts, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const items = [];
  for (const text of texts) {
    const r = await scoreText(text);
    items.push({
      text,
      argmax: r.argmax.label,
      offensiveScore: r.offensiveScore,
      labels: r.labels,
      flag: r.offensiveScore >= threshold,
    });
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "triage", id, items, ms, device });
}

// Moderation GATE for multi-model composition: is this text clean (offensive-or-worse below threshold)?
async function gate(id, text, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreText(text);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "gate",
    id,
    text,
    clean: r.offensiveScore < threshold,
    offensiveScore: r.offensiveScore,
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
