// FinBERT financial-sentiment worker — inference off the main thread so the control UI stays smooth.
// Model: Xenova/finbert (task: text-classification), WASM, q8. Labels: positive / negative / neutral.
//
// FinBERT is BERT further pre-trained on a large financial corpus (Reuters TRC2) and then fine-tuned on
// the Financial PhraseBank, so it reads sentiment the way an analyst does: "the outlook was cut",
// "shares tumbled", "missed estimates", "impairment charge" register as NEGATIVE, while a general
// sentiment model (trained on movie reviews) often mislabels the same jargon as neutral/positive.
//
// Two operations:
//   run       → classify one text → probabilities for all three classes (softmax over 3 logits).
//   attribute → OCCLUSION attribution: re-score the text with each word removed and report how much
//               each word moved the NET sentiment (positive − negative), in log-odds. A real,
//               model-grounded saliency: no gradients, just N+1 forward passes, batched.

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
    model: "Xenova/finbert",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Normalize a text-classification result row to a {positive, negative, neutral} score map. top_k:3
// (or null) returns every class; we index by label so ordering never matters.
function scoreMap(row) {
  const arr = Array.isArray(row) ? row : [row];
  const m = { positive: 0, negative: 0, neutral: 0 };
  for (const r of arr) {
    const k = String(r.label).toLowerCase();
    if (k in m) m[k] = r.score;
  }
  return m;
}

function topLabel(m) {
  return Object.entries(m).sort((a, b) => b[1] - a[1])[0][0];
}

// Log-odds of a probability, clamped so occlusion deltas stay finite even near saturation.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

// Net-sentiment signal used for attribution: log-odds(positive) − log-odds(negative). Positive means
// the text leans bullish, negative means bearish; neutral text sits near zero.
function netSignal(m) {
  return logit(m.positive) - logit(m.negative);
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 3 });
  const scores = scoreMap(out);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    scores,
    label: topLabel(scores),
    net: netSignal(scores),
    ms,
    device,
  });
}

// Classify a batch of texts in a single pass → array of { text, scores, label, net }.
async function classifyBatch(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(texts, { top_k: 3 });
  const rows = texts.map((text, i) => {
    const scores = scoreMap(out[i]);
    return { text, scores, label: topLabel(scores), net: netSignal(scores) };
  });
  const ms = Math.round(performance.now() - t0);
  post({ type: "batch", id, rows, ms, device });
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
      scores: scoreMap([]),
      label: "neutral",
      ms: 0,
      device,
    });
    return;
  }
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [text, ...words.map((_, i) => words.filter((_, j) => j !== i).join(" "))];
  const out = await pipe(variants, { top_k: 3 });
  const fullScores = scoreMap(out[0]);
  const fullNet = netSignal(fullScores);
  // attribution_i = net(full) − net(without word i): a word whose removal drops the bullish signal was
  // pushing POSITIVE (green); a word whose removal raises it was pushing NEGATIVE (red).
  const attributions = words.map((_, i) => fullNet - netSignal(scoreMap(out[i + 1])));
  const ms = Math.round(performance.now() - t0);
  post({
    type: "attr",
    id,
    text,
    words,
    attributions,
    scores: fullScores,
    label: topLabel(fullScores),
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "batch") await classifyBatch(e.data.id, e.data.texts);
    else if (type === "attribute") await attribute(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
