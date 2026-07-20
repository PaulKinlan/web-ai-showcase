// Spanish-native sentiment worker — inference off the main thread so the "as you type" UI stays smooth.
// Model: Xenova/robertuito-sentiment-analysis (task: text-classification), WASM, q8.
// This is the ONNX export of pysentimiento/robertuito-sentiment-analysis: RoBERTuito — a RoBERTa
// pre-trained on ~500M SPANISH tweets (user/hashtag/emoji aware) — fine-tuned for sentiment on TASS.
//
// Why this is DISTINCT, not a relabel: RoBERTuito's whole vocabulary + pre-training corpus is Spanish,
// so it reads Spanish morphology, diminutives, regional slang and emoji as first-class signal. It is a
// SPANISH-SPECIALIST, unlike the English DistilBERT SST-2 demo and unlike the ~8-language XLM-R
// multilingual demo (whose capacity is split across 100 languages). We return the FULL 3-class softmax
// (NEG / NEU / POS from the model config) so the page can show every class's probability.
//
// Two operations:
//   run       → classify one text; return {negative, neutral, positive} probabilities + the winner.
//   attribute → OCCLUSION attribution: re-score the text with each word removed and report how much each
//               word moved the WINNING class (in log-odds). N+1 batched forward passes — a real, model-
//               grounded saliency (no gradients), and it works for Spanish slang/emoji because the
//               tokenizer, not a keyword list, decides what a "word" contributes.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/robertuito-sentiment-analysis";
// Fixed label order from the model config id2label: 0 NEG · 1 NEU · 2 POS.
const LABEL_MAP = { NEG: "negative", NEU: "neutral", POS: "positive" };
const CLASSES = ["negative", "neutral", "positive"];

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "text-classification",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Turn a text-classification result (rows of {label, score}) into a {negative,neutral,positive}
// distribution, mapping the model's NEG/NEU/POS labels to our internal keys by name (order never bites).
function distOf(row) {
  const arr = Array.isArray(row) ? row : [row];
  const d = { negative: 0, neutral: 0, positive: 0 };
  let known = 0;
  for (const r of arr) {
    const key = LABEL_MAP[String(r.label).toUpperCase()] || String(r.label).toLowerCase();
    if (key in d) {
      d[key] = r.score;
      known++;
    }
  }
  // If only the winner came back, spread the remainder so the bars still sum to ~1.
  if (known < CLASSES.length) {
    const remaining = Math.max(0, 1 - Object.values(d).reduce((a, b) => a + b, 0));
    const each = remaining / (CLASSES.length - known);
    for (const c of CLASSES) if (d[c] === 0) d[c] = each;
  }
  return d;
}

function winnerOf(dist) {
  let best = CLASSES[0];
  for (const c of CLASSES) if (dist[c] > dist[best]) best = c;
  return best;
}

// Log-odds of a probability — keeps each word's occlusion pull legible even when the model saturates.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 3 });
  const dist = distOf(out);
  const label = winnerOf(dist);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, dist, label, ms, device });
}

// Split into human-readable words but keep them re-joinable. Trailing punctuation and emoji ride along.
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
      dist: { negative: 0, neutral: 1, positive: 0 },
      label: "neutral",
      target: "neutral",
      ms: 0,
      device,
    });
    return;
  }
  // Cap the pass count so a pasted paragraph can't spawn hundreds of forward passes on the model.
  const MAX = 40;
  const capped = words.length > MAX;
  const scoreWords = capped ? words.slice(0, MAX) : words;
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [
    text,
    ...scoreWords.map((_, i) => scoreWords.filter((_, j) => j !== i).join(" ")),
  ];
  const out = await pipe(variants, { top_k: 3 });
  const fullDist = distOf(out[0]);
  const target = winnerOf(fullDist); // attribute against the winning class
  const fullLogit = logit(fullDist[target]);
  const attributions = scoreWords.map((_, i) => fullLogit - logit(distOf(out[i + 1])[target]));
  if (capped) { for (let i = MAX; i < words.length; i++) attributions.push(0); }
  const ms = Math.round(performance.now() - t0);
  post({
    type: "attr",
    id,
    text,
    words,
    attributions,
    dist: fullDist,
    label: target,
    target,
    capped,
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
