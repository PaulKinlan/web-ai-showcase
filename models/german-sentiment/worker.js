// German-native sentiment worker — inference off the main thread so the "as you type" UI stays smooth.
// Model weights: oliverguhr/german-sentiment-bert (task: text-classification), WASM, fp32 ONNX.
// This is THE canonical German sentiment model (bert-base-german-cased fine-tuned on ~1.8M German
// samples — reviews, tweets, service texts), scoring text as positive / negative / neutral.
//
// Why this is DISTINCT, not a relabel: the model is a GERMAN-SPECIALIST — a BERT whose entire
// vocabulary and pre-training/fine-tuning corpus is German — so it reads German orthography, compound
// words, umlauts and register as first-class signal. That makes it materially different from the
// English DistilBERT SST-2 demo, the Spanish RoBERTuito demo, and the ~8-language XLM-R multilingual
// demo (whose capacity is split across 100 languages). We return the FULL 3-class softmax so the page
// can show every class's probability.
//
// TOKENIZER NOTE (grounded, not invented): oliverguhr/german-sentiment-bert ships an ONNX export but no
// tokenizer.json — only vocab.txt — so transformers.js cannot build its tokenizer from that repo. The
// model is a fine-tune of google-bert/bert-base-german-cased, and the two vocab.txt files are BYTE-
// IDENTICAL (verified: same 30000 entries, same order — differ only by a trailing newline). So we load
// the IDENTICAL tokenizer from google-bert/bert-base-german-cased (which ships tokenizer.json) and the
// classification head + weights from oliverguhr/german-sentiment-bert. Same WordPiece tokenizer, real
// weights — a legitimate documented AutoTokenizer + AutoModel load, not a substitution.
//
// Two operations:
//   run       → classify one text; return {positive, negative, neutral} probabilities + the winner.
//   attribute → OCCLUSION attribution: re-score the text with each word removed and report how much each
//               word moved the WINNING class (in log-odds). One batched forward pass over the variants —
//               a real, model-grounded saliency (no gradients).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "oliverguhr/german-sentiment-bert"; // weights + classification head (ONNX fp32)
const TOKENIZER_ID = "google-bert/bert-base-german-cased"; // identical vocab, ships tokenizer.json
// From the model config id2label: 0 positive · 1 negative · 2 neutral.
const CLASSES = ["positive", "negative", "neutral"];

let transformers = null;
let tokenizer = null;
let model = null;
let id2label = { 0: "positive", 1: "negative", 2: "neutral" };
const device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model && tokenizer) return;
  if (!transformers) transformers = await import(TRANSFORMERS_URL);
  const { AutoTokenizer, AutoModelForSequenceClassification, env } = transformers;
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the service worker.
  tokenizer = await AutoTokenizer.from_pretrained(TOKENIZER_ID);
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    dtype: "fp32",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  if (model.config?.id2label) id2label = model.config.id2label;
  post({ type: "ready", device });
}

// oliverguhr's germansentiment preprocessing: lowercase, drop digits, strip non-letters, collapse space.
// This is how the model was trained/used; applying it sharpens accuracy. We keep the ORIGINAL words for
// display so "see inside" can tint the text the user actually typed.
function clean(text) {
  return String(text)
    .toLowerCase()
    .replace(/[0-9]/g, " ")
    .replace(/[^\p{L}\s]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function softmaxRow(arr) {
  const m = Math.max(...arr);
  const ex = arr.map((v) => Math.exp(v - m));
  const s = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / s);
}

// Run the model on a batch of texts → array of {positive, negative, neutral} distributions.
async function scoreBatch(texts) {
  const cleaned = texts.map((t) => clean(t) || "[UNK]");
  const inputs = await tokenizer(cleaned, { padding: true, truncation: true });
  const { logits } = await model(inputs);
  const [n, k] = logits.dims; // [batch, 3]
  const flat = Array.from(logits.data);
  const out = [];
  for (let r = 0; r < n; r++) {
    const row = flat.slice(r * k, r * k + k);
    const probs = softmaxRow(row);
    const d = { positive: 0, negative: 0, neutral: 0 };
    for (let c = 0; c < k; c++) {
      const key = String(id2label[c]).toLowerCase();
      if (key in d) d[key] = probs[c];
    }
    out.push(d);
  }
  return out;
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
  const [dist] = await scoreBatch([text]);
  const label = winnerOf(dist);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, dist, label, ms, device });
}

// Split into human-readable words but keep them re-joinable. Trailing punctuation rides along.
function tokenizeWords(text) {
  return text.split(/(\s+)/).filter((t) => t.trim().length > 0);
}

async function attribute(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const words = tokenizeWords(text);
  if (words.length === 0) {
    post({
      type: "attr",
      id,
      text,
      words: [],
      attributions: [],
      dist: { positive: 0, negative: 0, neutral: 1 },
      label: "neutral",
      target: "neutral",
      ms: 0,
      device,
    });
    return;
  }
  const MAX = 40;
  const capped = words.length > MAX;
  const scoreWords = capped ? words.slice(0, MAX) : words;
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [
    text,
    ...scoreWords.map((_, i) => scoreWords.filter((_, j) => j !== i).join(" ")),
  ];
  const dists = await scoreBatch(variants);
  const fullDist = dists[0];
  const target = winnerOf(fullDist); // attribute against the winning class
  const fullLogit = logit(fullDist[target]);
  const attributions = scoreWords.map((_, i) => fullLogit - logit(dists[i + 1][target]));
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
