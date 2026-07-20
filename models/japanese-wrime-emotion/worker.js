// Japanese WRIME emotion worker — inference off the main thread so the "as you type" UI stays smooth.
// Model: arajun/ruri-v3-30m-wrime-onnx (task: text-classification), WASM, fp32 (the repo's onnx/model.onnx
// full-precision graph — verified clean in the browser; fp16 exists too but WASM has no fp16 kernel path).
//
// This is a Ruri v3 (30M) model — a Japanese ModernBERT encoder (cl-nagoya/ruri-v3-pt-30m) fine-tuned on
// WRIME, the Japanese emotion corpus, to predict Plutchik's EIGHT emotions from Japanese text:
//   0 喜び joy · 1 悲しみ sadness · 2 期待 anticipation · 3 驚き surprise ·
//   4 怒り anger · 5 恐れ fear · 6 嫌悪 disgust · 7 信頼 trust
// It is JAPANESE-NATIVE: distinct from the English/Spanish/German polarity classifiers and from the
// ~8-language multilingual XLM-R (which reads 3-way negative/neutral/positive). Two things make it
// browser-runnable where most Japanese BERTs are not: (1) ModernBERT IS registered in transformers.js,
// and (2) its tokenizer is a SentencePiece/Llama tokenizer (tokenizer.json + tokenizer.model), NOT a
// MeCab/fugashi word-splitter — so it runs fully client-side with no native morphological analyser.
//
// We load the tokenizer + model MANUALLY (AutoModelForSequenceClassification) rather than the
// text-classification pipeline, so the "see inside" surface can read the EXACT 8-way logits (raw
// pre-softmax scores) and softmax them ourselves, and so we can show the SentencePiece token strip —
// how a spaceless Japanese sentence is split into subword pieces.
//
// Operations:
//   run       → classify one text → 8-emotion distribution + winner + raw logits + SentencePiece tokens.
//   attribute → OCCLUSION attribution: re-score the text with each SentencePiece piece removed and report
//               how much each piece moved the WINNING emotion (in log-odds). A real, model-grounded
//               saliency — no gradients, just N+1 forward passes — and it works on Japanese because the
//               tokenizer, not a word list, decides what a unit is.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "arajun/ruri-v3-30m-wrime-onnx";
// Fixed label order from the model config (id2label): index → emotion key.
const CLASSES = ["joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust"];

let tokenizer = null;
let model = null;
const device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the SW.
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    dtype: "fp32", // repo ships fp32 model.onnx; WASM has no fp16 compute kernel, so run full precision
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

// Softmax an 8-logit row into a {joy, sadness, …} distribution keyed by emotion.
function distOf(row) {
  let max = -Infinity;
  for (const v of row) if (v > max) max = v;
  let sum = 0;
  const ex = row.map((v) => {
    const e = Math.exp(v - max);
    sum += e;
    return e;
  });
  const d = {};
  CLASSES.forEach((c, i) => (d[c] = { prob: ex[i] / sum, logit: row[i] }));
  return d;
}

function winnerOf(dist) {
  let best = CLASSES[0];
  for (const c of CLASSES) if (dist[c].prob > dist[best].prob) best = c;
  return best;
}

// SentencePiece decodes word-start pieces with a leading "▁" (U+2581). Strip it for a clean chip; the
// token strip keeps the marker so you can SEE where the tokenizer put a boundary in a spaceless sentence.
function cleanPiece(raw) {
  return raw.replace(/▁/g, " ").trim();
}

function tokenStrip(ids) {
  const cls = tokenizer.cls_token_id, sep = tokenizer.sep_token_id;
  const bos = tokenizer.bos_token_id, eos = tokenizer.eos_token_id;
  return ids.map((id) => {
    const raw = tokenizer.decode([id]);
    const isSpecial = id === cls || id === sep || id === bos || id === eos;
    const wordStart = /▁/.test(raw);
    return { id, piece: isSpecial ? raw.trim() : (cleanPiece(raw) || raw), wordStart, isSpecial };
  });
}

async function runOne(text) {
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const { logits } = await model(inputs);
  const row = Array.from(logits.data, Number); // [1, 8] → 8
  const dist = distOf(row);
  const label = winnerOf(dist);
  return { text, dist, label, tokens: tokenStrip(ids), ids };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await runOne(text);
  post({
    type: "result",
    id,
    text: r.text,
    dist: r.dist,
    label: r.label,
    tokens: r.tokens,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

// Log-odds of a probability — keeps each piece's occlusion pull legible even when the model saturates.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

// Occlusion attribution over the winning emotion. We segment on the tokenizer's OWN content pieces
// (dropping the special CLS/SEP), decode each all-but-one variant back to text, and re-classify — an
// N+1-pass, model-grounded saliency that respects Japanese subword boundaries. Capped so a pasted
// paragraph can't spawn hundreds of passes.
async function attribute(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const base = await runOne(text);
  const target = base.label;
  const fullLogit = logit(base.dist[target].prob);

  // Content token positions (skip specials) from the base tokenisation.
  const content = [];
  base.tokens.forEach((t, i) => {
    if (!t.isSpecial) content.push({ pos: i, piece: t.piece, wordStart: t.wordStart });
  });
  const MAX = 40;
  const capped = content.length > MAX;
  const use = capped ? content.slice(0, MAX) : content;

  const attributions = [];
  for (const c of use) {
    // Rebuild the id sequence without this one token, decode to text, re-classify.
    const keptIds = base.ids.filter((_, i) => i !== c.pos);
    const variantText = tokenizer.decode(keptIds, { skip_special_tokens: true });
    const r = await runOne(variantText);
    attributions.push(fullLogit - logit(r.dist[target].prob));
  }

  post({
    type: "attr",
    id,
    text,
    pieces: use.map((c) => c.piece),
    wordStarts: use.map((c) => c.wordStart),
    attributions,
    dist: base.dist,
    label: target,
    target,
    capped,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

// Classify a batch of texts (writing variants, emotion probes). Returns per-text {text, dist, label}.
async function classifyMany(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  for (const text of texts) {
    const r = await runOne(text);
    results.push({ text: r.text, dist: r.dist, label: r.label });
  }
  post({ type: "many", id, results, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "attribute") await attribute(e.data.id, e.data.text);
    else if (type === "classifyMany") await classifyMany(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
