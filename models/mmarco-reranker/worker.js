// mMARCO mMiniLMv2 multilingual cross-encoder reranker worker — inference off the main thread.
// Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (XLMRobertaForSequenceClassification, num_labels=1),
// WASM, fp32 (→ onnx/model.onnx, ~470 MB).
//
// This is the MULTILINGUAL sibling of the classic English MS MARCO MiniLM cross-encoder that also lives
// on this site (ms-marco-MiniLM-L-6-v2). Where that one is an English BERT trained on the English MS MARCO
// passage-ranking data, THIS model is a 12-layer XLM-RoBERTa MiniLMv2 (distilled from XLM-R Large) trained
// on mMARCO — MS MARCO machine-translated into 13 more languages (ar, zh, nl, fr, de, hi, id, it, ja, pt,
// ru, es, vi + en). So one model reranks a query against passages in any of 14 languages, and does it
// cross-lingually (an English query can promote a Spanish or Japanese passage). It is Apache-2.0 —
// commercially usable, unlike the CC-BY-NC jina-reranker-v2.
//
// Like every cross-encoder it reads the query and a passage TOGETHER ([CLS] query [SEP] passage [SEP]) and
// emits ONE relevance logit; attention flows across both texts, so it judges relevance far better than
// comparing two independent embeddings. We read the RAW logit straight off the model (the text-classification
// pipeline would squash it through a sigmoid), rank by it, and also compute a naive lexical-overlap score so
// the "see inside" surface can show the reranker rewarding MEANING across languages, not shared words.
//
// dtype note (honest, surfaced on the page): the repo's quantised ONNX files carry hardware-specific names
// (model_qint8_avx512.onnx, model_quint8_avx2.onnx, …) that transformers.js' standard dtype resolver does
// NOT map to, so the portable, verified path is fp32 (onnx/model.onnx). Bigger download, exact scores.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";

let tokenizer = null;
let model = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  const progress_callback = (p) => post({ type: "progress", p });
  // config.json already carries model_type:"xlm-roberta", so a plain load routes correctly — no injection
  // needed (unlike jina-reranker-v2). fp32 = the portable ONNX the standard dtype resolver can find.
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    dtype: "fp32",
    device: "wasm",
    progress_callback,
  });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  device = "wasm";
  post({ type: "ready", device });
}

// Stop-words in several of the model's languages, so lexical overlap is a fairer (if still naive)
// baseline on the multilingual demos — it is only ever the CONTRAST, never the ranking.
const STOP = new Set(
  ("the a an of to in on and or is are was were be been for with as at by from this that it its into " +
    "el la los las un una de a en y o que es son con por para del al lo se su " +
    "le les de la des un une et ou que est sont dans pour par sur au aux ce cette " +
    "der die das den und oder ist sind mit für von im auf zu ein eine")
    .split(" "),
);
function contentWords(s) {
  return new Set(
    s.toLowerCase().replace(/[^\p{L}\p{N}\s]/gu, " ").split(/\s+/).filter((w) =>
      w.length > 1 && !STOP.has(w)
    ),
  );
}
function lexicalOverlap(query, passage) {
  const q = contentWords(query);
  if (q.size === 0) return 0;
  const p = contentWords(passage);
  let hit = 0;
  for (const w of q) if (p.has(w)) hit++;
  return hit / q.size;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

async function rerank(id, query, passages) {
  await ensureLoaded();
  const t0 = performance.now();

  // Batch every (query, passage) pair through the cross-encoder in one pass.
  const enc = tokenizer(passages.map(() => query), {
    text_pair: passages,
    padding: true,
    truncation: true,
  });
  const out = await model(enc);
  const logitsT = out.logits;
  const nCols = logitsT.dims[logitsT.dims.length - 1];
  const flat = Array.from(logitsT.data, Number);

  const results = passages.map((passage, i) => {
    // num_labels === 1: one relevance logit per pair.
    const logit = flat[i * nCols];
    return {
      idx: i, // ORIGINAL (as-retrieved) position — used for the before/after view
      passage,
      logit,
      prob: sigmoid(logit),
      lexical: lexicalOverlap(query, passage),
    };
  });
  const ranked = [...results].sort((a, b) => b.logit - a.logit);

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, query, results, ranked, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await rerank(e.data.id, e.data.query, e.data.passages);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
