// Jina-reranker-v2 multilingual cross-encoder worker — inference off the main thread.
// Model: jinaai/jina-reranker-v2-base-multilingual (XLMRobertaForSequenceClassification, num_labels=1),
// WASM, q8 (→ onnx/model_quantized.onnx, ~280 MB).
//
// This is Jina AI's SECOND-generation reranker: a multilingual cross-encoder covering 100+ languages,
// trained to also handle CODE and function-calling relevance — a big step up from the v1 tiny English-only
// reranker. Like every cross-encoder it reads the query and a passage TOGETHER
// ([CLS] query [SEP] passage [SEP]) and emits ONE relevance logit; attention flows across both texts, so
// it judges relevance far better than comparing two independent embeddings, and it does so ACROSS
// languages (an English query can rank a Chinese passage) and across modalities (natural-language query →
// code snippet). We read the RAW logit straight off the model, rank by it, and also compute a naive
// lexical-overlap score so the "see inside" surface can show the reranker rewarding meaning, not words.
//
// LOADING NOTE (honest detail, surfaced on the page): jina-reranker-v2 ships a custom XLMRobertaFlashConfig
// whose serialized ONNX config omits the `model_type` field, so a plain transformers.js pipeline() errors
// with "Unsupported model type: null". The underlying architecture IS the supported
// XLMRobertaForSequenceClassification, so we load AutoConfig, set model_type = "xlm-roberta", and load
// AutoModelForSequenceClassification with that config. No remote code is executed — only the official
// weights + tokenizer from the canonical repo.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "jinaai/jina-reranker-v2-base-multilingual";

let tokenizer = null;
let model = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, env } = await import(
    TRANSFORMERS_URL
  );
  env.allowLocalModels = false;
  const progress_callback = (p) => post({ type: "progress", p });
  // The ONNX config omits model_type; inject the real backbone type so transformers.js routes correctly.
  const config = await AutoConfig.from_pretrained(MODEL_ID, { progress_callback });
  config.model_type = "xlm-roberta";
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    config,
    progress_callback,
  });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback });
  device = "wasm";
  post({ type: "ready", device });
}

// Stop-words in several of the model's languages, so lexical overlap is a fairer (if still naive)
// baseline on the multilingual demos — it's only ever the CONTRAST, never the ranking.
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
    // num_labels === 1 for this reranker: one relevance logit per pair.
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
