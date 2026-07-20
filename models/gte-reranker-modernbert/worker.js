// GTE-reranker-ModernBERT cross-encoder worker — inference off the main thread.
// Model: Alibaba-NLP/gte-reranker-modernbert-base (ModernBertForSequenceClassification, num_labels=1),
// WASM, q8 (→ onnx/model_quantized.onnx, ~151 MB).
//
// This is Alibaba's GTE reranker built on the **ModernBERT** backbone — a 2025 encoder redesign
// (rotary embeddings, GeGLU, alternating local/global attention) with an **8192-token context**, so it
// reranks LONG passages a base-BERT reranker would have to truncate. Like every cross-encoder it reads
// the query and a passage TOGETHER ([CLS] query [SEP] passage [SEP]) and emits ONE relevance logit —
// attention flows across both texts, so it judges relevance far better than comparing two independent
// embeddings. We read the RAW logit straight off the model, rank passages by it, and also compute a
// naive lexical-overlap score so the "see inside" surface can show the reranker rewarding meaning,
// not shared words. The ModernBertForSequenceClassification class ships in the shared Transformers.js
// 3.7.5 pin — no version-pin escape hatch needed.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Alibaba-NLP/gte-reranker-modernbert-base";

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

// English stop-words so lexical overlap is a fairer (if still naive) baseline — it's only ever the
// CONTRAST column, never the ranking.
const STOP = new Set(
  ("the a an of to in on and or is are was were be been being for with as at by from this that it its " +
    "into over under about after before between out up down off no not do does did have has had i you " +
    "he she we they them his her their my your our what which who how when where why can could should " +
    "would will just than then so if but because while during against within without")
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
  const tok = pipe.tokenizer;
  const model = pipe.model;

  // Batch every (query, passage) pair through the cross-encoder in one pass.
  const enc = tok(passages.map(() => query), {
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
