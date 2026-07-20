// Cross-encoder rerank worker for the all-mpnet-base-v2 multi-model demo — inference off the main
// thread. Model: Xenova/ms-marco-MiniLM-L-6-v2 (task: text-classification, num_labels=1), WASM, q8.
//
// This is stage TWO of the classic two-stage search pipeline. Stage one (the MPNet bi-encoder) embeds
// query and passages independently — fast, cacheable, but it can only compare vectors it computed
// blindly. A CROSS-encoder reads the query and one passage TOGETHER ([CLS] query [SEP] passage [SEP])
// so attention flows across both texts, and emits a single relevance logit — slower per pair, far more
// precise. Retrieve wide with the bi-encoder, rerank the shortlist with this.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2";

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

  // num_labels is 1 for this reranker: one raw relevance logit per pair.
  const scores = passages.map((_, i) => flat[i * nCols]);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, scores, ms, device });
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
