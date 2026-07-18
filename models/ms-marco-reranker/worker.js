// MS MARCO MiniLM cross-encoder reranker worker — inference off the main thread.
// Model: Xenova/ms-marco-MiniLM-L-6-v2 (task: text-classification, num_labels=1), WASM, q8.
//
// A CROSS-ENCODER reads the query and a passage TOGETHER ([CLS] query [SEP] passage [SEP]) and emits a
// single relevance logit — attention flows across both texts, so it can judge relevance far better than
// comparing two independent embeddings. We read the RAW logit straight off the model (the pipeline would
// squash it through a sigmoid), rank passages by it, and also compute a naive lexical-overlap score for
// contrast so you can see the cross-encoder rewarding meaning, not just shared words.

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

const STOP = new Set(
  "the a an of to in on and or is are was were be been for with as at by from this that it its into their his her our your"
    .split(" "),
);
function contentWords(s) {
  return new Set(
    s.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter((w) =>
      w.length > 1 && !STOP.has(w)
    ),
  );
}
// Naive lexical relevance: fraction of the query's content words that appear in the passage.
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
    // num_labels is 1 for this reranker: one relevance logit per pair.
    const logit = flat[i * nCols];
    return {
      idx: i,
      passage,
      logit,
      prob: sigmoid(logit),
      lexical: lexicalOverlap(query, passage),
    };
  });
  results.sort((a, b) => b.logit - a.logit);

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, query, results, ms, device });
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
