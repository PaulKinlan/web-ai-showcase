// Jina reranker (tiny) cross-encoder worker — inference off the main thread.
// Model: jinaai/jina-reranker-v1-tiny-en (JinaBERT ForSequenceClassification, num_labels=1), WASM, q8.
//
// This is a TINY cross-encoder: 4 Transformer layers, ~33 MB q8. Like every cross-encoder it reads the
// query and a passage TOGETHER ([CLS] query [SEP] passage [SEP]) and emits a single relevance logit —
// attention flows across both texts, so it judges relevance far better than comparing two independent
// embeddings. Its whole pitch is speed: it's small enough to rerank a shortlist in milliseconds on a CPU
// via WASM, which is exactly what you want for on-device RAG. We read the RAW logit straight off the
// model, rank passages by it, count the tokens actually processed, and report real per-pair timing so
// the "see inside" surface can tell the tiny-model speed story honestly (measured, never claimed).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "jinaai/jina-reranker-v1-tiny-en";

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

// Naive lexical overlap — only ever the CONTRAST baseline, never the ranking.
const STOP = new Set(
  ("the a an of to in on and or is are was were be been for with as at by from this that it its into " +
    "how what when where why who which do does can i you your my our their")
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
  // Count the tokens actually processed (rows × padded columns) for the throughput readout.
  const idDims = enc.input_ids?.dims ?? [passages.length, 0];
  const totalTokens = idDims.reduce((a, b) => a * b, 1);

  const out = await model(enc);
  const logitsT = out.logits;
  const nCols = logitsT.dims[logitsT.dims.length - 1];
  const flat = Array.from(logitsT.data, Number);

  const results = passages.map((passage, i) => {
    const logit = flat[i * nCols]; // num_labels === 1: one relevance logit per pair
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
  const perPair = passages.length ? +(ms / passages.length).toFixed(1) : 0;
  post({
    type: "result",
    id,
    query,
    results,
    ranked,
    ms,
    perPair,
    totalTokens,
    pairs: passages.length,
    device,
  });
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
