// mxbai-rerank-xsmall-v1 cross-encoder worker — inference off the main thread.
// Model: mixedbread-ai/mxbai-rerank-xsmall-v1 (DebertaV2ForSequenceClassification, num_labels=1), WASM, q8.
//
// This is mixedbread's own reranker family. Like every cross-encoder it reads the query and a passage
// TOGETHER ([CLS] query [SEP] passage [SEP]) and emits a single relevance logit — attention flows
// across both texts, so it judges relevance far better than comparing two independent embeddings. What
// makes THIS reranker distinct from the BGE (XLM-RoBERTa) / MS-MARCO (BERT-MiniLM) / Jina (JinaBERT)
// rerankers already in the showcase is its backbone: DeBERTa-v2 with disentangled attention, tuned by
// mixedbread on English retrieval. We read the RAW logit straight off the model, rank passages by it,
// compute the decision margin between adjacent ranks, and also a naive lexical-overlap score so the
// "see inside" surface can show the reranker rewarding meaning, not shared words.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "mixedbread-ai/mxbai-rerank-xsmall-v1";

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

// Content-word overlap as a naive lexical baseline — only ever the CONTRAST, never the ranking.
const STOP = new Set(
  ("the a an of to in on and or is are was were be been for with as at by from this that it its into " +
    "how can i my while what which who when where why do does did will would should could than then")
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
  const margin = ranked.length > 1 ? ranked[0].logit - ranked[1].logit : null;

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, query, results, ranked, margin, ms, device });
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
