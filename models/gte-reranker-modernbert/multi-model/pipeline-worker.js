// Two-model retrieve-then-rerank worker — both models run off the main thread, both ModernBERT.
//   Stage 1 (retriever): Alibaba-NLP/gte-modernbert-base (feature-extraction) — a bi-encoder embedder.
//   Stage 2 (reranker):  Alibaba-NLP/gte-reranker-modernbert-base (cross-encoder) — precise.
// The classic production pattern, kept on ONE backbone end to end: the GTE-ModernBERT bi-encoder embeds
// every passage and the query cheaply and ranks by cosine similarity; the GTE-reranker-ModernBERT
// cross-encoder then reads each shortlisted (query, passage) pair TOGETHER and reorders for precision.
// We report BOTH orders so the value of stage 2 is visible. Both share the ModernBERT backbone (8192-token
// context, rotary attention) — one architecture, two heads: a bi-encoder for recall, a cross-encoder for
// precision.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const EMBED_ID = "Alibaba-NLP/gte-modernbert-base";
const RERANK_ID = "Alibaba-NLP/gte-reranker-modernbert-base";

let embedPipe = null;
let rerankPipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (embedPipe && rerankPipe) return;
  if (!embedPipe) {
    post({ type: "stage", stage: "Loading GTE-ModernBERT retriever…" });
    const e = await loadPipeline({
      task: "feature-extraction",
      model: EMBED_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    embedPipe = e.pipe;
    device = e.device;
  }
  if (!rerankPipe) {
    post({ type: "stage", stage: "Loading GTE-reranker-ModernBERT…" });
    const r = await loadPipeline({
      task: "text-classification",
      model: RERANK_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    rerankPipe = r.pipe;
  }
  post({ type: "ready", device });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function retrieveThenRerank(id, query, passages, topK) {
  await ensureLoaded();
  const t0 = performance.now();

  // --- Stage 1: embed query + passages (GTE-ModernBERT uses CLS pooling), rank by cosine ---
  const texts = [query, ...passages];
  const emb = await embedPipe(texts, { pooling: "cls", normalize: true });
  const dim = emb.dims[emb.dims.length - 1];
  const rows = [];
  for (let i = 0; i < texts.length; i++) {
    rows.push(Array.from(emb.data.slice(i * dim, (i + 1) * dim), Number));
  }
  const qVec = rows[0];
  const scored = passages.map((passage, i) => ({
    idx: i,
    passage,
    cosine: cosine(qVec, rows[i + 1]),
  }));
  const retrieved = [...scored].sort((a, b) => b.cosine - a.cosine);
  const shortlist = retrieved.slice(0, Math.min(topK, retrieved.length));
  const t1 = performance.now();

  // --- Stage 2: cross-encoder rerank the shortlist (the reranker) ---
  const tok = rerankPipe.tokenizer;
  const model = rerankPipe.model;
  const enc = tok(shortlist.map(() => query), {
    text_pair: shortlist.map((s) => s.passage),
    padding: true,
    truncation: true,
  });
  const out = await model(enc);
  const nCols = out.logits.dims[out.logits.dims.length - 1];
  const flat = Array.from(out.logits.data, Number);
  const reranked = shortlist.map((s, i) => ({
    ...s,
    logit: flat[i * nCols],
  })).sort((a, b) => b.logit - a.logit);
  const t2 = performance.now();

  post({
    type: "result",
    id,
    query,
    retrieved: shortlist, // shown in embedding (retrieval) order
    reranked, // shown after cross-encoder
    retrieveMs: Math.round(t1 - t0),
    rerankMs: Math.round(t2 - t1),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") {
      await retrieveThenRerank(e.data.id, e.data.query, e.data.passages, e.data.topK ?? 4);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
