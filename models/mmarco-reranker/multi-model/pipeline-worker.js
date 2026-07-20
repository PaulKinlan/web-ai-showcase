// Two-model retrieve-then-rerank worker — both models run off the main thread, both multilingual.
//   Stage 1 (retriever): Xenova/multilingual-e5-small (feature-extraction) — multilingual bi-encoder.
//   Stage 2 (reranker):  cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (cross-encoder) — precise, multilingual.
// The classic production pattern, kept multilingual end to end: E5 embeddings cast a wide net across
// languages cheaply; the mMARCO cross-encoder reads each shortlisted (query, passage) pair together and
// reorders for precision. We report BOTH orders so the value of stage 2 is visible — and because both
// stages are multilingual, an English query can surface (and correctly rerank) a passage in another language.
//
// The reranker's config already carries model_type:"xlm-roberta", so a plain load routes correctly.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const EMBED_ID = "Xenova/multilingual-e5-small";
const RERANK_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1";

let embedPipe = null;
let rerankTokenizer = null;
let rerankModel = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (embedPipe && rerankModel) return;
  if (!embedPipe) {
    post({ type: "stage", stage: "Loading multilingual E5 retriever…" });
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
  if (!rerankModel) {
    post({ type: "stage", stage: "Loading mMARCO mMiniLMv2 reranker…" });
    const { AutoTokenizer, AutoModelForSequenceClassification, env } = await import(TRANSFORMERS_URL);
    env.allowLocalModels = false;
    const progress_callback = (p) => post({ type: "progress", p });
    rerankModel = await AutoModelForSequenceClassification.from_pretrained(RERANK_ID, {
      dtype: "fp32",
      device: "wasm",
      progress_callback,
    });
    rerankTokenizer = await AutoTokenizer.from_pretrained(RERANK_ID, { progress_callback });
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

  // --- Stage 1: embed query + passages (E5 wants "query:" / "passage:" prefixes), rank by cosine ---
  const texts = [`query: ${query}`, ...passages.map((p) => `passage: ${p}`)];
  const emb = await embedPipe(texts, { pooling: "mean", normalize: true });
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
  const enc = rerankTokenizer(shortlist.map(() => query), {
    text_pair: shortlist.map((s) => s.passage),
    padding: true,
    truncation: true,
  });
  const out = await rerankModel(enc);
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
