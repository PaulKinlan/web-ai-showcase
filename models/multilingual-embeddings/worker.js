// Multilingual sentence-embeddings worker — all inference off the main thread so the UI stays smooth.
// Model: Xenova/paraphrase-multilingual-MiniLM-L12-v2 (task: feature-extraction), WASM, q8.
//
// What makes THIS embedder distinct from the English MiniLM/BGE/GTE/E5/mxbai/Jina pages: it is
// CROSS-LINGUAL. The model was distilled (via multilingual knowledge distillation, Reimers & Gurevych
// 2020) so that a sentence and its translation land at nearly the SAME point in a shared 384-d space —
// across 50+ languages. "The weather is nice" (EN), "Il fait beau" (FR) and "今日はいい天気です" (JA)
// get near-identical vectors, so cosine similarity measures MEANING, not language. Like MiniLM it uses
// mean pooling and needs no instruction prefix: you embed a query and a document — in any language —
// exactly the same way. We pool with normalize:false so "See inside" can report the true pre-norm
// magnitude, then L2-normalize in JS so cosine similarity is a plain dot product.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "feature-extraction",
    model: "Xenova/paraphrase-multilingual-MiniLM-L12-v2",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Embed a batch of texts → mean-pooled, L2-normalized 384-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → mask-aware average of the per-token vectors (this model's trained representation).
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
  const out = await pipe(texts, { pooling: "mean", normalize: false });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);

  const embeddings = [];
  const norms = [];
  for (let i = 0; i < texts.length; i++) {
    const raw = flat.slice(i * dim, (i + 1) * dim);
    const n = l2norm(raw);
    norms.push(n);
    embeddings.push(raw.map((v) => v / (n || 1))); // unit vectors → cosine = dot product
  }

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, texts, embeddings, norms, dim, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
