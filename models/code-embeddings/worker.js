// Code-embeddings worker — all inference off the main thread so the control UI stays responsive.
// Model: jinaai/jina-embeddings-v2-base-code (pipeline task: feature-extraction), WASM backend, q8.
//
// What makes this model distinct from the TEXT embedders in the showcase (MiniLM / BGE / GTE): it was
// contrastively trained on CODE — 150M+ (natural-language, code) and (code, code) pairs across 30
// programming languages — so its "meaning space" aligns an English description with the function that
// implements it, and aligns two functions that do the same thing in DIFFERENT languages. That is what
// powers natural-language code search, duplicate/clone detection, and cross-language clone finding.
//
// Architecture: JinaBERT (a BERT encoder with ALiBi positional bias, so it handles up to 8192 tokens)
// → a 768-dimensional sentence embedding. We pool with normalize:false so "See inside" can report the
// true pre-normalization magnitude, then L2-normalize in JS so cosine similarity is a plain dot product.

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
    model: "jinaai/jina-embeddings-v2-base-code",
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

// Embed a batch of code/NL strings → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → mask-aware average of the per-token vectors (Jina's trained sentence representation).
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
