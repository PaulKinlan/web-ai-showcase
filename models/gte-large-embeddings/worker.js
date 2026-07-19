// GTE-large embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/gte-large (pipeline task: feature-extraction), WASM, q8.
//
// GTE-large is the BIG sibling of GTE-small: the same recipe (multi-stage contrastive training, mean
// pooling, NO instruction prefix) but a BERT-large backbone (~335M params) that emits a 1024-dimensional
// sentence vector instead of 384. More parameters + more dimensions buy a higher MTEB retrieval score and
// sharper separation on hard, near-synonymous pairs — at the cost of a much larger download (~337 MB q8)
// and slower per-embedding latency. We pool with normalize:false so we can report the true pre-
// normalization magnitude in "See inside", then L2-normalize in JS ourselves so cosine similarity is a
// plain dot product.

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
    model: "Xenova/gte-large",
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

// Embed a batch of texts → mean-pooled, L2-normalized 1024-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → mask-aware average of the per-token vectors (GTE's trained representation).
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
