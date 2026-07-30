// all-mpnet-base-v2 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/all-mpnet-base-v2 (ONNX export of sentence-transformers/all-mpnet-base-v2), pipeline task
// feature-extraction, WASM backend, int8-quantized (q8). Pinned to an immutable revision.
//
// The canonical model card specifies attention-mask-aware MEAN pooling followed by L2 normalization,
// with no instruction prefix — query and document are embedded identically (symmetric). MPNet emits one
// 768-dimensional sentence vector. We request normalize:false so "See inside" can report the real
// pre-normalization magnitude, then L2-normalize here so cosine similarity becomes a plain dot product.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/all-mpnet-base-v2";
// Immutable pinned revision — verified in-browser. Pinning means there is no mutable "latest" to drift to.
const REVISION = "e086c5e0b3a57b0ce46dd6d9c0662948860b35f3";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "feature-extraction",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "q8",
    revision: REVISION,
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

// Embed a batch of texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → mask-aware average of the per-token vectors (MPNet's trained representation).
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
