// nomic-embed-text-v1 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/nomic-embed-text-v1 (pipeline task: feature-extraction), WASM backend, q8. 768-d.
//
// Two things make Nomic distinct and MUST be handled correctly:
//   1. TASK PREFIXES. Nomic is trained with instruction prefixes baked into the text. For retrieval you
//      MUST prepend "search_query: " to queries and "search_document: " to the passages; for symmetric
//      jobs use "classification: " or "clustering: ". The prefix is part of the string the model sees —
//      get it wrong and retrieval quality drops. The PAGE decides the prefix; the worker just embeds.
//   2. MEAN POOLING. Nomic pools by averaging all token vectors (masked), NOT the [CLS] token. We pool
//      with pooling:"mean" and normalize:false so "See inside" can show the true pre-normalization
//      magnitude, then L2-normalize in JS so cosine similarity is a plain dot product.

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
    model: "Xenova/nomic-embed-text-v1",
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

// Embed already-prefixed texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

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
