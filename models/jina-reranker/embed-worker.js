// Retrieval worker for the Jina reranker multi-model demo — inference off the main thread.
// Model: Xenova/all-MiniLM-L6-v2 (task: feature-extraction), WASM, q8. This is STAGE 1: a fast
// bi-encoder that embeds the query and every passage independently and ranks by cosine similarity —
// cheap and good enough to build a shortlist. STAGE 2 (the tiny Jina cross-encoder) then reranks that
// shortlist for precision. Vectors are L2-normalized so cosine similarity is a plain dot product.

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
    model: "Xenova/all-MiniLM-L6-v2",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(texts, { pooling: "mean", normalize: true });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);
  const embeddings = [];
  for (let i = 0; i < texts.length; i++) embeddings.push(flat.slice(i * dim, (i + 1) * dim));
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, texts, embeddings, dim, ms, device });
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
