// Embedding worker for the RAG demo — all-MiniLM-L6-v2 via the shared Transformers.js helper in
// lib/webai.js (feature-extraction, mean-pooled + normalized so a dot product is cosine similarity).
// This is the "retrieve" half of the pipeline; the "answer" half is StableLM 2 Zephyr in ../worker.js.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensure() {
  if (pipe) return;
  const r = await loadPipeline({
    task: "feature-extraction",
    model: "Xenova/all-MiniLM-L6-v2",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = r.pipe;
  post({ type: "ready" });
}

async function embed(id, texts) {
  await ensure();
  const out = await pipe(texts, { pooling: "mean", normalize: true });
  post({ type: "embeds", id, vectors: out.tolist() });
}

self.addEventListener("message", async (e) => {
  try {
    if (e.data.type === "load") await ensure();
    else if (e.data.type === "embed") await embed(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
