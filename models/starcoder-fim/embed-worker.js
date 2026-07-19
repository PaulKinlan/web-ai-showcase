// Embedding worker for the StarCoder multi-model demo — off the main thread.
// Model: Xenova/all-MiniLM-L6-v2 (feature-extraction, ~23 MB, WASM). We embed the code StarCoder just
// generated AND a small labelled corpus, then the page ranks the corpus by cosine similarity to find
// the nearest example. A genuine two-model pipeline: StarCoder writes → MiniLM finds the neighbour.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";
let extractor = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (extractor) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  extractor = await pipeline("feature-extraction", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "wasm" });
}

// Mean-pooled, L2-normalised sentence embeddings — the standard sentence-transformers recipe.
async function embed(texts) {
  const out = await extractor(texts, { pooling: "mean", normalize: true });
  // out is a Tensor [n, dim]; return plain arrays so they postMessage cleanly.
  const dim = out.dims[out.dims.length - 1];
  const n = out.dims.length > 1 ? out.dims[0] : 1;
  const data = Array.from(out.data);
  const vecs = [];
  for (let i = 0; i < n; i++) vecs.push(data.slice(i * dim, i * dim + dim));
  return vecs;
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "embed") {
      await ensureLoaded();
      const vecs = await embed(e.data.texts);
      post({ type: "embedded", id, vecs, dim: vecs[0]?.length ?? 0 });
    }
  } catch (err) {
    console.error("[embed worker] error", err);
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
