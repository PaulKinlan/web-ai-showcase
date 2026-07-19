// Tiny embedding worker for the paraphrase multi-model page — measures SEMANTIC similarity (meaning),
// not just word overlap. Model: Xenova/all-MiniLM-L6-v2 (task: feature-extraction), WASM, q8. Mean-pooled,
// L2-normalized 384-d vectors so cosine similarity is a plain dot product. Off the main thread; nothing
// leaves the device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
function post(m) {
  self.postMessage(m);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  pipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "wasm" });
}

// Embed a batch of texts → mean-pooled, L2-normalized 384-d unit vectors.
async function embed(id, texts) {
  await ensureLoaded();
  const out = await pipe(texts, { pooling: "mean", normalize: false });
  const dim = out.dims[1];
  const flat = Array.from(out.data);
  const embeddings = [];
  for (let i = 0; i < texts.length; i++) {
    const raw = flat.slice(i * dim, (i + 1) * dim);
    const n = Math.sqrt(raw.reduce((s, v) => s + v * v, 0));
    embeddings.push(raw.map((v) => v / (n || 1)));
  }
  post({ type: "result", id, embeddings });
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
