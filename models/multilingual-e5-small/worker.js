// multilingual-e5-small worker — inference off the main thread so the control UI stays responsive.
// Model: Xenova/multilingual-e5-small (task: feature-extraction), WASM, q8. The ONNX q8 export of
// intfloat/multilingual-e5-small (MIT) — a 384-dim multilingual text embedder (XLM-RoBERTa base,
// ~118M params) trained on ~1B multilingual text pairs covering 100 languages.
//
// E5 models are PREFIX-CONDITIONED: every input must start with "query: " (for search queries /
// questions) or "passage: " (for documents / candidate texts) — the model was trained that way and
// the card is explicit that un-prefixed text scores noticeably worse. The worker therefore exposes
// embed(items) where each item is {text, kind: "query"|"passage"} and applies the prefix itself, so
// no page can accidentally send a bare string.
//
// Embeddings are mean-pooled over the token axis and L2-normalised (the card's canonical usage), so
// cosine similarity is a plain dot product. Vectors (Float32Array → plain arrays) and the full
// cosine matrix are returned so pages can show the REAL numbers.

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
    model: "Xenova/multilingual-e5-small",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// items: [{ text, kind: "query" | "passage" }] → { vectors: number[][], sims: number[][] }
async function embedItems(items) {
  await ensureLoaded();
  const prefixed = items.map((it) => `${it.kind === "passage" ? "passage" : "query"}: ${it.text}`);
  const out = await pipe(prefixed, { pooling: "mean", normalize: true });
  const vectors = [];
  for (let i = 0; i < out.dims[0]; i++) vectors.push(Array.from(out[i].data));
  const sims = vectors.map((v) => vectors.map((w) => dot(v, w)));
  return { vectors, sims };
}

async function embed(id, items) {
  const t0 = performance.now();
  const r = await embedItems(items);
  post({ type: "embed", id, ...r, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "embed") await embed(e.data.id, e.data.items);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
