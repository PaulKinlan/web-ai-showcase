// EmbeddingGemma-300m embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: onnx-community/embeddinggemma-300m-ONNX (canonical weights google/embeddinggemma-300m),
// pipeline task feature-extraction, WASM backend, q8.
//
// What makes THIS embedder distinct from the built MiniLM / GTE / BGE / E5 / Nomic / Arctic pages:
//   1. It is the FIRST embedding model built on the **Gemma 3** backbone (Google, 2025) — a 300M-param
//      decoder-derived encoder, not a BERT encoder — trained for retrieval over 100+ languages.
//   2. It is **instruction-prompted**: EmbeddingGemma expects a task prefix. Retrieval queries are
//      wrapped as `task: search result | query: {text}` and documents as `title: none | text: {text}`.
//      Getting these prefixes right is part of using the model honestly, so the worker owns them.
//   3. It is a **Matryoshka** model: the 768-d vector is trained so that its first 512 / 256 / 128 dims
//      are each a usable embedding on their own. We can truncate + re-normalize to trade a little
//      accuracy for far smaller vectors — the key on-device win (4× less storage at 128-d). The worker
//      returns the full 768-d unit vector; the page truncates for the Matryoshka demo.
// Pooling is **mean** over the token embeddings (the trained sentence representation). We ask for
// normalize:false so "See inside" can report the true pre-normalization magnitude, then L2-normalize
// in JS ourselves so cosine similarity is a plain dot product.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

// EmbeddingGemma's documented retrieval prompt prefixes (from the model card).
const PROMPTS = {
  query: (t) => `task: search result | query: ${t}`,
  document: (t) => `title: none | text: ${t}`,
  none: (t) => t,
};

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "feature-extraction",
    model: "onnx-community/embeddinggemma-300m-ONNX",
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

// Embed a batch of texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes + token
// counts). `mode` selects the EmbeddingGemma prompt prefix (query | document | none).
async function embed(id, texts, mode = "document") {
  await ensureLoaded();
  const t0 = performance.now();

  const wrap = PROMPTS[mode] || PROMPTS.none;
  const prompted = texts.map((t) => wrap(t));

  // pooling:"mean" → the trained sentence representation. normalize:false → we normalize ourselves so
  // "See inside" can show the real magnitude.
  const out = await pipe(prompted, { pooling: "mean", normalize: false });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);

  // Approximate token counts (per input) via the pipeline's tokenizer, for the readout.
  let tokenCounts = null;
  try {
    const enc = pipe.tokenizer(prompted, { padding: false, truncation: false });
    if (Array.isArray(enc.input_ids)) {
      tokenCounts = enc.input_ids.map((r) => (Array.isArray(r) ? r.length : r.dims?.[0] ?? null));
    } else if (enc.input_ids?.tolist) {
      tokenCounts = enc.input_ids.tolist().map((r) => r.length);
    }
  } catch { /* token counts are a nicety, not required */ }

  const embeddings = [];
  const norms = [];
  for (let i = 0; i < texts.length; i++) {
    const raw = flat.slice(i * dim, (i + 1) * dim);
    const n = l2norm(raw);
    norms.push(n);
    embeddings.push(raw.map((v) => v / (n || 1))); // unit vectors → cosine = dot product
  }

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, texts, embeddings, norms, dim, tokenCounts, mode, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.texts, e.data.mode);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
