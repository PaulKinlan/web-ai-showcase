// Snowflake Arctic-Embed-S worker — all inference off the main thread so the UI stays responsive.
// Model: Snowflake/snowflake-arctic-embed-s (pipeline task: feature-extraction), WASM backend, q8.
//
// What makes Arctic-Embed distinct from the other embedders in this showcase: it is a
// RETRIEVAL-optimized family (Snowflake) with an asymmetric recipe — the QUERY is prefixed with a
// short instruction, the DOCUMENTS are embedded raw, and pooling is on the **[CLS]** token (not mean).
// The query instruction is:
//   "Represent this sentence for searching relevant passages: <query>"
// That single trick pulls a question and the passage that answers it closer together in vector space,
// which is exactly what a first-stage retriever needs. We pool with normalize:false so "See inside" can
// show the true pre-normalization magnitude, then L2-normalize in JS so cosine similarity is a plain
// dot product. arctic-embed-s emits 384-dimensional vectors.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

export const QUERY_PREFIX = "Represent this sentence for searching relevant passages: ";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "feature-extraction",
    model: "Snowflake/snowflake-arctic-embed-s",
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

// Embed a batch of texts → CLS-pooled, L2-normalized 384-d vectors (+ pre-norm magnitudes).
// When `query` is true, every text is wrapped in Arctic's retrieval instruction prefix first.
async function embed(id, texts, query) {
  await ensureLoaded();
  const inputs = query ? texts.map((t) => QUERY_PREFIX + t) : texts;
  const t0 = performance.now();

  // pooling:"cls" → Arctic reads the [CLS] token's vector as the sentence representation.
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
  const out = await pipe(inputs, { pooling: "cls", normalize: false });
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
  post({ type: "result", id, texts, embeddings, norms, dim, ms, device, query: !!query });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.texts, e.data.query);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
