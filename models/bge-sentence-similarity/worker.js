// bge-small-en-v1.5 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/bge-small-en-v1.5 (pipeline task: feature-extraction), WASM backend, q8.
//
// The discipline that matters for BGE: unlike MiniLM (which mean-pools), the BGE v1.5 family is
// trained with **CLS pooling** — the sentence vector is the transformer's first ([CLS]) token, not an
// average of every token. We pool with `pooling:"cls"` and `normalize:false` so we can report the true
// pre-normalization magnitude in "See inside", then L2-normalize in JS so cosine similarity is a plain
// dot product. The page (not the worker) decides whether to prepend BGE's retrieval query instruction.

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
    model: "Xenova/bge-small-en-v1.5",
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
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"cls" → the [CLS] token vector (BGE's trained sentence representation).
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
  const out = await pipe(texts, { pooling: "cls", normalize: false });
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
