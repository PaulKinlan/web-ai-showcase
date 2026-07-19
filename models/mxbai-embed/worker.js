// mxbai-embed-large-v1 embeddings worker — all inference off the main thread.
// Model: mixedbread-ai/mxbai-embed-large-v1 (task: feature-extraction), WASM backend, q8. ~337 MB.
//
// What makes mxbai distinct: it's a LARGE, high-quality retrieval embedder — 335M params, BERT-large
// backbone → a 1024-DIMENSIONAL vector, trained by Mixedbread with the AnglE loss for strong MTEB
// retrieval. Two conventions matter and the page drives both:
//   1. RETRIEVAL PROMPT — prepend "Represent this sentence for searching relevant passages: " to the
//      QUERY (not the documents). The page prepends it before sending, so this worker just embeds text.
//   2. MATRYOSHKA — the vector is trained so a leading slice (1024→512→256…) is still a good embedding.
//      We return the full un-normalized 1024-d vector; the page truncates + re-normalizes as needed.
// We pool with normalize:false so "See inside" can show the true pre-normalization magnitude, then the
// page L2-normalizes (full or truncated) itself. Uses the SHARED loader from lib/webai.js.

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
    model: "mixedbread-ai/mxbai-embed-large-v1",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Embed a batch of texts → mean-pooled, UN-normalized 1024-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  // pooling:"mean" → mask-aware average of the per-token vectors; normalize:false → raw magnitude kept.
  const out = await pipe(texts, { pooling: "mean", normalize: false });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);
  const vectors = [];
  const norms = [];
  for (let i = 0; i < texts.length; i++) {
    const raw = flat.slice(i * dim, (i + 1) * dim);
    let s = 0;
    for (const v of raw) s += v * v;
    norms.push(Math.sqrt(s));
    vectors.push(raw);
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, texts, vectors, norms, dim, ms, device });
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
