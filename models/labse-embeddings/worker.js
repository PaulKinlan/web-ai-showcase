// LaBSE sentence-embeddings worker — all inference off the main thread so the UI stays smooth.
// Model: Xenova/LaBSE (task: feature-extraction), WASM, q8. 768-dimensional vectors.
//
// What makes LaBSE distinct from the other cross-lingual embedders on this site (multilingual-MiniLM,
// bge-m3, paraphrase-multilingual): LaBSE is a BERT DUAL-ENCODER trained specifically for TRANSLATION
// RETRIEVAL. Google trained it with a translation-ranking (bitext) objective across 109 languages, so
// a sentence and its human translation are pulled to nearly the SAME 768-d point while unrelated
// sentences are pushed apart with a margin. That makes it unusually strong at BITEXT MINING — finding,
// among a pool of candidates in many languages, the one that is the translation of a given source.
//
// Unlike SentenceTransformers' default LaBSE (which L2-normalizes internally), the Xenova ONNX export
// emits the raw pooled [CLS]-style sentence vector; we pool with normalize:false so "See inside" can
// report the true pre-norm magnitude, then L2-normalize in JS so cosine similarity is a plain dot
// product. (LaBSE pools the encoder's first-token / pooler output, not mean-pooling.)

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
    model: "Xenova/LaBSE",
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

// Embed a batch of texts → L2-normalized 768-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" gives the sentence-level representation transformers.js exposes for the LaBSE ONNX
  // export; normalize:false so we can report the real pre-normalization magnitude in "See inside".
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
