// Granite Embedding multilingual (r2) worker — all inference off the main thread so the UI stays
// responsive. Model: onnx-community/granite-embedding-97m-multilingual-r2-ONNX
// (pipeline task: feature-extraction), WASM backend, q8 (model_quantized.onnx, ~98 MB).
//
// What makes THIS embedder distinct from the GTE / MiniLM / BGE / E5 / Arctic / Nomic pages: it is
// IBM's **Granite Embedding r2** family — trained by IBM for enterprise retrieval and released under
// Apache-2.0 — and it is genuinely **multilingual** (built and evaluated across a dozen+ languages),
// so an English query lands next to its French, Spanish, Chinese or Japanese equivalent in the same
// 384-dimensional space. The r2 generation sits on a **ModernBERT** backbone (rotary positions,
// local/global attention, an 8192-token context) rather than a classic BERT encoder. It pools by
// **mean** over the token vectors (not the [CLS] token). We ask for normalize:false so "See inside"
// can report the true pre-normalization magnitude, then L2-normalize in JS ourselves so cosine
// similarity is a plain dot product.

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
    model: "onnx-community/granite-embedding-97m-multilingual-r2-ONNX",
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

// Embed a batch of texts → mean-pooled, L2-normalized 384-d vectors (+ pre-norm magnitudes + token counts).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → the mean of the token vectors (Granite Embedding's trained sentence representation).
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
  const out = await pipe(texts, { pooling: "mean", normalize: false });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);

  // Approximate token counts (for the readout) via the pipeline's tokenizer.
  let tokenCounts = null;
  try {
    const enc = pipe.tokenizer(texts, { padding: false, truncation: false });
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
  post({ type: "result", id, texts, embeddings, norms, dim, tokenCounts, ms, device });
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
