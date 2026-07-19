// GTE-ModernBERT embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Alibaba-NLP/gte-modernbert-base (pipeline task: feature-extraction), WASM backend, q8.
//
// What makes THIS embedder distinct from the GTE-small / MiniLM / BGE / E5 pages: its backbone is
// **ModernBERT** (Warner et al., 2024) — a from-scratch redesign of the BERT encoder with rotary
// position embeddings, alternating local/global attention, GeGLU, and unpadding — trained natively on an
// **8192-token context** (vs 512 for classic BERT encoders). GTE fine-tuned it with contrastive learning
// into a strong 768-dimensional retrieval embedder that tops MTEB for its class while accepting long
// documents whole. Crucially it uses **CLS pooling** (the [CLS] token's vector), NOT the mean pooling
// GTE-small uses — so we pool with pooling:"cls". We ask for normalize:false so "See inside" can report
// the true pre-normalization magnitude, then L2-normalize in JS ourselves so cosine similarity is a
// plain dot product.

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
    model: "Alibaba-NLP/gte-modernbert-base",
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

// Embed a batch of texts → CLS-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes + token counts).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"cls" → the [CLS] token vector (gte-modernbert's trained sentence representation).
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
  const out = await pipe(texts, { pooling: "cls", normalize: false });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);

  // Approximate token counts (for the long-context readout) via the pipeline's tokenizer.
  let tokenCounts = null;
  try {
    const enc = pipe.tokenizer(texts, { padding: false, truncation: false });
    // enc.input_ids is a Tensor [n, seqLen] when padded, or list; derive per-text length robustly.
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
