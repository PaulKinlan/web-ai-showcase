// all-distilroberta-v1 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: sentence-transformers/all-distilroberta-v1 (pipeline task: feature-extraction), WASM backend, fp32.
//
// The canonical model card specifies attention-mask-aware mean pooling followed by L2 normalization,
// with no instruction prefix. The DistilRoBERTa encoder was contrastively fine-tuned over more than one
// billion paired sentences and emits one 768-dimensional sentence vector. We request normalize:false so
// “See inside” can report the real pre-normalization magnitude, then L2-normalize in this worker so cosine
// similarity becomes a plain dot product.

// Staged pin (web-ai-showcase-9v4 / reports/transformers-version-policy.md):
// @huggingface/transformers@4.3.0 pinned locally for Phase 1 verification.
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  device = "wasm";
  pipe = await pipeline("feature-extraction", "sentence-transformers/all-distilroberta-v1", {
    device,
    dtype: "fp32",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Embed a batch of texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → mask-aware average of the per-token vectors (DistilRoBERTa v1's trained representation).
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
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
