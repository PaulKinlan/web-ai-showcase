// NeoBERT embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: onnx-community/NeoBERT-ONNX (base encoder of chandar-lab/NeoBERT), task feature-extraction,
// WASM backend, q8 (model_quantized.onnx, ~223 MB).
//
// What makes NeoBERT distinct from the BERT / DistilBERT / ModernBERT / MiniLM encoders in this
// showcase: it is a from-scratch, next-generation BERT SUCCESSOR (Breton et al., 2025, arXiv:2502.19587).
// Instead of the 2018 BERT recipe it uses the modern transformer stack:
//   • Rotary position embeddings (RoPE) instead of learned absolute positions
//   • SwiGLU feed-forward and pre-norm blocks
//   • a depth-over-width shape (28 layers, 768 hidden ≈ 250M params)
//   • a 4,096-token context (8× classic BERT's 512), trained on RefinedWeb
// It uses the classic BERT-uncased WordPiece tokenizer (30,522 tokens), so it's a drop-in modern
// backbone. This ONNX export is the BASE encoder (no fine-tuned sentence-embedding head): we MEAN-POOL
// its per-token vectors (mask-aware) into one 768-d sentence vector, then L2-NORMALIZE so cosine
// similarity is a plain dot product. We pool with normalize:false so "See inside" can report the true
// pre-normalization magnitude, then normalize in JS ourselves.

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
    model: "onnx-community/NeoBERT-ONNX",
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

// Embed a batch of texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → mask-aware average of the per-token vectors. normalize:false so we can see the
  // real magnitude before we normalize it ourselves.
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
