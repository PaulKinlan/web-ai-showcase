// Qwen3-Embedding-0.6B embeddings worker — all inference off the main thread.
// Model: onnx-community/Qwen3-Embedding-0.6B-ONNX (task: feature-extraction), WASM backend, q8. ~585 MB.
//
// What makes Qwen3-Embedding distinct (2025, top of MTEB multilingual): it is an LLM-BACKBONE
// embedder — a Qwen3 0.6B decoder — so unlike a BERT mean-pooling encoder it uses:
//   1. LAST-TOKEN POOLING — the hidden state of the final (EOS) position is the sentence vector
//      (causal attention means only the last token has seen the whole input).
//   2. INSTRUCTION-AWARE QUERIES — a QUERY is wrapped as
//        "Instruct: {task}\nQuery:{query}"
//      while DOCUMENTS are embedded raw. The instruction steers the same model toward the retrieval
//      objective. The page adds the wrapper; this worker just embeds whatever text it is given.
//   3. 100+ LANGUAGES and MRL — the 1024-d vector degrades gracefully when truncated (Matryoshka),
//      so the page can trade dimensions for index size without re-embedding.
// We pool with normalize:false so "See inside" can show the true pre-normalization magnitude, then the
// page L2-normalizes (full or truncated) itself.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Qwen3-Embedding-0.6B-ONNX";
const DEVICE = "wasm";
const DTYPE = "q8";
let pipe = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline } = await import(TRANSFORMERS_URL);
  console.log(`[qwen3-emb worker] loading ${MODEL_ID} on ${DEVICE} (${DTYPE})`);
  pipe = await pipeline("feature-extraction", MODEL_ID, {
    device: DEVICE,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[qwen3-emb worker] ready on ${DEVICE}`);
  post({ type: "ready", device: DEVICE });
}

// Embed a batch of texts → last-token-pooled, UN-normalized vectors (+ pre-norm magnitudes).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  // pooling:"last_token" → the final (EOS) position's hidden state; normalize:false → keep magnitude.
  const out = await pipe(texts, { pooling: "last_token", normalize: false });
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
  post({ type: "result", id, texts, vectors, norms, dim, ms, device: DEVICE });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.texts);
  } catch (err) {
    console.error("[qwen3-emb worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
