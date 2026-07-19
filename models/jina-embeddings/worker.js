// Jina v2 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/jina-embeddings-v2-base-en (task: feature-extraction), WASM backend, q8.
//
// What makes Jina v2 distinct: it is a LONG-CONTEXT embedder. Where a typical sentence encoder caps at
// 512 tokens, Jina uses ALiBi (Attention with Linear Biases) instead of learned positional embeddings, so
// one 768-d vector can faithfully represent up to 8192 tokens — a whole document, not just its opening.
// Like MiniLM/GTE it uses MEAN pooling and needs no instruction prefix. We pool with normalize:false so we
// can report the true pre-normalization magnitude in "See inside", then L2-normalize in JS ourselves so
// cosine similarity is a plain dot product. We also return each text's real token count so the pages can
// show when a 512-token model would have had to truncate.

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
    model: "Xenova/jina-embeddings-v2-base-en",
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

async function tokenCount(text) {
  try {
    const enc = await pipe.tokenizer(text, { truncation: false, add_special_tokens: true });
    return enc.input_ids.dims.at(-1);
  } catch {
    return null;
  }
}

// Embed a batch of texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes + token counts).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  const out = await pipe(texts, { pooling: "mean", normalize: false });
  const dim = out.dims[out.dims.length - 1];
  const flat = Array.from(out.data);

  const embeddings = [];
  const norms = [];
  for (let i = 0; i < texts.length; i++) {
    const raw = flat.slice(i * dim, (i + 1) * dim);
    const n = l2norm(raw);
    norms.push(n);
    embeddings.push(raw.map((v) => v / (n || 1)));
  }
  const tokenCounts = [];
  for (const t of texts) tokenCounts.push(await tokenCount(t));

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, texts, embeddings, norms, tokenCounts, dim, ms, device });
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
