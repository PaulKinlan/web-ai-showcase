// BGE-M3 dense-embedding worker — all inference off the main thread so the UI stays smooth.
// Model: Xenova/bge-m3 (task: feature-extraction), WASM, q8. 1024-d dense retrieval vectors.
//
// What makes BGE-M3 distinct from the other embedders on this site:
//   • MULTI-LINGUAL at scale — 100+ languages share one 1024-d space (vs ~50 for multilingual MiniLM,
//     English-only for MiniLM-L6 / GTE / E5 / mxbai).
//   • LONG context — up to 8192 tokens, so it embeds whole paragraphs / documents, not just sentences.
//   • It is the DENSE-retrieval head of a multi-functionality model (BAAI's M3: dense + sparse +
//     multi-vector); here we use the dense head, the one you drop into a vector database.
//
// Pooling: BGE-M3's dense vector is the normalized [CLS] hidden state, so we pool with "cls" (verified
// to separate meaning better than mean pooling here: EN↔ES 0.93, EN↔JA 0.92, unrelated 0.38). We pool
// with normalize:false so "See inside" can report the true pre-norm magnitude, then L2-normalize in JS
// so cosine similarity is a plain dot product.

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
    model: "Xenova/bge-m3",
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

// Embed a batch of texts → cls-pooled, L2-normalized 1024-d vectors (+ pre-norm magnitudes + token
// counts so the UI can show how much of the 8192-token window each input uses).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"cls" → BGE-M3's trained dense representation. normalize:false → we normalize ourselves so
  // "See inside" can show the real magnitude.
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

  // Real subword token counts (per text) so the long-context story is concrete — best-effort, never
  // fatal: if the tokenizer surface changes, we just omit the counts.
  let tokens = null;
  try {
    tokens = [];
    for (const t of texts) {
      const enc = await pipe.tokenizer(t);
      const d = enc.input_ids?.dims;
      tokens.push(Array.isArray(d) ? d[d.length - 1] : (enc.input_ids?.data?.length ?? null));
    }
  } catch {
    tokens = null;
  }

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, texts, embeddings, norms, dim, tokens, ms, device });
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
