// arctic-embed-m-v2.0 dense-embedding worker — all inference off the main thread so the UI stays smooth.
// Model: Snowflake/snowflake-arctic-embed-m-v2.0 (task: feature-extraction), WASM, q8. 768-d vectors.
//
// What makes arctic-embed-m-v2.0 distinct from the other embedders on this site:
//   • It is Snowflake's SECOND-generation ("v2.0") multilingual embedder, built on the GTE-multilingual
//     backbone (model_type "gte", GteModel) — a different architecture from the built arctic-embed (which
//     is arctic-embed-s v1: an English BERT, 384-d). This is 305M params, ~50 languages, 768-d.
//   • LONG context — up to 8192 tokens (RoPE), so it embeds whole paragraphs / documents, not just sentences.
//   • MATRYOSHKA (MRL) — the 768-d vector is trained so a truncated prefix (e.g. first 256 dims) is still a
//     usable embedding, letting you trade a little accuracy for much smaller vectors. We expose this in the
//     "see inside" / wild demos.
//   • Apache-2.0 — commercially usable.
//
// Pooling: arctic-embed uses CLS pooling (verified to separate meaning best here: EN-paraphrase 0.78,
// EN↔ES 0.74, unrelated 0.23 with cls; mean pools slightly worse). We pool with normalize:false so
// "See inside" can report the true pre-norm magnitude, then L2-normalize in JS so cosine = dot product.
// For retrieval the model expects the query prefixed with "query: " (documents get no prefix) — the pages
// add that prefix to the QUERY string before calling embed(); the symmetric similarity matrix embeds raw.

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
    model: "Snowflake/snowflake-arctic-embed-m-v2.0",
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

// Embed a batch of texts → cls-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes + token counts
// so the UI can show how much of the 8192-token window each input uses).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"cls" → arctic-embed's trained dense representation. normalize:false → we normalize ourselves
  // so "See inside" can show the real magnitude.
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
