// all-mpnet-base-v2 dense-embedding worker — all inference off the main thread so the UI stays smooth.
// Model: Xenova/all-mpnet-base-v2 (task: feature-extraction), WASM, q8. 768-d vectors.
//
// What makes all-mpnet-base-v2 distinct from the other embedders on this site:
//   • It is THE canonical sentence-transformers workhorse — ~250M monthly downloads of the base repo,
//     the literal `SentenceTransformer("all-mpnet-base-v2")` default in a decade of semantic-search
//     tutorials, and the baseline row on embedding benchmarks ever since.
//   • MPNet architecture (109M params): pre-trained with Masked AND Permuted language modelling
//     (MLM ∪ PLM) — it predicts tokens in a permuted order while still seeing full position
//     information, fixing BERT's masked-token independence assumption and XLNet's position blindness.
//   • SYMMETRIC embeddings — no "query: "/"passage: " prefixes (unlike arctic-embed / e5): the same
//     encoder embeds queries and documents identically, which is why it slots into so many pipelines.
//   • MEAN pooling over every token (not CLS) — the sentence vector is literally the average of the
//     token vectors, which the "see inside" surface visualises token by token.
//   • Fine-tuned with a contrastive objective on 1.17B sentence pairs; inputs truncate at 384 word
//     pieces. Apache-2.0.
//
// Pooling: mean, per the model card (mean_pooling over the attention mask, then L2-normalize). We pool
// with normalize:false so "See inside" can report the true pre-norm magnitude, then L2-normalize in JS
// so cosine = dot product. No prefixes anywhere — the space is symmetric.

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
    model: "Xenova/all-mpnet-base-v2",
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

// Embed a batch of texts → mean-pooled, L2-normalized 768-d vectors (+ pre-norm magnitudes + token
// counts so the 384-word-piece truncation window is concrete in the UI).
async function embed(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();

  // pooling:"mean" → the model card's trained recipe. normalize:false → we normalize ourselves so
  // "See inside" can show the real magnitude.
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

  // Real subword token counts (per text) so the 384-token truncation window is concrete — best-effort,
  // never fatal: if the tokenizer surface changes, we just omit the counts.
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

// Mean-pooling anatomy for ONE text: the real word-piece tokens plus, per token, the cosine between
// that token's contextual vector and the pooled sentence vector (how much it "agrees with" the final
// meaning) and the token vector's magnitude (how much weight it carries in the average).
async function inspect(id, text) {
  await ensureLoaded();
  const t0 = performance.now();

  const enc = await pipe.tokenizer(text);
  const ids = Array.from(enc.input_ids.data, Number);
  let tokens;
  try {
    tokens = pipe.tokenizer.model.convert_ids_to_tokens(ids);
  } catch {
    tokens = ids.map((x) => String(x));
  }

  const raw = await pipe(text); // no pooling → [1, T, 768] per-token contextual vectors
  const T = raw.dims[1];
  const D = raw.dims[2];
  const data = raw.data;

  // mean vector (this IS the sentence embedding before normalization — verified equal to the
  // pipeline's pooling:"mean" output)
  const mean = new Array(D).fill(0);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < D; d++) mean[d] += data[t * D + d] / T;
  }
  const meanNorm = l2norm(mean);

  const tokenCos = [];
  const tokenNorm = [];
  for (let t = 0; t < T; t++) {
    let dot = 0;
    let sq = 0;
    for (let d = 0; d < D; d++) {
      const v = data[t * D + d];
      dot += v * mean[d];
      sq += v * v;
    }
    const n = Math.sqrt(sq);
    tokenNorm.push(n);
    tokenCos.push(dot / ((n || 1) * (meanNorm || 1)));
  }

  const ms = Math.round(performance.now() - t0);
  post({ type: "inspect", id, text, tokens, tokenCos, tokenNorm, meanNorm, count: T, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.texts);
    else if (type === "inspect") await inspect(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
