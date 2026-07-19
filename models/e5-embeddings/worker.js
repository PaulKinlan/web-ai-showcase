// E5-small-v2 embeddings worker — all inference off the main thread so the UI stays responsive.
// Model: Xenova/e5-small-v2 (pipeline task: feature-extraction), WASM backend, q8, 384-d.
//
// What makes E5 distinct: it is trained with ASYMMETRIC INSTRUCTION PREFIXES. Every input must be
// prefixed — a search query with "query: " and a document/passage with "passage: " — because E5 was
// contrastively trained on (query, passage) pairs where the two sides play different roles. Skip the
// prefix, or use the wrong one, and retrieval quality drops. This is the opposite of GTE (no prefix)
// and different from BGE (a single query-only instruction). Like the others it MEAN-pools the per-token
// vectors and L2-normalizes; we pool with normalize:false so "See inside" can show the true pre-norm
// magnitude, then normalize in JS so cosine similarity is a plain dot product.

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
    model: "Xenova/e5-small-v2",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Apply the E5 instruction prefix. kind: "query" | "passage" | "raw" (raw = no prefix, to DEMONSTRATE
// the effect of skipping it in the See-inside panel — never the recommended path).
function applyPrefix(text, kind) {
  if (kind === "query") return "query: " + text;
  if (kind === "passage") return "passage: " + text;
  return text; // raw
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Embed a batch → mean-pooled, L2-normalized 384-d vectors (+ pre-norm magnitudes).
async function embed(id, texts, kind) {
  await ensureLoaded();
  const t0 = performance.now();
  const prefixed = texts.map((t) => applyPrefix(t, kind));

  // pooling:"mean" → mask-aware average of the per-token vectors (E5's trained representation).
  // normalize:false → we normalize ourselves so "See inside" can show the real magnitude.
  const out = await pipe(prefixed, { pooling: "mean", normalize: false });
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
  post({ type: "result", id, texts, kind, embeddings, norms, dim, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.texts, e.data.kind || "query");
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
