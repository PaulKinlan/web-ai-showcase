// Nomic multimodal-embedding worker — all inference off the main thread. Loads TWO aligned models:
//   • nomic-ai/nomic-embed-vision-v1.5  → image embeddings (a ViT with a nomic_bert head)
//   • nomic-ai/nomic-embed-text-v1.5    → text embeddings (long-context nomic-BERT)
// Both project into the SAME embedding space, so a text query and an image can be compared directly by
// cosine similarity — CLIP-style retrieval, but from the Nomic family (distinct from OpenAI CLIP /
// SigLIP / DINOv2). We L2-normalise every vector in the worker so the page's cosine = a plain dot
// product. WASM backend, q8 (vision ~96 MB + text ~140 MB).
//
// Image embedding = the CLS token of the vision model's last_hidden_state (dim 768) — the pooled image
// representation. Text embedding = mean-pooling of the text model's last_hidden_state over the
// attention mask. Text QUERIES are prefixed with "search_query: " (the Nomic task-prefix convention),
// which sharpened the ranking in our verification. Everything here uses the real Transformers.js
// AutoModel/AutoProcessor/AutoTokenizer API — no invented surface.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let mod = null;
let vProc = null, vModel = null, tTok = null, tModel = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (vModel && tModel) return;
  mod = await import(TRANSFORMERS_URL);
  const { AutoModel, AutoProcessor, AutoTokenizer, env } = mod;
  env.allowLocalModels = false;
  const onProgress = (p) => post({ type: "progress", p });
  // Vision model + its image processor.
  vProc = await AutoProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", {
    progress_callback: onProgress,
  });
  vModel = await AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", {
    dtype: "q8",
    device: "wasm",
    progress_callback: onProgress,
  });
  // Text model + tokenizer (the v1.5 aligned text tower).
  tTok = await AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", {
    progress_callback: onProgress,
  });
  tModel = await AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", {
    dtype: "q8",
    device: "wasm",
    progress_callback: onProgress,
  });
  post({ type: "ready", device });
}

function l2normalize(vec) {
  let s = 0;
  for (let i = 0; i < vec.length; i++) s += vec[i] * vec[i];
  const n = Math.sqrt(s) || 1;
  for (let i = 0; i < vec.length; i++) vec[i] /= n;
  return vec;
}

async function embedImage(url) {
  const { RawImage } = mod;
  const image = await RawImage.read(url);
  const inputs = await vProc(image);
  const out = await vModel(inputs);
  // CLS token = the first token of last_hidden_state [1, seq, dim].
  const lhs = out.last_hidden_state ?? out.image_embeds ?? out.pooler_output;
  const dim = lhs.dims[lhs.dims.length - 1];
  const cls = Float32Array.from(lhs.data.slice(0, dim));
  return l2normalize(cls);
}

async function embedText(text, isQuery) {
  const prefixed = (isQuery ? "search_query: " : "search_document: ") + text;
  const inputs = tTok(prefixed, { padding: true, truncation: true });
  const out = await tModel(inputs);
  const lhs = out.last_hidden_state ?? out.token_embeddings;
  const [, seq, dim] = lhs.dims;
  const data = lhs.data;
  const mask = inputs.attention_mask.data;
  const v = new Float32Array(dim);
  let cnt = 0;
  for (let s = 0; s < seq; s++) {
    if (!mask[s]) continue;
    cnt++;
    for (let d = 0; d < dim; d++) v[d] += data[s * dim + d];
  }
  for (let d = 0; d < dim; d++) v[d] /= cnt || 1;
  return l2normalize(v);
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "embedImage") {
      await ensureLoaded();
      const t0 = performance.now();
      const emb = await embedImage(e.data.url);
      post({
        type: "imageEmbedding",
        id,
        key: e.data.key,
        embedding: emb.buffer,
        dim: emb.length,
        ms: Math.round(performance.now() - t0),
        device,
      }, [emb.buffer]);
    } else if (type === "embedText") {
      await ensureLoaded();
      const t0 = performance.now();
      const emb = await embedText(e.data.text, e.data.isQuery !== false);
      post({
        type: "textEmbedding",
        id,
        embedding: emb.buffer,
        dim: emb.length,
        ms: Math.round(performance.now() - t0),
        device,
      }, [emb.buffer]);
    }
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
