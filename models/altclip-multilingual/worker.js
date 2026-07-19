// Multilingual zero-shot image worker — runs ALL inference off the main thread (invariant 3).
//
// Model: jinaai/jina-clip-v2 (JinaCLIPModel), WASM backend, q8 (~874 MB, cached after first load).
// jina-clip-v2 is a genuinely multilingual CLIP (89 languages). Unlike OpenAI CLIP it returns
// L2-normalised image + text embeddings rather than logits_per_image, so we compute the cosine
// similarity here. Same concept in different languages lands at a similar cosine — the whole point.
//
// We reuse the processor's image_processor (vision) and tokenizer (XLM-RoBERTa, text) directly.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "jinaai/jina-clip-v2";
let model = null;
let imageProcessor = null;
let tokenizer = null;
let RawImage = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoModel, AutoProcessor, env } = mod;
  RawImage = mod.RawImage;
  env.allowLocalModels = false;
  model = await AutoModel.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  const processor = await AutoProcessor.from_pretrained(MODEL_ID);
  imageProcessor = processor.image_processor;
  tokenizer = processor.tokenizer;
  device = "wasm";
  post({ type: "ready", device });
}

function l2(vec, dim, i = 0) {
  let s = 0;
  for (let d = 0; d < dim; d++) {
    const v = vec[i * dim + d];
    s += v * v;
  }
  return Math.sqrt(s) || 1;
}

async function run(id, imageURL, labels) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);

  // Image embedding (one forward pass with only pixel_values).
  const imgInputs = await imageProcessor(image);
  const imgOut = await model({ pixel_values: imgInputs.pixel_values });
  const imgEmbT = imgOut.l2norm_image_embeddings ?? imgOut.image_embeddings;

  // Text embeddings for every label (one forward pass with only text).
  const txtInputs = tokenizer(labels, { padding: true, truncation: true });
  const txtOut = await model({
    input_ids: txtInputs.input_ids,
    attention_mask: txtInputs.attention_mask,
  });
  const txtEmbT = txtOut.l2norm_text_embeddings ?? txtOut.text_embeddings;

  const dim = imgEmbT.dims[imgEmbT.dims.length - 1];
  const iv = Array.from(imgEmbT.data);
  const tv = Array.from(txtEmbT.data);
  const inrm = l2(iv, dim);

  const cosines = [];
  for (let i = 0; i < labels.length; i++) {
    let dot = 0;
    const tnrm = l2(tv, dim, i);
    for (let d = 0; d < dim; d++) dot += iv[d] * tv[i * dim + d];
    cosines.push(dot / (inrm * tnrm));
  }

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, labels, cosines, dim, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.labels);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
