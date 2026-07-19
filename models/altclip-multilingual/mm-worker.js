// Multilingual-CLIP multi-model worker — a real two-model composition, off the main thread:
//   1. ViT-GPT2 (image-to-text) writes an English caption for the image.
//   2. jina-clip-v2 grounds that caption cross-lingually — scoring the caption (and any multilingual
//      paraphrases the user adds) against the image via cosine similarity.
// Captioner: Xenova/vit-gpt2-image-captioning (~250 MB, q8). CLIP: jinaai/jina-clip-v2 (~874 MB, q8).

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const CAP_ID = "Xenova/vit-gpt2-image-captioning";
const CLIP_ID = "jinaai/jina-clip-v2";
let captioner = null;
let model = null;
let imageProcessor = null;
let tokenizer = null;
let RawImage = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model && captioner) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoModel, AutoProcessor, env } = mod;
  RawImage = mod.RawImage;
  env.allowLocalModels = false;
  const cap = await loadPipeline({
    task: "image-to-text",
    model: CAP_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  captioner = cap.pipe;
  model = await AutoModel.from_pretrained(CLIP_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  const processor = await AutoProcessor.from_pretrained(CLIP_ID);
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

async function run(id, imageURL, extraLabels) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);

  // 1) Caption the image (English).
  const capOut = await captioner(image);
  const caption = (Array.isArray(capOut) ? capOut[0]?.generated_text : capOut?.generated_text) ||
    "";
  const capMs = Math.round(performance.now() - t0);

  // 2) Ground: score the caption + any extra multilingual labels against the image.
  const labels = [caption, ...(extraLabels || [])].filter(Boolean);
  const imgInputs = await imageProcessor(image);
  const imgOut = await model({ pixel_values: imgInputs.pixel_values });
  const imgEmbT = imgOut.l2norm_image_embeddings ?? imgOut.image_embeddings;
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

  post({
    type: "result",
    id,
    caption,
    labels,
    cosines,
    capMs,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.extraLabels);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
