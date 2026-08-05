// fashion-clip worker — text→image retrieval off the main thread, WASM fp32.
// Model: patrickjohncyh/fashion-clip (task: zero-shot-image-classification / CLIP retrieval).
//
// Fashion-CLIP is a CLIP model fine-tuned on fashion pairs, but its text+vision encoders do generic
// CLIP retrieval: embed a catalog of images once, embed a free-text query, and rank the images by
// cosine similarity of the real embeddings. We call the model directly (CLIPModel) so every score
// shown is a genuine cosine similarity — no canned rankings. Files live in the repo's onnx/
// subfolder (single-file fp32 export, no external data), so we pass subfolder:"onnx".

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "patrickjohncyh/fashion-clip";
// Immutable revision — MUST stay byte-for-byte identical to models/fashion-clip/worker.js so the
// shared transformers-cache serves both demos from ONE download (conformance assertion
// revision-pin-matches-main-fashion-clip enforces this; a one-sided bump fails the gate).
const REVISION = "7e3ba62ce16b379a1ab479346b66f192e76f51b7";
const SUBFOLDER = "onnx";
let model = null;
let tokenizer = null;
let processor = null;
let device = "wasm";
let catalogEmbeds = null; // Float32Array [n, dim]
let cachedPixelValues = null; // the catalog's vision inputs (the merged CLIP graph needs them present)

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoProcessor, CLIPModel } = await import(TRANSFORMERS_URL);
  tokenizer = await AutoTokenizer.from_pretrained("patrickjohncyh/fashion-clip", {
    subfolder: SUBFOLDER,
    revision: REVISION,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip", {
    subfolder: SUBFOLDER,
    revision: REVISION,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await CLIPModel.from_pretrained("patrickjohncyh/fashion-clip", {
    subfolder: SUBFOLDER,
    revision: REVISION,
    dtype: "fp32",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

function cosine(a, b, dim) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < dim; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1e-9);
}

async function embedCatalog(id, imageURLs) {
  await ensureLoaded();
  const { RawImage } = await import(TRANSFORMERS_URL);
  const t0 = performance.now();
  const images = [];
  for (let i = 0; i < imageURLs.length; i++) {
    post({ type: "phase", id, phase: `embedding image ${i + 1}/${imageURLs.length}` });
    images.push(await RawImage.read(imageURLs[i]));
  }
  const inputs = await processor(images);
  cachedPixelValues = inputs.pixel_values;
  const out = await model({ ...(await tokenizer([""])), ...inputs });
  const embeds = (out.image_embeds ?? out).data; // Float32Array
  const dim = (out.image_embeds ?? out).dims[1];
  catalogEmbeds = new Float32Array(embeds);
  post({
    type: "catalog",
    id,
    n: imageURLs.length,
    dim,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

async function search(id, query) {
  await ensureLoaded();
  if (!catalogEmbeds) {
    post({ type: "error", id, message: "Embed the catalog first." });
    return;
  }
  const t0 = performance.now();
  const inputs = await tokenizer([query], { padding: true, truncation: true });
  const out = await model({ ...inputs, pixel_values: cachedPixelValues });
  const textEmbeds = (out.text_embeds ?? out).data;
  const dim = (out.text_embeds ?? out).dims[1];
  const n = catalogEmbeds.length / dim;
  const scores = [];
  for (let i = 0; i < n; i++) {
    scores.push(cosine(catalogEmbeds.subarray(i * dim, (i + 1) * dim), textEmbeds.subarray(0, dim), dim));
  }
  const ranked = scores
    .map((score, idx) => ({ idx, score }))
    .sort((a, b) => b.score - a.score);
  post({
    type: "result",
    id,
    ranked,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "embedCatalog") await embedCatalog(id, e.data.imageURLs);
    else if (type === "search") await search(id, e.data.query);
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
