// CLIP zero-shot worker — runs ALL inference off the main thread so the control UI stays responsive.
// One forward pass returns everything both the "Run it" bars and the "See inside" surface need:
// softmax probabilities, the raw scaled logits, the cosine similarities, and the embedding dims.
//
// Model: Xenova/clip-vit-base-patch16 (task: zero-shot-image-classification), WASM backend, q8.
// We import the SHARED loaders from lib/webai.js and reuse the pipeline's own model/processor/
// tokenizer to reach the low-level tensors — no second download, no invented API.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let RawImage = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  const loaded = await loadPipeline({
    task: "zero-shot-image-classification",
    model: "Xenova/clip-vit-base-patch16",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Cosine similarity between the single image embedding and each per-label text embedding.
function cosines(imageEmb, textEmb, dim, n) {
  const imgN = l2norm(imageEmb);
  const out = [];
  for (let i = 0; i < n; i++) {
    let dot = 0;
    let tN = 0;
    for (let d = 0; d < dim; d++) {
      const t = textEmb[i * dim + d];
      dot += imageEmb[d] * t;
      tN += t * t;
    }
    out.push(dot / (imgN * Math.sqrt(tN) || 1));
  }
  return out;
}

async function run(id, imageURL, labels) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const imageInputs = await pipe.processor(image);
  const textInputs = pipe.tokenizer(labels, { padding: true, truncation: true });
  const output = await pipe.model({ ...textInputs, ...imageInputs });

  const logits = Array.from(output.logits_per_image.data); // scaled cosine sims (× logit_scale)
  const probs = softmax(logits);

  // Real embedding tensors from the same forward pass — the heart of "See inside".
  const imgDims = output.image_embeds?.dims ?? null;
  const txtDims = output.text_embeds?.dims ?? null;
  let cos = null;
  if (output.image_embeds && output.text_embeds && imgDims && txtDims) {
    const dim = imgDims[imgDims.length - 1];
    cos = cosines(
      Array.from(output.image_embeds.data),
      Array.from(output.text_embeds.data),
      dim,
      labels.length,
    );
  } else {
    // Fallback: recover cosine from the scaled logits (logit_scale ≈ exp(model.logit_scale)).
    const scale = Math.exp(pipe.model?.config?.logit_scale_init_value ?? Math.log(100));
    cos = logits.map((l) => l / scale);
  }

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    labels,
    probs,
    logits,
    cosines: cos,
    imgDims,
    txtDims,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.labels);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
