// MobileCLIP zero-shot worker — ALL inference off the main thread (modern-web-guidance:
// break-up-long-tasks). MobileCLIP is Apple's FAST on-device CLIP: a mobile-optimised vision tower
// that encodes an image in a fraction of the time a ViT-B/16 (the built CLIP page) takes, at similar
// zero-shot accuracy. This page's whole point is LATENCY — we time the vision encode and the text
// encode separately so you can compare.
//
// Model: Xenova/mobileclip_s0 (task: zero-shot-image-classification). This repo ships the two towers
// SEPARATELY (vision_model + text_model ONNX), not a single fused CLIPModel, so the standard
// zero-shot pipeline can't load it — we load the two projection models by hand (the documented
// transformers.js pattern) and compute the image↔text cosine ourselves. No invented API.
//
// DTYPE (measured, not guessed): quantising the VISION tower to q8 measurably FLIPS the ranking
// (cats→dog on the reference image), so the vision tower runs fp16 and only the (much larger) TEXT
// tower is q8. See _questions.json for the evidence.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/mobileclip_s0";
const VISION_DTYPE = "fp16";
const TEXT_DTYPE = "q8";
const CONTEXT_LENGTH = 77; // OpenCLIP text tower has fixed 77-token positional embeddings.
const LOGIT_SCALE = 100; // Standard CLIP temperature (exp(ln 100)); scales cosine → logits for softmax.

let tokenizer = null;
let processor = null;
let visionModel = null;
let textModel = null;
let RawImage = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (visionModel && textModel) return;
  const mod = await import(TRANSFORMERS_URL);
  const {
    AutoProcessor,
    AutoTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    env,
  } = mod;
  RawImage = mod.RawImage;
  env.allowLocalModels = false;
  const onProgress = (p) => post({ type: "progress", p });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress });
  processor = await AutoProcessor.from_pretrained(MODEL_ID, { progress_callback: onProgress });
  visionModel = await CLIPVisionModelWithProjection.from_pretrained(MODEL_ID, {
    dtype: VISION_DTYPE,
    device: "wasm",
    progress_callback: onProgress,
  });
  textModel = await CLIPTextModelWithProjection.from_pretrained(MODEL_ID, {
    dtype: TEXT_DTYPE,
    device: "wasm",
    progress_callback: onProgress,
  });
  post({ type: "ready", device });
}

function l2normalize(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  s = Math.sqrt(s) || 1;
  return vec.map((v) => v / s);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function run(id, imageURL, labels) {
  await ensureLoaded();

  // --- vision encode (timed on its own — the MobileCLIP latency story) ---
  const image = await RawImage.read(imageURL);
  const imageInputs = await processor(image);
  const tV = performance.now();
  const { image_embeds } = await visionModel(imageInputs);
  const visionMs = Math.round(performance.now() - tV);

  // --- text encode (fixed 77-token context) ---
  const textInputs = tokenizer(labels, {
    padding: "max_length",
    max_length: CONTEXT_LENGTH,
    truncation: true,
  });
  const tT = performance.now();
  const { text_embeds } = await textModel(textInputs);
  const textMs = Math.round(performance.now() - tT);

  // --- cosine similarity in the shared embedding space ---
  const dim = image_embeds.dims[image_embeds.dims.length - 1];
  const imgVec = l2normalize(Array.from(image_embeds.data));
  const txt = Array.from(text_embeds.data);
  const cosines = labels.map((_, i) => {
    const tVec = l2normalize(txt.slice(i * dim, (i + 1) * dim));
    let dot = 0;
    for (let k = 0; k < dim; k++) dot += imgVec[k] * tVec[k];
    return dot;
  });
  const logits = cosines.map((c) => c * LOGIT_SCALE);
  const probs = softmax(logits);

  post({
    type: "result",
    id,
    labels,
    probs,
    logits,
    cosines,
    imgDims: image_embeds.dims,
    txtDims: text_embeds.dims,
    visionMs,
    textMs,
    ms: visionMs + textMs,
    device,
  });
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
