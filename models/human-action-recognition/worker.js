// Human action-recognition worker — runs ALL inference off the main thread so the control UI stays
// responsive. Like the Food-101 and ViT demos it runs the LOW-LEVEL forward pass (processor → model)
// instead of the convenience pipeline, so a single pass yields the raw pre-softmax logits, the full
// softmax distribution over the 15 actions, the human-readable action names, and confidence diagnostics
// (entropy, top-1 margin).
//
// Model: onnx-community/Human-Action-Recognition-VIT-Base-patch16-224-ONNX (task: image-classification),
// WASM backend, q8 (model_quantized.onnx). A ViT-Base fine-tuned to recognise WHAT A PERSON IS DOING from
// a single still — 15 everyday actions (Running, Cycling, Drinking, Eating, Dancing, Using Laptop, …).
// DISTINCT from the built object/food/document/expression/age classifiers: it reads ACTIVITY, not the
// object, dish, page type, emotion, or age. The weights are Apache-2.0 (rvv-karma/Human-Action-Recognition-
// VIT-Base-patch16-224, a fine-tune of Apache-2.0 google/vit-base-patch16-224); Apache-2.0 permits
// redistribution, so they stay Apache-2.0 in the onnx-community conversion despite its blank license field.
//
// Correctness proven FIRST in headless Chrome (transformers.js 3.7.5, WASM, q8): a licensed marathon photo
// classifies as Running 0.98, a person holding a cup as Drinking 0.96, a market scene as Cycling 0.85.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Human-Action-Recognition-VIT-Base-patch16-224-ONNX";

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
    task: "image-classification",
    model: MODEL_ID,
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

// Shannon entropy of the distribution, normalised to [0,1] against log(numClasses). High = uncertain.
function normEntropy(probs) {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h / Math.log(probs.length);
}

async function run(id, imageURL, topK) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const inputs = await pipe.processor(image);
  const output = await pipe.model(inputs);

  const logits = Array.from(output.logits.data); // [15] raw, pre-softmax
  const probs = softmax(logits);
  const id2label = pipe.model.config.id2label || {};

  const order = probs
    .map((p, i) => i)
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, Math.max(1, Math.min(15, topK || 5)));
  const top = order.map((i) => ({
    label: id2label[i] ?? `class ${i}`,
    prob: probs[i],
    logit: logits[i],
    index: i,
  }));

  const sorted = [...probs].sort((a, b) => b - a);
  const margin = sorted[0] - (sorted[1] ?? 0); // top-1 vs top-2 gap
  const entropy = normEntropy(probs);

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, top, entropy, margin, numClasses: probs.length, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.topK);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
