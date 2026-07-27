// MobileNetV3-Small image-classification worker — all inference runs off the main thread. Like the
// established image classifiers, a low-level processor → model pass exposes raw pre-softmax logits,
// the full softmax, ImageNet labels, and confidence diagnostics from one real inference.
//
// Exact upstream: timm/mobilenetv3_small_100.lamb_in1k (Apache-2.0), via the canonical
// onnx-community/mobilenetv3_small_100.lamb_in1k Transformers.js conversion. WASM fp32, 2.5M params,
// 1,000 ImageNet outputs. MobileNetV3-Small combines hardware-aware architecture search, inverted
// residuals, squeeze-and-excitation, and hard-swish activations (Howard et al., 2019).
//
// The strict browser probe found the canonical q8 graph semantically broken on a ledgered cats image,
// while fp32 returned Egyptian/tabby/Persian cat. fp32 is the honest choice and is still only ~10 MB.

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
    task: "image-classification",
    model: "onnx-community/mobilenetv3_small_100.lamb_in1k",
    backend: "wasm",
    dtype: "fp32",
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

  const logits = Array.from(output.logits.data); // [1000] raw, pre-softmax ImageNet logits
  const probs = softmax(logits);
  const id2label = pipe.model.config.id2label || {};

  const order = probs
    .map((p, i) => i)
    .sort((a, b) => probs[b] - probs[a])
    .slice(0, Math.max(1, Math.min(20, topK || 5)));
  const top = order.map((i) => ({
    label: id2label[i] ?? `class ${i}`,
    prob: probs[i],
    logit: logits[i],
    index: i,
  }));

  const sorted = [...probs].sort((a, b) => b - a);
  const margin = sorted[0] - (sorted[1] ?? 0);
  const entropy = normEntropy(probs);

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, top, entropy, margin, numClasses: probs.length, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.topK);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
