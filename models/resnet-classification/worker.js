// ResNet-50 image-classification worker — runs ALL inference off the main thread so the control UI
// stays responsive. Like the ViT / ConvNeXt / MobileNet workers, we run the LOW-LEVEL forward pass
// (processor → model) rather than the convenience pipeline, so one pass yields the raw pre-softmax
// logits, the full softmax over ImageNet-1k, the human labels, and confidence diagnostics (entropy,
// top-1 margin).
//
// Model: Xenova/resnet-50 (task: image-classification), WASM backend, q8. ~25.6M params, ImageNet-1k
// (1000 classes). ResNet-50 (He et al., 2015, "Deep Residual Learning for Image Recognition") is the
// ORIGINAL deep-residual CNN: bottleneck blocks wired with identity SKIP CONNECTIONS so gradients flow
// through 50 layers without vanishing. It is the classic backbone almost every later vision model was
// measured against — the story on this page is ARCHITECTURE LINEAGE (2015 residual CNN vs the 2020s
// ConvNeXt and the ViT transformer), same ImageNet task.
//
// dtype note: q8 (model_quantized.onnx, ~26 MB) is verified NON-degenerate here — unlike MobileNetV2's
// depthwise-separable convs, ResNet-50's plain 3×3 / 1×1 convs quantise cleanly and return correct
// ImageNet labels (cats sample → tiger cat 0.96). So q8 is the honest, correct choice.

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
    model: "Xenova/resnet-50",
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

  const logits = Array.from(output.logits.data); // [1000] raw, pre-softmax
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
