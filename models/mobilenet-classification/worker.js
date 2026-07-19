// MobileNetV2 image-classification worker — runs ALL inference off the main thread. Like the ViT and
// ConvNeXt workers, we run the LOW-LEVEL forward pass (processor → model) so one pass yields the raw
// pre-softmax logits, the full softmax, the human labels, and confidence diagnostics.
//
// Model: onnx-community/mobilenet_v2_1.0_224 (task: image-classification), WASM backend, fp32.
// ~3.5M params — the "runs fast on a phone" backbone. MobileNetV2 (Sandler et al., 2018) is built from
// depthwise-separable convolutions + inverted residual blocks with linear bottlenecks, trading a little
// accuracy for a tiny, low-latency model. Its head has 1001 classes: ImageNet-1k PLUS a `background`
// class at index 0 (the TF-Slim / Google convention) — the ViT and ConvNeXt demos have exactly 1000.
//
// dtype note: this repo prefers q8, but MobileNetV2's depthwise-separable convs quantise DEGENERATELY
// with the generic int8 export (ConvInteger has no WASM kernel, and the uint8/q8 builds emit garbage
// labels — verified). fp32 is the honest, correct choice and is still tiny (~14 MB) — a fraction of the
// ViT (88 MB) and ConvNeXt (29 MB) downloads, which is exactly the mobile-backbone story.

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
    model: "onnx-community/mobilenet_v2_1.0_224",
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

  const logits = Array.from(output.logits.data); // [1001] raw, pre-softmax (index 0 = background)
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
