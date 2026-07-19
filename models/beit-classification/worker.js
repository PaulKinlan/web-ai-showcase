// BEiT image-classification worker — runs ALL inference off the main thread so the control UI stays
// responsive. Like the ViT / ConvNeXt / ResNet workers, we run the LOW-LEVEL forward pass
// (processor → model) rather than the convenience pipeline, so one pass yields the raw pre-softmax
// logits, the full softmax over ImageNet-1k, the human labels, and confidence diagnostics (entropy,
// top-1 margin).
//
// Model: Xenova/beit-base-patch16-224 (task: image-classification), WASM backend, q8. ~86M params,
// ImageNet-1k (1000 classes). BEiT (Bao et al., 2021, "BEiT: BERT Pre-Training of Image Transformers)
// has the SAME ViT-Base backbone as the built ViT demo — 16×16 patches, a transformer, a [CLS] token —
// but a DIFFERENT origin story: it is PRE-TRAINED SELF-SUPERVISED with MASKED IMAGE MODELING (mask a
// chunk of patches, predict each masked patch's discrete visual token — literally "BERT for images"),
// then fine-tuned on ImageNet. The story on this page is PRETRAINING (self-supervised MIM vs the
// supervised ViT), same task, same architecture family.
//
// dtype note: q8 (model_quantized.onnx, ~110 MB) is verified NON-degenerate here — the cats sample
// returns correct ImageNet cat labels (Egyptian cat 0.55, tabby 0.26). So q8 is the honest choice.

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
    model: "Xenova/beit-base-patch16-224",
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
