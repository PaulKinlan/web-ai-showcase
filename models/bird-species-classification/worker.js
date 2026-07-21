// Bird species classification worker — inference off the main thread.
// Model: dennisjooo/Birds-Classifier-EfficientNetB2 (task: image-classification), WASM, fp32 (~34 MB).
// An EfficientNet-B2 fine-tuned on the "BIRDS 525 SPECIES" dataset — it names which of 525 bird species is
// in a photo. DISTINCT from the built general image classifiers (ImageNet ViT/ResNet/ConvNeXt have only ~50
// coarse bird classes): this is FINE-GRAINED species recognition, the way food-classification is fine-grained
// vs ImageNet. Apache-2.0. We import the SHARED loader from lib/webai.js — no invented API. Nothing leaves
// the tab.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "dennisjooo/Birds-Classifier-EfficientNetB2";
const TASK = "image-classification";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: TASK,
    model: MODEL,
    backend: "wasm",
    dtype: "fp32",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Classify one image (URL / data URL) → top-k species with probabilities.
async function classify(id, imageURL, topK) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(imageURL, { top_k: topK || 5 });
  const labels = (Array.isArray(out) ? out : [out]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "classify") await classify(d.id, d.imageURL, d.topK);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
