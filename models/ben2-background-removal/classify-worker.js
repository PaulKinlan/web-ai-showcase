// Lightweight image-classification worker for the BEN2 multi-model rung — classifies the SUBJECT after
// BEN2 has matted it out. Model: onnx-community/mobilenet_v2_1.0_224 (image-classification), ~14 MB,
// WASM q8. Off the main thread, via the standard transformers.js pipeline. Loaded on demand.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/mobilenet_v2_1.0_224";
let classifier = null;
const device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (classifier) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  classifier = await pipeline("image-classification", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

async function classify(id, imageURL, topk) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await classifier(imageURL, { top_k: topk || 5 });
  post({ type: "result", id, labels: out, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.image, e.data.topk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
