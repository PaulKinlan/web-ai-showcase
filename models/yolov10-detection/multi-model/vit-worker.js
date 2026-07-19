// ViT image-classification worker for the YOLOv10 multi-model demo. Off the main thread.
// Model: Xenova/vit-base-patch16-224 (task: image-classification), WASM q8, 1000 ImageNet classes.
// YOLOv10 finds and boxes an object at the COCO level ("dog", "bird"); we crop that box and ask ViT for
// a finer ImageNet label ("golden retriever", "junco"). Two models chained, both on-device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/vit-base-patch16-224";
let classifier = null;

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
  post({ type: "ready", device: "wasm" });
}

async function classify(id, dataUrl, topk) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await classifier(dataUrl, { top_k: topk || 5 });
  const preds = (Array.isArray(out) ? out : [out]).map((o) => ({ label: o.label, score: o.score }));
  post({ type: "classified", id, preds, ms: Math.round(performance.now() - t0), device: "wasm" });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "classify") await classify(e.data.id, e.data.dataUrl, e.data.topk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
