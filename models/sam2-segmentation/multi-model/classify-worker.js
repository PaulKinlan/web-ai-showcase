// Lightweight image-classification worker for the SAM 2 multi-model demo. Off the main thread.
// Model: Xenova/mobilevit-small (image-classification), Transformers.js v3.7.5, WASM. We classify the
// tight crop of whatever SAM 2 segmented — segmentation → recognition, composed on-device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  pipe = await pipeline("image-classification", "Xenova/mobilevit-small", {
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "wasm" });
}

async function classify(id, imageURL, topk) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(imageURL, { top_k: topk ?? 5 });
  post({ type: "result", id, labels: out, ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "classify") await classify(e.data.id, e.data.image, e.data.topk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
