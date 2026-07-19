// OWLViT (v1) zero-shot / open-vocabulary object-detection worker — runs ALL inference off the main
// thread. OWLViT is the ORIGINAL open-vocabulary detector: it embeds each free-text query and each image
// region into a shared CLIP space and boxes the regions that match. One forward pass per query set
// returns every candidate above a low floor; the page filters that cached list by the score slider
// client-side, so dragging the slider never re-runs the model.
//
// Model: Xenova/owlvit-base-patch32 (task: zero-shot-object-detection), WASM backend, q8
// (model_quantized.onnx, ~155 MB). Boxes come back in ORIGINAL image pixel coordinates
// (percentage:false). Shared loader from webai.js. Verified runnable: two cats + a remote correctly
// boxed on the sample image at ~1.7 s inference.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

// Run detection at this floor once per query set; the UI slider filters the cached result upward.
const FLOOR = 0.01;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "zero-shot-object-detection",
    model: "Xenova/owlvit-base-patch32",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function run(id, imageURL, queries) {
  await ensureLoaded();
  const t0 = performance.now();
  // percentage:false → boxes in absolute pixel coordinates of the source image.
  const output = await pipe(imageURL, queries, { threshold: FLOOR, percentage: false });
  const detections = output.map((d) => ({
    label: d.label,
    score: d.score,
    box: { xmin: d.box.xmin, ymin: d.box.ymin, xmax: d.box.xmax, ymax: d.box.ymax },
  }));
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, detections, queries, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.queries);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
