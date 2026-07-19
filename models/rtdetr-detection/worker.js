// RT-DETR object-detection worker — all inference off the main thread so the control UI stays
// responsive. RT-DETR is a *real-time* DETR: it keeps DETR's end-to-end, set-prediction design (no
// hand-tuned anchors) but is trained to be **NMS-free** — it emits one clean box per object directly,
// with no non-maximum-suppression post-pass. That is what makes it fast enough for live video where
// the original ResNet-50 DETR is not.
//
// Model: onnx-community/rtdetr_r18vd (task: object-detection), WASM backend, q8 (~22 MB).
// One forward pass returns every detection above a low floor threshold; the page filters that cached
// list by the score slider client-side, so dragging the slider never re-runs the model. Boxes come
// back in ORIGINAL image pixel coordinates (percentage:false). We use the SHARED loader from
// lib/webai.js — no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

// Run detection at this floor once; the UI slider filters the cached result upward from here.
const FLOOR = 0.05;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "object-detection",
    model: "onnx-community/rtdetr_r18vd",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  // percentage:false → boxes in absolute pixel coordinates of the source image. RT-DETR is NMS-free,
  // so this list is already de-duplicated — no extra suppression pass is applied.
  const output = await pipe(imageURL, { threshold: FLOOR, percentage: false });
  const detections = output.map((d) => ({
    label: d.label,
    score: d.score,
    box: {
      xmin: d.box.xmin,
      ymin: d.box.ymin,
      xmax: d.box.xmax,
      ymax: d.box.ymax,
    },
  }));
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, detections, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
