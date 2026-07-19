// D-FINE object-detection worker — all inference off the main thread so the control UI stays
// responsive. D-FINE is a *SOTA real-time* detector built on the RT-DETR decoder, but it redefines
// bounding-box regression: instead of predicting four raw box coordinates, each edge is predicted as a
// probability DISTRIBUTION over discretized offsets that the decoder layers iteratively refine
// (Fine-grained Distribution Refinement, FDR), and the final refined distributions are distilled back
// into the earlier layers (Global Optimal Localization Self-Distillation, GO-LSD). The result is
// higher COCO accuracy than RT-DETR at the same real-time speed — and it stays NMS-free.
//
// Model: onnx-community/dfine_s_coco-ONNX (task: object-detection, model_type d_fine, wired via the
// registered RTDetrImageProcessor), WASM backend, q8 (~10 MB). One forward pass returns every
// detection above a low floor; the page filters that cached list by the score slider client-side, so
// dragging the slider never re-runs the model. Boxes come back in ORIGINAL image pixel coordinates
// (percentage:false). We use the SHARED loader from lib/webai.js — no invented API.

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
    model: "onnx-community/dfine_s_coco-ONNX",
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
  // percentage:false → boxes in absolute pixel coordinates of the source image. D-FINE is NMS-free
  // (set-prediction decoder), so this list is already de-duplicated — no extra suppression pass.
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
