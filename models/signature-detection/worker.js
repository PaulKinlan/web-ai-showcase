// Signature detection worker — object detection off the main thread so the UI stays responsive.
// Model: onnx-community/yolos-base-signature-detection-ONNX (task: object-detection), WASM, q4f16 (~210 MB).
// A YOLOS (DETR-style, ViT backbone) detector fine-tuned to LOCATE handwritten signatures in document
// images — a single class, "signature". DISTINCT from every built detector (DETR/YOLO/RT-DETR/D-FINE/YOLOS
// on COCO objects, face-detector, craft text detection, table-transformer): it finds SIGNATURES, the way
// table-transformer finds tables. Apache-2.0. Boxes come back in original-image pixel coordinates
// (percentage:false); the page scales them to the display. We import the SHARED loader — no invented API.
// Nothing leaves the tab.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/yolos-base-signature-detection-ONNX";
const TASK = "object-detection";

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
    dtype: "q4f16",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Detect signatures in an image → boxes (original pixel coords) + scores, above `threshold`.
async function detect(id, imageURL, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(imageURL, { threshold: threshold ?? 0.3, percentage: false });
  const dets = (Array.isArray(out) ? out : []).map((d) => ({
    label: d.label,
    score: d.score,
    xmin: d.box.xmin,
    ymin: d.box.ymin,
    xmax: d.box.xmax,
    ymax: d.box.ymax,
  }));
  post({ type: "result", id, dets, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "detect") await detect(d.id, d.imageURL, d.threshold);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
