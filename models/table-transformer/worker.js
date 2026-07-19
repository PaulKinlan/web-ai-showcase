// Table Transformer DETECTION worker — runs all inference off the main thread.
// Model: Xenova/table-transformer-detection (task: object-detection), WASM backend, q8. ~30 MB.
//
// This is a DETR variant fine-tuned on PubTables-1M to answer one question: WHERE are the tables in a
// document image? It emits two classes — "table" and "table rotated" — as boxes with confidence scores.
// We run it once at a low floor threshold; the page filters that cached list by the score slider so
// dragging never re-runs the model. Boxes come back in ORIGINAL image pixel coordinates
// (percentage:false); the page scales them to the canvas. Uses the SHARED loader from lib/webai.js.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
const FLOOR = 0.3; // run once at this floor; the UI slider filters upward from here.

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "object-detection",
    model: "Xenova/table-transformer-detection",
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
  const output = await pipe(imageURL, { threshold: FLOOR, percentage: false });
  const detections = output.map((d) => ({
    label: d.label,
    score: d.score,
    box: { xmin: d.box.xmin, ymin: d.box.ymin, xmax: d.box.xmax, ymax: d.box.ymax },
  }));
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, detections, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
