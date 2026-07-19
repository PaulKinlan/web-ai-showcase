// Table Transformer STRUCTURE-RECOGNITION worker — inference off the main thread.
// Model: Xenova/table-transformer-structure-recognition (task: object-detection), WASM, q8. ~30 MB.
//
// Stage two of the pipeline. Given a CROPPED table image (the region stage one found), it decomposes the
// table into its parts, each as its own box + score. Six classes:
//   table · table row · table column · table column header · table projected row header · table spanning cell
// Rows span the full width, columns the full height; their intersection is a cell. We run at a modest
// floor and return everything in the cropped image's own pixel coordinates. The page reconstructs the
// grid (rows × columns) and can read individual cells. Uses the SHARED loader from lib/webai.js.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
const FLOOR = 0.25;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "object-detection",
    model: "Xenova/table-transformer-structure-recognition",
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
  const cells = output.map((d) => ({
    label: d.label,
    score: d.score,
    box: { xmin: d.box.xmin, ymin: d.box.ymin, xmax: d.box.xmax, ymax: d.box.ymax },
  }));
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, cells, ms, device });
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
