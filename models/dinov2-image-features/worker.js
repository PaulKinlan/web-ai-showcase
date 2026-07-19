// DINOv2-small image-feature-extraction worker — all inference off the main thread so the UI stays
// responsive. Model: Xenova/dinov2-small (task: image-feature-extraction), WASM backend, q8.
//
// DINOv2 is a self-SUPERVISED vision transformer: no labels, no text — it just learns to describe an
// image. One forward pass over the patch grid emits a per-token feature map. We reach the raw tensors
// (like the CLIP page does) so we can return BOTH:
//   • the global image descriptor — the [CLS] token vector (384-d), L2-normalized → cosine = dot product
//   • the patch feature grid — each patch token's cosine similarity to the [CLS] token, a legible
//     "which regions drive the global descriptor" heatmap.
// No second download, no invented API — we drive the pipeline's own processor + model.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let RawImage = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  const loaded = await loadPipeline({
    task: "image-feature-extraction",
    model: "Xenova/dinov2-small",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function l2norm(vec) {
  let s = 0;
  for (const v of vec) s += v * v;
  return Math.sqrt(s);
}

// Largest square that fits in `p` patch tokens — robust to any register/extra tokens.
function gridSideFor(p) {
  const s = Math.round(Math.sqrt(p));
  for (let g = s; g >= 1; g--) if (g * g <= p) return g;
  return 1;
}

async function embed(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const inputs = await pipe.processor(image);
  const out = await pipe.model(inputs);

  // last_hidden_state: [1, 1 + numPatches(+registers), hidden]. Token 0 is the [CLS] descriptor.
  const lhs = out.last_hidden_state;
  const [, tokens, hidden] = lhs.dims;
  const data = lhs.data;

  const cls = Array.from(data.slice(0, hidden));
  const clsNorm = l2norm(cls) || 1;
  const clsEmb = cls.map((v) => v / clsNorm);

  // Patch tokens (skip the CLS token at index 0). Cosine of each patch to the CLS descriptor.
  const numPatches = tokens - 1;
  const side = gridSideFor(numPatches);
  const grid = side * side; // trim to a clean square for the heatmap
  const patchSims = new Array(grid);
  for (let p = 0; p < grid; p++) {
    const base = (p + 1) * hidden;
    let dot = 0, n = 0;
    for (let d = 0; d < hidden; d++) {
      const v = data[base + d];
      dot += v * clsEmb[d];
      n += v * v;
    }
    patchSims[p] = dot / (Math.sqrt(n) || 1);
  }

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    clsEmb,
    dim: hidden,
    clsPreNorm: clsNorm,
    patchSims,
    gridSize: side,
    numPatches,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.image);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
