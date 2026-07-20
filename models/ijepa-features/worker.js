// I-JEPA (ijepa_vith14_1k) image-feature-extraction worker — all inference off the main thread so the
// UI stays responsive. Model: onnx-community/ijepa_vith14_1k (task: image-feature-extraction), WASM,
// q8. No version pin — the model_type "ijepa" (IJepaModel) is registered in transformers.js 3.7.5, and
// loads on the existing image-feature-extraction pipeline.
//
// I-JEPA (Image Joint-Embedding Predictive Architecture; Assran et al., 2023) is a DISTINCT flavour of
// self-supervised vision from DINOv2/CLIP:
//   • It learns by PREDICTION IN REPRESENTATION SPACE. Given a "context" block of an image, a predictor
//     network must predict the ENCODER's representations of several masked "target" blocks — not their
//     pixels (unlike MAE), and not via image-image contrast (unlike DINO/CLIP). There are no hand-crafted
//     augmentations and no negative pairs. The target encoder is an EMA of the context encoder.
//   • The backbone is a PLAIN ViT with NO [CLS] token and NO register tokens — the output is purely the
//     patch-token grid. The image-level descriptor is the MEAN of the patch tokens (average pooling),
//     which is exactly how I-JEPA features are used for retrieval/linear-probe.
//
// ijepa_vith14_1k is ViT-Huge/14 @ 224px → a 16×16 = 256-patch grid, hidden size 1280. One forward pass
// emits last_hidden_state [1, 256, 1280]. We reach the raw tensors (like the DINOv2/CLIP pages) so we
// can return:
//   • the global image descriptor — the MEAN of the 256 patch tokens (1280-d), L2-normalized → cosine = dot
//   • the patch feature grid — each patch token's cosine to the global descriptor (a "which regions define
//     the image" heatmap) AND each patch token's L2 NORM (dense feature magnitude across the grid)
//   • (on request) per-patch unit embeddings, for dense patch-to-patch correspondence between two images.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/ijepa_vith14_1k";

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
    model: MODEL_ID,
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

// Largest square that fits in `p` patch tokens — robust to the exact token count.
function gridSideFor(p) {
  const s = Math.round(Math.sqrt(p));
  for (let g = s; g >= 1; g--) if (g * g <= p) return g;
  return 1;
}

async function embed(id, imageURL, wantPatches) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const inputs = await pipe.processor(image);
  const out = await pipe.model(inputs);

  // last_hidden_state: [1, numPatches, hidden]. I-JEPA has NO [CLS] and NO register tokens — every token
  // is a patch (row-major over the patch grid).
  const lhs = out.last_hidden_state;
  const [, tokens, hidden] = lhs.dims;
  const data = lhs.data;

  const side = gridSideFor(tokens);
  const grid = side * side; // trim to a clean square for the heatmap (256 → 16×16 here)

  // Global image descriptor = MEAN of the patch tokens (I-JEPA's pooled embedding), then L2-normalized.
  const mean = new Float64Array(hidden);
  for (let p = 0; p < tokens; p++) {
    const base = p * hidden;
    for (let d = 0; d < hidden; d++) mean[d] += data[base + d];
  }
  for (let d = 0; d < hidden; d++) mean[d] /= tokens;
  const meanPreNorm = l2norm(mean) || 1;
  const imgEmb = Array.from(mean, (v) => v / meanPreNorm);

  // Per-patch: cosine to the global descriptor + L2 norm; optionally the unit embedding for correspondence.
  const patchSims = new Array(grid);
  const patchNorms = new Array(grid);
  let patchEmbs = null;
  if (wantPatches) patchEmbs = new Float32Array(grid * hidden);
  for (let p = 0; p < grid; p++) {
    const base = p * hidden;
    let dot = 0, n = 0;
    for (let d = 0; d < hidden; d++) {
      const v = data[base + d];
      dot += v * imgEmb[d];
      n += v * v;
    }
    const norm = Math.sqrt(n) || 1;
    patchSims[p] = dot / norm;
    patchNorms[p] = norm;
    if (patchEmbs) {
      for (let d = 0; d < hidden; d++) patchEmbs[p * hidden + d] = data[base + d] / norm;
    }
  }

  const ms = Math.round(performance.now() - t0);
  const msg = {
    type: "result",
    id,
    clsEmb: imgEmb, // "clsEmb" kept as the field name for the shared front-end; here it's the MEAN descriptor
    dim: hidden,
    clsPreNorm: meanPreNorm,
    patchSims,
    patchNorms,
    gridSize: side,
    numPatches: grid,
    ms,
    device,
  };
  if (patchEmbs) {
    msg.patchEmbs = patchEmbs;
    msg.patchDim = hidden;
    self.postMessage(msg, [patchEmbs.buffer]);
  } else {
    post(msg);
  }
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.image, e.data.patches);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
