// DINOv2-with-registers-small image-feature-extraction worker — all inference off the main thread so
// the UI stays responsive. Model: onnx-community/dinov2-with-registers-small
// (task: image-feature-extraction), WASM backend, q8. No version pin — the model_type
// "dinov2_with_registers" loads on the existing image-feature-extraction pipeline in transformers.js
// 3.7.5.
//
// The register-token idea (Darcet et al., 2023): a plain ViT trained self-supervised (DINOv2) develops
// a handful of HIGH-NORM "artifact" patch tokens in low-information background regions — the network
// hijacks those patches to stash global information, which corrupts the local patch features used for
// dense tasks. Adding a few dedicated REGISTER tokens to the sequence gives the model somewhere to put
// that global scratch state, so the patch tokens stay clean. The result: smoother attention maps and
// per-patch features that segment the foreground without ever being told what it is.
//
// One forward pass over the patch grid emits a per-token feature map. We reach the raw tensors (like the
// CLIP/DINOv2 pages do) so we can return:
//   • the global image descriptor — the [CLS] token vector (384-d), L2-normalized → cosine = dot product
//   • the patch feature grid — each patch token's cosine to the [CLS] token (a "which regions define the
//     image" heatmap) AND each patch token's L2 NORM (the artifact story — with registers these stay
//     uniform instead of spiking)
//   • the register tokens' norms — they carry the global scratch state the patches used to hold
//   • (on request) per-patch unit embeddings, for dense patch-to-patch correspondence between two images.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/dinov2-with-registers-small";

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

// Largest square that fits in `p` patch tokens — robust to any register/extra token count.
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

  // last_hidden_state: [1, 1 + numRegisters + numPatches, hidden]. Token 0 is the [CLS] descriptor,
  // then the register tokens, then the patch tokens (row-major over the patch grid).
  const lhs = out.last_hidden_state;
  const [, tokens, hidden] = lhs.dims;
  const data = lhs.data;

  // How many register tokens? Prefer the model config; otherwise recover it from the sequence length
  // (tokens - 1 - largestSquare). dinov2-with-registers-small uses 4.
  const cfg = pipe.model.config || {};
  let numRegisters = Number.isInteger(cfg.num_register_tokens) ? cfg.num_register_tokens : null;
  if (numRegisters == null) {
    const rest = tokens - 1;
    numRegisters = rest - gridSideFor(rest) ** 2;
    if (numRegisters < 0) numRegisters = 0;
  }

  const cls = Array.from(data.slice(0, hidden));
  const clsNorm = l2norm(cls) || 1;
  const clsEmb = cls.map((v) => v / clsNorm);

  // Register token norms (tokens 1..numRegisters).
  const registerNorms = [];
  for (let r = 0; r < numRegisters; r++) {
    const base = (1 + r) * hidden;
    let n = 0;
    for (let d = 0; d < hidden; d++) n += data[base + d] * data[base + d];
    registerNorms.push(Math.sqrt(n));
  }

  // Patch tokens start after CLS + registers.
  const patchStart = 1 + numRegisters;
  const numPatches = tokens - patchStart;
  const side = gridSideFor(numPatches);
  const grid = side * side; // trim to a clean square for the heatmap
  const patchSims = new Array(grid);
  const patchNorms = new Array(grid);
  let patchEmbs = null;
  if (wantPatches) patchEmbs = new Float32Array(grid * hidden);
  for (let p = 0; p < grid; p++) {
    const base = (patchStart + p) * hidden;
    let dot = 0, n = 0;
    for (let d = 0; d < hidden; d++) {
      const v = data[base + d];
      dot += v * clsEmb[d];
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
    clsEmb,
    dim: hidden,
    clsPreNorm: clsNorm,
    patchSims,
    patchNorms,
    registerNorms,
    numRegisters,
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
