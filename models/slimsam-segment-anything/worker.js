// SlimSAM point-prompt worker — runs ALL inference off the main thread. SAM has two stages:
// an expensive vision encoder (run ONCE per image, cached) and a cheap prompt/mask decoder (run per
// click). We cache the image embeddings so every extra click is fast — the canonical Transformers.js
// segment-anything pattern, not an invented API.
//
// Model: Xenova/slimsam-77-uniform (task: mask-generation / SAM), WASM backend, q8.
// We use SamModel + AutoProcessor directly (the pipeline can't take a point prompt).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let model = null;
let processor = null;
let Tensor = null;
let RawImage = null;
let device = "wasm";

// Per-image cache: the vision-encoder output + the sizes post_process_masks needs.
let cache = null; // { embeddings, imageInputs, originalSize:[h,w] }

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Overlay colour + opacity — kept in sync with sam.js MASK_RGB (indigo accent). Baking the coloured
// overlay HERE (off the main thread) is the invariant-15 dense-output composite: the page used to run
// maskToImageData + strokeMaskEdge (two per-pixel loops over W×H) on every click/keyboard redraw
// (~11–22ms @1080p, measured). Now the worker builds one translucent-fill + crisp-edge RGBA layer into
// an OffscreenCanvas and transfers an ImageBitmap; the page only drawImage()s it. Mask is identical.
const MASK_RGB = [75, 58, 255];
const MASK_ALPHA = 0.5;

/** Build a coloured translucent overlay (fill + 1px opaque edge) for a raw H×W 0/1 plane as an
 *  ImageBitmap, transparent outside the mask so the page can drawImage() it straight over the photo. */
function overlayBitmap(plane, w, h) {
  const img = new ImageData(w, h);
  const d = img.data;
  const [r, g, b] = MASK_RGB;
  const fa = Math.round(MASK_ALPHA * 255);
  for (let i = 0; i < w * h; i++) {
    if (plane[i]) {
      const j = i * 4;
      d[j] = r;
      d[j + 1] = g;
      d[j + 2] = b;
      d[j + 3] = fa;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!plane[i]) continue;
      const boundary = x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
        !plane[i - 1] || !plane[i + 1] || !plane[i - w] || !plane[i + w];
      if (boundary) {
        const j = i * 4;
        d[j] = r;
        d[j + 1] = g;
        d[j + 2] = b;
        d[j + 3] = 255;
      }
    }
  }
  const oc = new OffscreenCanvas(w, h);
  oc.getContext("2d").putImageData(img, 0, 0);
  return oc.transferToImageBitmap();
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { SamModel, AutoProcessor, RawImage: RI, Tensor: T, env } = mod;
  env.allowLocalModels = false;
  Tensor = T;
  RawImage = RI;
  // q8 vision encoder keeps the download small; the mask decoder stays fp32 so the masks are crisp.
  model = await SamModel.from_pretrained("Xenova/slimsam-77-uniform", {
    dtype: { vision_encoder: "q8", prompt_encoder_mask_decoder: "fp32" },
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained("Xenova/slimsam-77-uniform");
  post({ type: "ready", device });
}

// Compute + cache the vision-encoder embeddings for one image. Clicks reuse this.
async function embed(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const imageInputs = await processor(image);
  const embeddings = await model.get_image_embeddings(imageInputs);
  const originalSize = imageInputs.original_sizes[0]; // [height, width]
  cache = { embeddings, imageInputs, originalSize };
  post({ type: "embedded", id, originalSize, ms: Math.round(performance.now() - t0), device });
}

// Decode a mask from one or more point prompts (normalized [0,1] coords + label 1=fg/0=bg).
async function segment(id, points) {
  if (!cache) throw new Error("Call embed(image) before segment().");
  const t0 = performance.now();
  const reshaped = cache.imageInputs.reshaped_input_sizes[0]; // [rH, rW]
  const flatPoints = [];
  const flatLabels = [];
  for (const p of points) {
    flatPoints.push(p.x * reshaped[1], p.y * reshaped[0]);
    flatLabels.push(BigInt(p.label ?? 1));
  }
  const input_points = new Tensor("float32", flatPoints, [1, 1, points.length, 2]);
  const input_labels = new Tensor("int64", flatLabels, [1, 1, points.length]);

  const outputs = await model({
    ...cache.embeddings,
    input_points,
    input_labels,
  });

  const masks = await processor.post_process_masks(
    outputs.pred_masks,
    cache.imageInputs.original_sizes,
    cache.imageInputs.reshaped_input_sizes,
  );
  const maskTensor = masks[0]; // dims [1, numMasks, H, W], bool
  const [, numMasks, H, W] = maskTensor.dims;
  const scores = Array.from(outputs.iou_scores.data); // numMasks predicted-IoU scores

  // Pick the highest-IoU mask and extract its single H×W plane.
  let best = 0;
  for (let i = 1; i < numMasks; i++) if (scores[i] > scores[best]) best = i;
  const plane = new Uint8Array(H * W);
  const src = maskTensor.data;
  const off = best * H * W;
  let area = 0;
  for (let i = 0; i < H * W; i++) {
    plane[i] = src[off + i] ? 1 : 0;
    if (plane[i]) area++;
  }

  // Bake the coloured overlay off the main thread; the page blits it with one drawImage(). The raw
  // plane is still transferred so the page can build the transparent-PNG cut-out on export.
  const overlay = overlayBitmap(plane, W, H);

  post({
    type: "result",
    id,
    width: W,
    height: H,
    mask: plane,
    overlay,
    score: scores[best],
    allScores: scores,
    bestIndex: best,
    numMasks,
    area,
    point: points[0],
    ms: Math.round(performance.now() - t0),
    device,
  }, [plane.buffer, overlay]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "embed") await embed(e.data.id, e.data.image);
    else if (type === "segment") await segment(e.data.id, e.data.points);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
