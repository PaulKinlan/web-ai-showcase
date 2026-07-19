// CLIPSeg worker — text-prompted image segmentation OFF the main thread, so the control UI stays
// responsive while the model runs. Model: Xenova/clipseg-rd64-refined (CLIPSegForImageSegmentation),
// WASM backend, q8 (model_quantized.onnx). ~140 MB, cached by transformers.js after first load.
//
// How CLIPSeg actually works (the real API — not invented; verified against transformers.js 3.7.5):
//   • AutoProcessor for this repo exposes an IMAGE processor only (no tokenizer bundled), so we load
//     the text tokenizer separately with AutoTokenizer.
//   • Tokenise the N phrases (padded) → input_ids / attention_mask  ([N, L]).
//   • Run the image processor once → pixel_values [1, 3, 352, 352]; TILE it to [N, 3, 352, 352] so
//     every phrase is scored against the same image in one batched forward pass.
//   • model({ input_ids, attention_mask, pixel_values }) → logits [N, 352, 352] — a RAW per-pixel
//     score map per phrase at the model's 352×352 working resolution.
// We return the raw logits (Float32) per prompt; sigmoid + thresholding happen on the page so the
// "see inside" heatmap and the threshold slider stay live without re-running the model.

import { pickDevice, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/clipseg-rd64-refined";
let tokenizer = null;
let imageProcessor = null;
let model = null;
let RawImageRef = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// ── Off-main-thread compositing (invariant 15) ───────────────────────────────────────────────────
// The page used to build the mask overlay + inferno heatmap RGBA (per-pixel sigmoid/threshold/colour
// loops) AND the photo-resolution cut-out (getImageData→per-pixel alpha knockout, ~40ms @1080p) on the
// MAIN thread on every threshold/opacity slider input. We now cache the raw logits here and build every
// coloured layer into an OffscreenCanvas off the main thread, transferring an ImageBitmap back; the page
// only drawImage()s (and cuts out via a GPU destination-in composite — no per-pixel loop). Colours and
// thresholding are identical to the former clipseg.js painters.
let cachedMaps = null; // [{prompt, data:Float32Array}] at cachedW×cachedH — kept for slider re-composite
let cachedW = 0, cachedH = 0;

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Golden-angle phrase colour — identical to clipseg.js colorForIndex/hslToRgb.
function hslToRgb(h, s, l) {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const hk = (t) => {
    let x = t;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  };
  return [
    Math.round(hk(h + 1 / 3) * 255),
    Math.round(hk(h) * 255),
    Math.round(hk(h - 1 / 3) * 255),
  ];
}
function colorForIndex(i) {
  const h = (i * 137.508 + 15) % 360;
  return hslToRgb(h / 360, i % 2 ? 0.72 : 0.62, i % 3 === 0 ? 0.55 : 0.5);
}
// Inferno-style ramp — identical to clipseg.js RAMP/probColor.
const RAMP = [
  [0, 0, 4],
  [40, 11, 84],
  [101, 21, 110],
  [159, 42, 99],
  [212, 72, 66],
  [245, 125, 21],
  [250, 193, 39],
  [252, 255, 164],
];
function probColor(t) {
  const x = Math.min(1, Math.max(0, t)) * (RAMP.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = RAMP[i], b = RAMP[Math.min(RAMP.length - 1, i + 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function bitmapFrom(rgba, w, h) {
  const oc = new OffscreenCanvas(w, h);
  oc.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  return oc.transferToImageBitmap();
}

// Combine one or more thresholded, coloured mask layers (source-over, matching the old paintMasks).
// items: [{mapIndex, colorIndex}] — colorIndex lets basics/multi-model reuse colour 0 like paintMasks([m]).
function buildOverlay(items, threshold, opacity) {
  const w = cachedW, h = cachedH;
  const oc = new OffscreenCanvas(w, h);
  const ctx = oc.getContext("2d");
  const a = Math.round(opacity * 255);
  for (const { mapIndex, colorIndex } of items) {
    const map = cachedMaps[mapIndex].data;
    const [r, g, b] = colorForIndex(colorIndex);
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      if (sigmoid(map[i]) >= threshold) {
        const o = i * 4;
        rgba[o] = r;
        rgba[o + 1] = g;
        rgba[o + 2] = b;
        rgba[o + 3] = a;
      }
    }
    const tmp = new OffscreenCanvas(w, h);
    tmp.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
    ctx.drawImage(tmp, 0, 0);
  }
  return oc.transferToImageBitmap();
}

// Inferno heatmap of sigmoid(logits) — the "see inside" probability field.
function buildHeat(mapIndex) {
  const w = cachedW, h = cachedH, map = cachedMaps[mapIndex].data;
  const rgba = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    const [r, g, b] = probColor(sigmoid(map[i]));
    const o = i * 4;
    rgba[o] = r;
    rgba[o + 1] = g;
    rgba[o + 2] = b;
    rgba[o + 3] = 255;
  }
  return bitmapFrom(rgba, w, h);
}

// Binary alpha mask (255 above threshold, else transparent) — the page uses it as a GPU destination-in
// stencil to cut the object out of the photo without any main-thread per-pixel loop.
function buildMaskAlpha(mapIndex, threshold) {
  const w = cachedW, h = cachedH, map = cachedMaps[mapIndex].data;
  const rgba = new Uint8ClampedArray(w * h * 4);
  let above = 0;
  for (let i = 0; i < w * h; i++) {
    if (sigmoid(map[i]) >= threshold) {
      rgba[i * 4 + 3] = 255;
      above++;
    }
  }
  return { bitmap: bitmapFrom(rgba, w, h), above };
}

// Per-image normalised render for the "wild" page: sigmoid → stretch to the map's own min–max, then
// build a relative inferno heatmap + a relative-threshold mask overlay (colour 0, alpha 140).
function buildNorm(mapIndex, threshold) {
  const w = cachedW, h = cachedH, map = cachedMaps[mapIndex].data;
  const p = new Float32Array(w * h);
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < w * h; i++) {
    const v = sigmoid(map[i]);
    p[i] = v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const span = mx - mn || 1;
  const heat = new Uint8ClampedArray(w * h * 4);
  const over = new Uint8ClampedArray(w * h * 4);
  const [mr, mg, mb] = colorForIndex(0);
  let above = 0;
  for (let i = 0; i < w * h; i++) {
    const n = (p[i] - mn) / span;
    const [r, g, b] = probColor(n);
    const o = i * 4;
    heat[o] = r;
    heat[o + 1] = g;
    heat[o + 2] = b;
    heat[o + 3] = 255;
    if (n >= threshold) {
      over[o] = mr;
      over[o + 1] = mg;
      over[o + 2] = mb;
      over[o + 3] = 140;
      above++;
    }
  }
  return { heat: bitmapFrom(heat, w, h), overlay: bitmapFrom(over, w, h), above };
}

// Dispatch a composite request against the cached logits; transfer the ImageBitmap(s) back.
function composite(id, op, args) {
  if (!cachedMaps) throw new Error("No segmentation cached yet — run a prompt first.");
  if (op === "overlay") {
    const bitmap = buildOverlay(args.items, args.threshold, args.opacity);
    post({ type: "composite", id, mapW: cachedW, mapH: cachedH, bitmap }, [bitmap]);
  } else if (op === "heatmap") {
    const bitmap = buildHeat(args.mapIndex);
    post({ type: "composite", id, mapW: cachedW, mapH: cachedH, bitmap }, [bitmap]);
  } else if (op === "maskAlpha") {
    const { bitmap, above } = buildMaskAlpha(args.mapIndex, args.threshold);
    post({ type: "composite", id, mapW: cachedW, mapH: cachedH, bitmap, above }, [bitmap]);
  } else if (op === "norm") {
    const { heat, overlay, above } = buildNorm(args.mapIndex, args.threshold);
    post({ type: "composite", id, mapW: cachedW, mapH: cachedH, heat, overlay, above }, [
      heat,
      overlay,
    ]);
  } else {
    throw new Error("Unknown composite op: " + op);
  }
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoProcessor, CLIPSegForImageSegmentation, RawImage, env } = await import(
    TRANSFORMERS_URL
  );
  env.allowLocalModels = false;
  RawImageRef = RawImage;
  // CLIPSeg runs happily on WASM; q8 quantised. Prefer the honest device (wasm here) — no WebGPU need.
  device = await pickDevice("wasm");
  const onProgress = (p) => post({ type: "progress", p });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress });
  const processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: onProgress,
  });
  imageProcessor = processor.image_processor || processor;
  model = await CLIPSegForImageSegmentation.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device,
    progress_callback: onProgress,
  });
  post({ type: "ready", device });
}

// Tile pixel_values [1,C,H,W] → [n,C,H,W] (batch the same image against every phrase).
function tile(pv, n) {
  if (pv.dims[0] === n) return pv;
  const [, C, H, W] = pv.dims;
  const per = C * H * W;
  const data = new pv.data.constructor(per * n);
  for (let i = 0; i < n; i++) data.set(pv.data, i * per);
  return new pv.constructor(pv.type, data, [n, C, H, W]);
}

async function run(id, imageURL, prompts) {
  await ensureLoaded();
  if (!Array.isArray(prompts) || prompts.length === 0) throw new Error("no prompts given");
  const t0 = performance.now();

  const image = (await RawImageRef.read(imageURL)).rgb();
  const textInputs = tokenizer(prompts, { padding: true, truncation: true });
  const imgInputs = await imageProcessor(image);
  const pixel_values = tile(imgInputs.pixel_values, prompts.length);

  const out = await model({
    input_ids: textInputs.input_ids,
    attention_mask: textInputs.attention_mask,
    pixel_values,
  });

  const logits = out.logits;
  const dims = logits.dims;
  const mapH = dims[dims.length - 2];
  const mapW = dims[dims.length - 1];
  const per = mapW * mapH;
  const src = logits.data; // Float32Array length prompts*per (or per when a single prompt is squeezed)
  const n = Math.max(1, Math.round(src.length / per));

  const maps = [];
  const transfer = [];
  const cache = [];
  for (let p = 0; p < prompts.length; p++) {
    const slice = new Float32Array(per);
    const base = (p % n) * per;
    slice.set(src.subarray(base, base + per));
    maps.push({ prompt: prompts[p], data: slice.buffer });
    transfer.push(slice.buffer);
    // Keep a private copy for off-main slider re-composite (the page's copy is transferred/detached).
    cache.push({ prompt: prompts[p], data: new Float32Array(slice) });
  }
  cachedMaps = cache;
  cachedW = mapW;
  cachedH = mapH;

  post({
    type: "result",
    id,
    mapW,
    mapH,
    imageW: image.width,
    imageH: image.height,
    maps,
    ms: Math.round(performance.now() - t0),
    device,
  }, transfer);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.prompts);
    else if (type === "composite") composite(e.data.id, e.data.op, e.data.args);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
