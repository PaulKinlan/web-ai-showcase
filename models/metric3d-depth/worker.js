// Metric3D worker — runs ALL depth inference AND the dense colour-map composite off the main thread so
// the control UI never freezes (invariant 15). The per-pixel colourise is done here on an OffscreenCanvas
// and transferred back as an ImageBitmap; the main thread only does a single drawImage.
//
// Model: onnx-community/metric3d-vit-small (task: depth-estimation, model_type metric3d). Metric3D is a
// METRIC (absolute-scale) depth model: unlike Depth Anything / DPT, which emit a smooth RELATIVE map in
// arbitrary units re-normalized per image, Metric3D predicts depth in METRES against a canonical camera.
// predicted_depth therefore carries real distances (e.g. ~2.4–7.4 m for an indoor scene), not just a
// nearer/farther ordering.
//
// Backend: runs on WebGPU (fp16, ~72 MB) when a real adapter exists, else WASM (fp32, ~143 MB) — it is
// small enough to run on CPU, so this demo works on devices without WebGPU. Output is the real
// transformers.js pipeline result: { depth: RawImage (0–255 normalized), predicted_depth: Tensor (raw
// metric-scale metres) }.
//
// Metric caveat (surfaced in the UI): the ONNX export assumes the model's CANONICAL camera intrinsics,
// so absolute metres are exact for that reference focal length and a calibrated estimate for an arbitrary
// photo. The RELATIVE metric structure across the scene is the robust signal.

import { loadPipeline, pickDevice } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/metric3d-vit-small";

let pipe = null;
let device = "wasm";
let dtype = "fp32";
let RawImage = null;
let lastGray = null; // retained for off-main-thread re-colourise on colour-map change
let lastW = 0, lastH = 0;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Perceptual colour ramps (identical control points to models/depth-anything/depth.js so the two depth
// demos stay visually consistent). Interpolated in sRGB.
const MAPS = {
  turbo: [
    [48, 18, 59],
    [65, 69, 171],
    [57, 118, 209],
    [32, 163, 181],
    [48, 196, 120],
    [140, 208, 52],
    [216, 182, 29],
    [238, 116, 32],
    [165, 20, 24],
  ],
  viridis: [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
  magma: [[0, 0, 4], [81, 18, 124], [183, 55, 121], [252, 137, 97], [252, 253, 191]],
  gray: [[0, 0, 0], [255, 255, 255]],
};
function sampleMap(name, t) {
  const stops = MAPS[name] ?? MAPS.turbo;
  const x = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.floor(x), f = x - i;
  const a = stops[i], b = stops[Math.min(stops.length - 1, i + 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// Colourise a normalized 0–255 depth buffer into an ImageBitmap, entirely off the main thread.
function colourise(gray, w, h, cmap) {
  const oc = new OffscreenCanvas(w, h);
  const ctx = oc.getContext("2d");
  const img = ctx.createImageData(w, h);
  const px = img.data;
  for (let i = 0; i < gray.length; i++) {
    const [r, g, b] = sampleMap(cmap, gray[i] / 255);
    const o = i * 4;
    px[o] = r;
    px[o + 1] = g;
    px[o + 2] = b;
    px[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return oc.transferToImageBitmap();
}

async function realDevice() {
  return await pickDevice("webgpu"); // "webgpu" only when a real adapter exists, else "wasm"
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5");
  RawImage = mod.RawImage;
  device = await realDevice();
  // fp16 is the compact WebGPU build; WASM needs fp32 (fp16 CPU kernels are unreliable).
  dtype = device === "webgpu" ? "fp16" : "fp32";
  const loaded = await loadPipeline({
    task: "depth-estimation",
    model: MODEL_ID,
    backend: device,
    dtype,
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device, dtype });
}

// 32-bin histogram over the normalized (0–255) depth so "See inside" can show the distribution.
function histogram(data, bins = 32) {
  const h = new Array(bins).fill(0);
  const scale = bins / 256;
  for (let i = 0; i < data.length; i++) h[Math.min(bins - 1, (data[i] * scale) | 0)]++;
  return h;
}

async function run(id, imageURL, cmap) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const out = await pipe(imageURL);

  // `depth` is a single-channel RawImage, min–max normalized to 0–255 at the original resolution.
  const depthImg = out.depth;
  const w = depthImg.width, h = depthImg.height;
  const ch = depthImg.channels ?? (depthImg.data.length / (w * h)) | 0;
  let gray;
  if (ch === 1) {
    gray = depthImg.data instanceof Uint8Array ? depthImg.data : Uint8Array.from(depthImg.data);
  } else {
    gray = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) gray[i] = depthImg.data[i * ch];
  }
  lastGray = gray;
  lastW = w;
  lastH = h;

  // Raw predicted depth tensor — real METRES straight from the model. For Metric3D these carry absolute
  // scale (against the canonical camera). We keep the full field so "measure distances" can read metres.
  let metricMin = null, metricMax = null, metricMean = null, rawDims = null;
  let metricField = null;
  const pd = out.predicted_depth;
  if (pd && pd.data) {
    rawDims = Array.from(pd.dims ?? []);
    let mn = Infinity, mx = -Infinity, sum = 0;
    const d = pd.data;
    for (let i = 0; i < d.length; i++) {
      const v = d[i];
      if (v < mn) mn = v;
      if (v > mx) mx = v;
      sum += v;
    }
    metricMin = mn;
    metricMax = mx;
    metricMean = sum / d.length;
    metricField = d instanceof Float32Array ? new Float32Array(d) : Float32Array.from(d);
  }

  const hist = histogram(gray);
  const bitmap = colourise(gray, w, h, cmap || "turbo");
  const ms = Math.round(performance.now() - t0);
  const grayCopy = gray.slice(0).buffer; // a copy for the main thread; worker keeps lastGray for recolour
  const transfer = [bitmap, grayCopy];
  const msg = {
    type: "result",
    id,
    width: w,
    height: h,
    bitmap,
    depth: grayCopy,
    hist,
    metricMin,
    metricMax,
    metricMean,
    rawDims,
    origW: image.width,
    origH: image.height,
    ms,
    device,
    dtype,
  };
  if (metricField) {
    msg.metricField = metricField.buffer;
    msg.metricW = rawDims.length >= 2 ? rawDims[rawDims.length - 1] : w;
    msg.metricH = rawDims.length >= 2 ? rawDims[rawDims.length - 2] : h;
    transfer.push(metricField.buffer);
  }
  post(msg, transfer);
}

// Re-colourise the last depth map with a new colour map — still off the main thread.
function recolor(id, cmap) {
  if (!lastGray) {
    post({ type: "error", id, message: "No depth map to re-colour yet." });
    return;
  }
  const bitmap = colourise(lastGray, lastW, lastH, cmap || "turbo");
  post({ type: "recolored", id, bitmap }, [bitmap]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.cmap);
    else if (type === "recolor") recolor(e.data.id, e.data.cmap);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
