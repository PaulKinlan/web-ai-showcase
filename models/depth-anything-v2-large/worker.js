// Depth Anything V2 LARGE worker — runs ALL depth inference off the main thread so the control UI
// stays responsive. One forward pass gives everything the pages need: the normalized per-pixel depth
// (for colourising + parallax) and the raw predicted-depth range/dims (for "See inside").
//
// Model: onnx-community/depth-anything-v2-large (task: depth-estimation). This is the ViT-Large V2
// checkpoint — a much bigger encoder than the ViT-Small used by the /depth-anything/ demo, so it
// resolves thinner structures (wires, railings, hair, foliage) and cleaner depth edges, at the cost of
// a larger download and slower inference. WebGPU fp16 with a WASM q8 fallback. Uses the SHARED loader
// from lib/webai.js and the real transformers.js pipeline output —
// { depth: RawImage (normalized 0–255 grayscale), predicted_depth: Tensor (raw relative depth) }.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";
import { colorizeDepth } from "./colormap.js";

const MODEL_ID = "onnx-community/depth-anything-v2-large";

let pipe = null;
let device = "wasm";
let dtype = "q8";
let RawImage = null;

// Retained normalized depth of the most recent run so a colour-map change (a control) can re-colourise
// off the main thread WITHOUT re-running inference. Detached buffers are never reused.
let lastGray = null, lastW = 0, lastH = 0;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Dense-output composite, off the main thread: colourise a normalized depth buffer and hand the main
// thread a finished ImageBitmap (transferred, zero-copy). The page only does drawImage — no per-pixel
// loop, so the ~40ms @1080p colourise never lands on the main thread / spikes INP.
function colourBitmap(gray, w, h, mapName) {
  const rgba = colorizeDepth(gray, w, h, mapName);
  const canvas = new OffscreenCanvas(w, h);
  canvas.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  return canvas.transferToImageBitmap();
}

// Honest device pick: navigator.gpu existing is NOT enough (headless exposes it with no adapter).
// Actually request an adapter; fall back to WASM when there isn't a usable one.
async function realDevice() {
  if ("gpu" in navigator) {
    try {
      if (await navigator.gpu.requestAdapter()) return "webgpu";
    } catch { /* fall through to wasm */ }
  }
  return "wasm";
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  // fp16 on the WebGPU path; the WASM fallback uses the q8 build (model_quantized.onnx) so it stays
  // downloadable and actually runnable on the CPU — an honest fallback, still real inference.
  device = await realDevice();
  dtype = device === "webgpu" ? "fp16" : "q8";
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

async function run(id, imageURL, mapName) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const out = await pipe(imageURL);

  // `depth` is a single-channel RawImage, min–max normalized to 0–255 at the original resolution.
  const depthImg = out.depth;
  const w = depthImg.width, h = depthImg.height;
  // Some builds return RGB(A); collapse to one channel per pixel.
  const ch = depthImg.channels ?? (depthImg.data.length / (w * h)) | 0;
  let gray;
  if (ch === 1) {
    gray = depthImg.data instanceof Uint8Array ? depthImg.data : Uint8Array.from(depthImg.data);
  } else {
    gray = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) gray[i] = depthImg.data[i * ch];
  }

  // Raw predicted depth tensor — real numbers straight from the model, for the range readout + dims.
  let rawMin = null, rawMax = null, rawDims = null, rawMean = null;
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
    rawMin = mn;
    rawMax = mx;
    rawMean = sum / d.length;
  }

  const hist = histogram(gray);
  // Retain this run's depth (worker-owned, never transferred) so recolour needs no re-inference.
  lastGray = gray;
  lastW = w;
  lastH = h;
  // Colourise off the main thread and transfer a finished ImageBitmap back (the dense composite).
  const colorBitmap = colourBitmap(gray, w, h, mapName || "turbo");
  const ms = Math.round(performance.now() - t0);
  const buf = gray.buffer.slice(0); // own copy so we can transfer without detaching library memory
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      depth: buf,
      colorBitmap,
      mapName: mapName || "turbo",
      hist,
      rawMin,
      rawMax,
      rawMean,
      rawDims,
      origW: image.width,
      origH: image.height,
      ms,
      device,
      dtype,
    },
    [buf, colorBitmap],
  );
}

// Re-colourise the most recent depth map with a new colour map — off the main thread, no re-inference.
function recolor(id, mapName) {
  if (!lastGray) {
    post({ type: "error", id, message: "No depth map to recolour yet" });
    return;
  }
  const colorBitmap = colourBitmap(lastGray, lastW, lastH, mapName || "turbo");
  post(
    { type: "recolor", id, colorBitmap, mapName: mapName || "turbo", width: lastW, height: lastH },
    [colorBitmap],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.colormap);
    else if (type === "recolor") recolor(e.data.id, e.data.colormap);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
