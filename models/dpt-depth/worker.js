// DPT (Intel dpt-hybrid-midas) depth worker — runs ALL depth inference off the main thread so the
// control UI stays responsive. One forward pass gives everything the pages need: the normalized
// per-pixel depth (for colourising + parallax) and the raw predicted-depth range/dims (for "See
// inside").
//
// Model: Xenova/dpt-hybrid-midas (task: depth-estimation) — the classic Dense Prediction Transformer:
// a ViT-hybrid encoder (ResNet-50 stem + ViT) with a MiDaS-trained DPT decode head. It predicts
// RELATIVE INVERSE depth (MiDaS convention: larger = nearer), which transformers.js min–max normalises
// to a 0–255 grayscale `depth` image (brighter = nearer) exactly like Depth Anything — so the same
// display pipeline works, but the numbers come from a different architecture + training set.
// WebGPU fp16 with a WASM q8 fallback. Uses the SHARED loader from lib/webai.js.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let dtype = "q8";
let RawImage = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
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
  // runnable on the CPU — an honest fallback, still real inference.
  device = await realDevice();
  dtype = device === "webgpu" ? "fp16" : "q8";
  const loaded = await loadPipeline({
    task: "depth-estimation",
    model: "Xenova/dpt-hybrid-midas",
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

async function run(id, imageURL) {
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
  const ms = Math.round(performance.now() - t0);
  const buf = gray.buffer.slice(0); // own copy so we can transfer without detaching library memory
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      depth: buf,
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
    [buf],
  );
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
