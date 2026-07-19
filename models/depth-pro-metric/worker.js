// DepthPro (Apple) worker — runs ALL depth inference off the main thread so the control UI stays
// responsive. DepthPro is DISTINCT from Depth Anything / DPT: it produces SHARP, boundary-accurate,
// METRIC-SCALE monocular depth. Where Depth Anything emits a smooth relative map (arbitrary units),
// DepthPro is trained for absolute-scale depth at high resolution (native 1536²), so its
// predicted_depth field carries a consistent scale and its object boundaries stay crisp.
//
// Model: onnx-community/DepthPro-ONNX (task: depth-estimation, model_type depth_pro), WebGPU with the
// q4f16 build (~572 MB — a large model). This is a WebGPU-class model: the page gates on a real
// adapter (requiresWebGPU) and shows an honest needs-WebGPU state where there isn't one — it never
// fakes output and never silently downloads on a device that can't run it. Uses the SHARED loader from
// lib/webai.js; output is the real transformers.js pipeline result — { depth: RawImage (normalized
// 0–255 grayscale), predicted_depth: Tensor (raw metric-scale values) }.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "webgpu";
let dtype = "q4f16";
let RawImage = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Honest device pick: navigator.gpu existing is NOT enough. Request a real adapter; if there isn't a
// usable one, report unsupported rather than pretending. (The page's loader already gates on this via
// requiresWebGPU, so in practice load() is only invoked when an adapter exists.)
async function realDevice() {
  if ("gpu" in navigator) {
    try {
      if (await navigator.gpu.requestAdapter()) return "webgpu";
    } catch { /* fall through */ }
  }
  return null;
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  const dev = await realDevice();
  if (!dev) {
    throw new Error(
      "DepthPro needs WebGPU — no GPU adapter is available here. Open in a WebGPU-capable browser (chrome://gpu should show WebGPU enabled).",
    );
  }
  device = dev;
  dtype = "q4f16"; // fp16-based 4-bit weights — the browser-feasible DepthPro build (~572 MB)
  const loaded = await loadPipeline({
    task: "depth-estimation",
    model: "onnx-community/DepthPro-ONNX",
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

// Boundary sharpness: mean Sobel gradient magnitude over the normalized depth, plus the fraction of
// "edge" pixels (gradient above a threshold). DepthPro's crisp boundaries show up as a high edge
// fraction; a smooth relative map (Depth Anything) has fewer, softer depth edges. This is computed
// from the REAL model output, not asserted.
function boundarySharpness(gray, w, h) {
  let sum = 0, edges = 0, n = 0;
  const at = (x, y) => gray[y * w + x];
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const gx = (at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)) -
        (at(x - 1, y - 1) + 2 * at(x - 1, y) + at(x - 1, y + 1));
      const gy = (at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)) -
        (at(x - 1, y - 1) + 2 * at(x, y - 1) + at(x + 1, y - 1));
      const mag = Math.sqrt(gx * gx + gy * gy);
      sum += mag;
      if (mag > 48) edges++; // ~19% of the 0–255 range: a genuine depth discontinuity
      n++;
    }
  }
  return { meanGradient: n ? sum / n : 0, edgeFraction: n ? edges / n : 0 };
}

async function run(id, imageURL) {
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

  // Raw predicted depth tensor — real numbers straight from the model (metric-scale), for the range
  // readout + dims. For DepthPro these values carry a consistent absolute scale.
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
  const sharp = boundarySharpness(gray, w, h);
  const ms = Math.round(performance.now() - t0);
  const buf = gray.buffer.slice(0);
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
      meanGradient: sharp.meanGradient,
      edgeFraction: sharp.edgeFraction,
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
