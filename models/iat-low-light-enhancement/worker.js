// IAT low-light-enhancement worker — ALL inference AND the dense output composite off the main thread
// via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no `iat` model class and there is no
// `low-light-enhancement` pipeline task. The ONNX export is a clean image→image graph, so we run it
// directly with onnxruntime-web and hand-write the pre/post a pipeline would own: (1) resize the dark
// input (long side ≤ maxSide) and pack it as [1,3,H,W] float32 in [0,1] (NO ImageNet normalisation —
// IAT works on plain RGB in [0,1]), and (2) clamp the enhanced tensor back to an RGBA image. This is
// the isolated per-worker ORT-web escape hatch (like models/nafnet-image-deblurring/worker.js) —
// onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: Pezhgorski/IAT-ONNX (onnx/iat_lol_v1.onnx + iat_lol_v1.onnx.data, apache-2.0, ~0.42 MB). IAT
// (Illumination Adaptive Transformer, Cui et al., BMVC 2022) — a ~90K-param network that corrects
// under-exposed / low-light photos. Input "input" [1,3,H,W] float32 in [0,1] (dynamic H,W). THREE
// outputs: "mul" (per-pixel multiplicative gain), "add" (additive offset), and "enhanced" (index 2, the
// corrected image, [0,1]) — enhanced ≈ input·mul + add. We display `enhanced` and visualise `mul` as
// the illumination-gain map (WHERE the model decided to brighten). The weights live in an external-data
// sidecar (.onnx.data); ORT-Web loads it via the session `externalData` option (verified working).
// Everything stays on-device: the image never leaves the tab.
//
// Variant choice: we use the LOL-v1 checkpoint. Measured in headless Chrome, the repo's iat_lol_v2
// export is DEGENERATE — on a real near-black input it emits a near-constant grey (luminance std ~0.009,
// range 0.19–0.37), no dynamic range. iat_lol_v1 is the faithful one: same near-black input →
// mean 0.054→0.375 with contrast PRESERVED (std 0.044→0.195) and full 0–0.76 range, a natural lit scene.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "Pezhgorski/IAT-ONNX";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/iat_lol_v1.onnx`;
const DATA_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/iat_lol_v1.onnx.data`;
const DATA_NAME = "iat_lol_v1.onnx.data"; // must match the external-data reference inside the ONNX
const CACHE_NAME = "iat-lowlight-onnx-cache";

let ort = null;
let session = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch a file THROUGH Cache Storage so lib/model-cache.js (which scans caches for "/Pezhgorski/IAT-ONNX/")
// sees it → auto-init on a returning visit, honest Download on first visit, and the per-model "clear
// cache" control all work. Streams download progress across both files.
async function fetchCached(url, cache, onChunk) {
  let resp = await cache.match(url);
  if (resp) return new Uint8Array(await resp.arrayBuffer());
  const net = await fetch(url);
  if (!net.ok || !net.body) throw new Error(`fetch failed (${net.status}) for ${url}`);
  const total = Number(net.headers.get("content-length")) || 0;
  const reader = net.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onChunk?.(received, total);
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  await cache.put(
    url,
    new Response(buf, {
      headers: { "content-length": String(received), "content-type": "application/octet-stream" },
    }),
  );
  return buf;
}

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  const cache = await caches.open(CACHE_NAME);
  // The two files are ~66 KB + ~347 KB; report combined progress.
  const dataBytes = await fetchCached(DATA_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 90 } });
  });
  const modelBytes = await fetchCached(MODEL_URL, cache, () => {
    post({ type: "progress", p: { status: "progress", progress: 95 } });
  });
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    externalData: [{ path: DATA_NAME, data: dataBytes }],
  });
  post({ type: "ready", device });
}

// Mean relative luminance (Rec.709) of a planar [3,H,W] float buffer — the standard brightness measure.
function luma(plane, n) {
  let s = 0;
  for (let i = 0; i < n; i++) {
    s += 0.2126 * plane[i] + 0.7152 * plane[n + i] + 0.0722 * plane[2 * n + i];
  }
  return s / n;
}
// RMS contrast of luminance — std-dev of per-pixel luma. A dark photo is flat; enhancement lifts it.
function contrast(plane, n) {
  let s = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const y = 0.2126 * plane[i] + 0.7152 * plane[n + i] + 0.0722 * plane[2 * n + i];
    s += y;
    ss += y * y;
  }
  const m = s / n;
  return Math.sqrt(Math.max(0, ss / n - m * m));
}

async function enhance(id, bitmap, opts) {
  await ensureLoaded();
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 512;
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  // IAT is dynamic-shaped but the transformer needs a multiple of 4 to be safe; round to 8.
  const w = Math.max(8, Math.round(iw * scale / 8) * 8);
  const h = Math.max(8, Math.round(ih * scale / 8) * 8);
  const t0 = performance.now();

  const c = new OffscreenCanvas(w, h);
  const cctx = c.getContext("2d", { willReadFrequently: true });
  cctx.drawImage(bitmap, 0, 0, iw, ih, 0, 0, w, h);
  bitmap.close?.();
  const src = cctx.getImageData(0, 0, w, h).data;
  const N = w * h;
  const feed = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    feed[i] = src[i * 4] / 255;
    feed[N + i] = src[i * 4 + 1] / 255;
    feed[2 * N + i] = src[i * 4 + 2] / 255;
  }
  const inLuma = luma(feed, N);
  const inContrast = contrast(feed, N);

  const results = await session.run({ input: new ort.Tensor("float32", feed, [1, 3, h, w]) });
  const enhanced = results.enhanced.data; // [1,3,h,w] in [0,1]
  const gain = results.mul?.data || null; // per-pixel multiplicative illumination gain
  const infMs = Math.round(performance.now() - t0);
  const outLuma = luma(enhanced, N);
  const outContrast = contrast(enhanced, N);

  // Enhanced RGBA + a gain heatmap (illumination map — how much each region was brightened).
  const rgba = new Uint8ClampedArray(N * 4);
  const gainMap = new Uint8ClampedArray(N * 4);
  let gMin = Infinity, gMax = -Infinity;
  if (gain) {
    for (let i = 0; i < N; i++) {
      const g = (gain[i] + gain[N + i] + gain[2 * N + i]) / 3;
      if (g < gMin) gMin = g;
      if (g > gMax) gMax = g;
    }
  }
  const gRange = gMax - gMin || 1;
  for (let i = 0; i < N; i++) {
    rgba[i * 4] = Math.max(0, Math.min(1, enhanced[i])) * 255;
    rgba[i * 4 + 1] = Math.max(0, Math.min(1, enhanced[N + i])) * 255;
    rgba[i * 4 + 2] = Math.max(0, Math.min(1, enhanced[2 * N + i])) * 255;
    rgba[i * 4 + 3] = 255;
    if (gain) {
      const g = (gain[i] + gain[N + i] + gain[2 * N + i]) / 3;
      const t = (g - gMin) / gRange; // 0..1 normalised gain
      // indigo→amber ramp: low gain = dark indigo, high gain = warm amber (brightened most)
      gainMap[i * 4] = 30 + t * 225;
      gainMap[i * 4 + 1] = 20 + t * 180;
      gainMap[i * 4 + 2] = 90 - t * 60;
      gainMap[i * 4 + 3] = 255;
    }
  }

  const outC = new OffscreenCanvas(w, h);
  outC.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  const enhancedBmp = outC.transferToImageBitmap();
  const gainBmp = gain ? await createImageBitmap(new ImageData(gainMap, w, h)) : null;

  const ms = Math.round(performance.now() - t0);
  const transfer = [enhancedBmp];
  if (gainBmp) transfer.push(gainBmp);
  post({
    type: "result",
    id,
    enhancedBmp,
    gainBmp,
    w,
    h,
    imgW: iw,
    imgH: ih,
    inLuma,
    outLuma,
    inContrast,
    outContrast,
    gMin: gain ? gMin : null,
    gMax: gain ? gMax : null,
    ms,
    infMs,
    device,
  }, transfer);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await enhance(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
