// MicroDehazeNet image-dehazing worker — ALL inference AND the dense output composite off the main
// thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no dehazing model class and there is
// no `image-dehazing`/`image-restoration` pipeline task. The ONNX export is a clean image→image graph,
// so we run it directly with onnxruntime-web and hand-write the pre/post a pipeline would own: (1) resize
// the hazy input (long side ≤ maxSide, rounded to a multiple of 8 for the U-Net's downsampling) and pack
// it as [1,3,H,W] float32 in [0,1] — plain ToTensor, NO ImageNet normalisation (matches the repo's
// inference.py) — and (2) clamp the dehazed tensor back to an RGBA image. This is the isolated per-worker
// ORT-web escape hatch (like models/iat-low-light-enhancement/worker.js) — onnxruntime-web is pinned HERE
// only, never in shared libs.
//
// Model: Vive-k-kumar/micro-dehaze-net (micro_dehaze_net.onnx + micro_dehaze_net.onnx.data, MIT,
// ~4 MB, 996K params, base_channels 24). MicroDehazeNet — a tiny convolutional single-image dehazing
// network trained on Haze4K (best PSNR 26.78 / SSIM 0.956). Input "hazy_input" [1,3,H,W] float32 in
// [0,1] (dynamic H,W). Output "dehazed_output" [1,3,H,W] float32 in ~[0,1] (the haze-removed image).
// The weights live in an external-data sidecar (.onnx.data); ORT-Web loads it via the session
// `externalData` option (verified working). Everything stays on-device: the image never leaves the tab.
//
// Runnability was proven FIRST in headless Chrome: on a hazy street scene the RMS-contrast of the
// luminance rose 0.117 → 0.274 (2.34×), landing on the un-hazed clean reference (0.261) — real haze
// removal / clarity restoration, not a flat global curve. Inference ~0.9 s @512² on single-thread WASM.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "Vive-k-kumar/micro-dehaze-net";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/micro_dehaze_net.onnx`;
const DATA_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/micro_dehaze_net.onnx.data`;
const DATA_NAME = "micro_dehaze_net.onnx.data"; // must match the external-data reference inside the ONNX
const CACHE_NAME = "microdehaze-onnx-cache";

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch a file THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/Vive-k-kumar/micro-dehaze-net/") sees it → auto-init on a returning visit, honest Download on first
// visit, and the per-model "clear cache" control all work. Streams download progress.
async function fetchCached(url, cache, onChunk) {
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
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
  // The two files are ~0.2 MB + ~3.8 MB; report combined progress (data first, weighted 90%).
  const dataBytes = await fetchCached(DATA_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 90 } });
  });
  const modelBytes = await fetchCached(MODEL_URL, cache, () => {
    post({ type: "progress", p: { status: "progress", progress: 96 } });
  });
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    externalData: [{ path: DATA_NAME, data: dataBytes }],
  });
  inputName = session.inputNames[0]; // "hazy_input"
  outputName = session.outputNames[0]; // "dehazed_output"
  post({ type: "ready", device });
}

// RMS contrast of relative luminance (Rec.709) over a planar [3,N] float buffer — the standard clarity
// measure. Haze flattens contrast; dehazing restores it.
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

async function dehaze(id, bitmap, opts) {
  await ensureLoaded();
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 512;
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  // Fully-convolutional but round to a multiple of 8 so the U-Net's poolings line up.
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
  const inContrast = contrast(feed, N);

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, h, w]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // [1,3,h,w] ~[0,1]
  const infMs = Math.round(performance.now() - t0);
  const outContrast = contrast(out, N);

  // Dehazed RGBA + a "haze removed" map: per-pixel change |out-in|, amplified. It lights up in the
  // regions the model cleared — typically the distant/low-transmission areas the haze washed out.
  const rgba = new Uint8ClampedArray(N * 4);
  const hazeMap = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    const r = out[i], g = out[N + i], b = out[2 * N + i];
    rgba[i * 4] = Math.max(0, Math.min(1, r)) * 255;
    rgba[i * 4 + 1] = Math.max(0, Math.min(1, g)) * 255;
    rgba[i * 4 + 2] = Math.max(0, Math.min(1, b)) * 255;
    rgba[i * 4 + 3] = 255;
    const d = (Math.abs(r - feed[i]) + Math.abs(g - feed[N + i]) + Math.abs(b - feed[2 * N + i])) /
      3;
    const t = Math.min(1, d * 3.5); // amplify for visibility
    // teal→amber ramp: little change = dark teal, big change (haze cleared) = warm amber
    hazeMap[i * 4] = 20 + t * 235;
    hazeMap[i * 4 + 1] = 60 + t * 150;
    hazeMap[i * 4 + 2] = 90 - t * 70;
    hazeMap[i * 4 + 3] = 255;
  }

  const outC = new OffscreenCanvas(w, h);
  outC.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  const dehazedBmp = outC.transferToImageBitmap();
  const hazeBmp = await createImageBitmap(new ImageData(hazeMap, w, h));

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    dehazedBmp,
    hazeBmp,
    w,
    h,
    imgW: iw,
    imgH: ih,
    inContrast,
    outContrast,
    ms,
    infMs,
    device,
  }, [dehazedBmp, hazeBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await dehaze(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
