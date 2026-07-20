// SCUNet image-denoising worker — ALL inference AND the dense output composite off the main thread via
// raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no SCUNet model class and there is no
// image-denoising pipeline task. The ONNX export is a clean image→image graph, so we run it directly
// with onnxruntime-web and hand-write the pre/post a pipeline would own: (1) resize the noisy input
// (long side ≤ maxSide, rounded to a multiple of 64 for the swin-conv U-Net's downsampling) and pack it
// as [1,3,H,W] float32 in [0,1] — plain RGB, no normalisation — and (2) clamp the denoised tensor back
// to an RGBA image. This is the isolated per-worker ORT-web escape hatch (like
// models/microdehaze-image-dehazing/worker.js) — onnxruntime-web is pinned HERE only, never in shared
// libs.
//
// Model: Heliosoph/scunet-onnx (scunet_color_real_psnr.onnx + .onnx.data, Apache-2.0, ~75 MB). SCUNet
// (Swin-Conv-UNet, Zhang et al. — "Practical Blind Denoising via Swin-Conv-UNet") — a strong real-image
// denoiser. We ship the BLIND real-world PSNR variant (trained on mixed synthetic degradations —
// Gaussian + JPEG + downsampling — with pixel loss), the model card's recommended general-purpose photo
// denoiser that stays faithful to the input (as opposed to the GAN variant, which invents texture).
// Input "image" [1,3,H,W] float32 in [0,1] (dynamic H,W). Output "denoised" [1,3,H,W] float32 in [0,1].
// The weights live in an external-data sidecar (.onnx.data); ORT-Web loads it via the session
// `externalData` option (verified working). Everything stays on-device: the image never leaves the tab.
//
// Runnability was proven FIRST in headless Chrome: on an image with added Gaussian noise (σ=25) the
// high-frequency (Laplacian) energy fell 0.276 → 0.069 while the RMSE-to-clean dropped 0.090 → 0.048 —
// real noise removal that moves TOWARD the clean reference, not a flat blur. Inference ~7 s @320² on
// single-thread WASM (a heavy model).

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "Heliosoph/scunet-onnx";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/scunet_color_real_psnr.onnx`;
const DATA_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/scunet_color_real_psnr.onnx.data`;
const DATA_NAME = "scunet_color_real_psnr.onnx.data"; // must match the external-data reference in the ONNX
const CACHE_NAME = "scunet-denoise-onnx-cache";

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch a file THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/Heliosoph/scunet-onnx/") sees it → auto-init on a returning visit, honest Download on first visit,
// and the per-model "clear cache" control all work. Streams download progress.
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
  // The two files are ~3.7 MB graph + ~71 MB weights; report combined progress (weights weighted 92%).
  const dataBytes = await fetchCached(DATA_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 92 } });
  });
  const modelBytes = await fetchCached(MODEL_URL, cache, () => {
    post({ type: "progress", p: { status: "progress", progress: 97 } });
  });
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    externalData: [{ path: DATA_NAME, data: dataBytes }],
  });
  inputName = session.inputNames[0]; // "image"
  outputName = session.outputNames[0]; // "denoised"
  post({ type: "ready", device });
}

// Mean absolute Laplacian of the luma plane — high-frequency energy. Noise injects HF energy;
// denoising removes it (a clean image sits in between: real detail but no noise).
function hfEnergy(plane, w, h) {
  let s = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * plane[i] - plane[i - 1] - plane[i + 1] - plane[i - w] - plane[i + w];
      s += Math.abs(lap);
      n++;
    }
  }
  return s / n;
}

async function denoise(id, bitmap, opts) {
  await ensureLoaded();
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 320;
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  // SCUNet is a swin-conv U-Net; round to a multiple of 64 so its windows + downsampling line up.
  const w = Math.max(64, Math.round(iw * scale / 64) * 64);
  const h = Math.max(64, Math.round(ih * scale / 64) * 64);
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
  const lumaOf = (p) => {
    const o = new Float32Array(N);
    for (let i = 0; i < N; i++) o[i] = 0.2126 * p[i] + 0.7152 * p[N + i] + 0.0722 * p[2 * N + i];
    return o;
  };
  const inHF = hfEnergy(lumaOf(feed), w, h);

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, h, w]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // [1,3,h,w] in [0,1]
  const infMs = Math.round(performance.now() - t0);
  const outHF = hfEnergy(lumaOf(out), w, h);

  // Denoised RGBA + a "noise removed" map: per-pixel change |out-in|, amplified. It lights up where the
  // model stripped noise/grain — busy noisy areas glow, already-clean flats stay dark.
  const rgba = new Uint8ClampedArray(N * 4);
  const noiseMap = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    const r = Math.max(0, Math.min(1, out[i]));
    const g = Math.max(0, Math.min(1, out[N + i]));
    const b = Math.max(0, Math.min(1, out[2 * N + i]));
    rgba[i * 4] = r * 255;
    rgba[i * 4 + 1] = g * 255;
    rgba[i * 4 + 2] = b * 255;
    rgba[i * 4 + 3] = 255;
    const d = (Math.abs(r - feed[i]) + Math.abs(g - feed[N + i]) + Math.abs(b - feed[2 * N + i])) /
      3;
    const t = Math.min(1, d * 6); // amplify for visibility
    // violet→green ramp: little change = dark violet, big change (noise removed) = green
    noiseMap[i * 4] = 40 + t * 60;
    noiseMap[i * 4 + 1] = 20 + t * 210;
    noiseMap[i * 4 + 2] = 70 - t * 40;
    noiseMap[i * 4 + 3] = 255;
  }

  const outC = new OffscreenCanvas(w, h);
  outC.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  const denoisedBmp = outC.transferToImageBitmap();
  const noiseBmp = await createImageBitmap(new ImageData(noiseMap, w, h));

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    denoisedBmp,
    noiseBmp,
    w,
    h,
    imgW: iw,
    imgH: ih,
    inHF,
    outHF,
    ms,
    infMs,
    device,
  }, [denoisedBmp, noiseBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await denoise(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
