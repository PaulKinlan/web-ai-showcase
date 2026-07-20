// Real-ESRGAN 4× super-resolution worker — ALL inference AND the dense output composite off the main
// thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js has no Real-ESRGAN (RRDBNet) model class and the
// generic `image-to-image` pipeline doesn't cover a GAN 4× upscaler with this I/O. The ONNX export is a
// clean image→image graph, so we run it directly with onnxruntime-web and hand-write the pre/post: (1)
// resize the input DOWN to a bounded thumbnail (long side ≤ maxIn) — 4× output of a large image would
// blow WASM memory and take minutes — and pack it as [1,3,h,w] float32 in [0,1] (plain RGB, no
// normalisation), and (2) clamp the 4× output tensor [1,3,4h,4w] to RGBA. We ALSO render a plain
// bicubic 4× of the same thumbnail so the page can show an HONEST like-for-like comparison (both at the
// output resolution) — Real-ESRGAN vs naive enlargement — plus a "detail-gain" map of what the GAN
// synthesised beyond bicubic. This is the isolated per-worker ORT-web escape hatch (like NAFNet /
// DDColor) — onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: SceneWorks/real-esrgan-onnx (real_esrgan_x4.onnx, BSD-3-Clause, ~67 MB) — an ONNX export of
// Real-ESRGAN x4plus (Wang et al., ICCVW 2021), the real-world blind super-resolution network. Input
// "input" [1,3,h,w] float32, output "output" [1,3,4h,4w] float32. Everything stays on-device.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "SceneWorks/real-esrgan-onnx";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/real_esrgan_x4.onnx`;
const CACHE_NAME = "real-esrgan-onnx-cache";
const SCALE = 4;

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/SceneWorks/real-esrgan-onnx/") sees them → auto-init on a returning visit, honest Download on first
// visit, and the per-model "clear cache" control all work. Streams download progress.
async function fetchModelBytes() {
  const cache = await caches.open(CACHE_NAME);
  let resp = await cache.match(MODEL_URL);
  if (!resp) {
    const net = await fetch(MODEL_URL);
    if (!net.ok || !net.body) throw new Error(`model fetch failed (${net.status})`);
    const total = Number(net.headers.get("content-length")) || 0;
    const reader = net.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      if (total) {
        post({ type: "progress", p: { status: "progress", progress: (received / total) * 100 } });
      }
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) {
      buf.set(c, off);
      off += c.length;
    }
    await cache.put(
      MODEL_URL,
      new Response(buf, {
        headers: { "content-length": String(received), "content-type": "application/octet-stream" },
      }),
    );
    return buf;
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  const bytes = await fetchModelBytes();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  post({ type: "ready", device });
}

// Laplacian variance of luma of an RGBA buffer — the standard image-sharpness / detail measure. A naive
// enlargement stays soft (low); Real-ESRGAN reconstructs edges and texture (higher).
function sharpnessRGBA(rgba, w, h) {
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) g[i] = (rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2]) / 3;
  let s = 0, ss = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * g[i] - g[i - 1] - g[i + 1] - g[i - w] - g[i + w];
      s += lap;
      ss += lap * lap;
      n++;
    }
  }
  const m = s / n;
  return ss / n - m * m;
}

async function upscale(id, bitmap, opts) {
  await ensureLoaded();
  const maxIn = opts?.maxIn ?? 192; // bound the input so 4× stays feasible on single-thread WASM
  const long = Math.max(bitmap.width, bitmap.height);
  const s = Math.min(maxIn / long, 1);
  const iw = Math.max(1, Math.round(bitmap.width * s));
  const ih = Math.max(1, Math.round(bitmap.height * s));
  const ow = iw * SCALE, oh = ih * SCALE;
  const t0 = performance.now();

  // Network input: the bounded thumbnail, packed [1,3,ih,iw] float32 in [0,1].
  const inC = new OffscreenCanvas(iw, ih);
  const ictx = inC.getContext("2d", { willReadFrequently: true });
  ictx.imageSmoothingQuality = "high";
  ictx.drawImage(bitmap, 0, 0, bitmap.width, bitmap.height, 0, 0, iw, ih);
  bitmap.close?.();
  const inData = ictx.getImageData(0, 0, iw, ih).data;
  const IN = iw * ih;
  const feed = new Float32Array(3 * IN);
  for (let i = 0; i < IN; i++) {
    feed[i] = inData[i * 4] / 255;
    feed[IN + i] = inData[i * 4 + 1] / 255;
    feed[2 * IN + i] = inData[i * 4 + 2] / 255;
  }

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, ih, iw]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // Float32Array(3*oh*ow), ~[0,1]
  const infMs = Math.round(performance.now() - t0);
  const OUT = ow * oh;

  // Real-ESRGAN output → RGBA at ow×oh.
  const esr = new Uint8ClampedArray(OUT * 4);
  for (let i = 0; i < OUT; i++) {
    esr[i * 4] = Math.max(0, Math.min(1, out[i])) * 255;
    esr[i * 4 + 1] = Math.max(0, Math.min(1, out[OUT + i])) * 255;
    esr[i * 4 + 2] = Math.max(0, Math.min(1, out[2 * OUT + i])) * 255;
    esr[i * 4 + 3] = 255;
  }

  // Honest baseline: a plain bicubic 4× of the SAME thumbnail (what "just enlarge it" gives you),
  // rendered at the output resolution so the comparison is like-for-like.
  const bicC = new OffscreenCanvas(ow, oh);
  const bctx = bicC.getContext("2d", { willReadFrequently: true });
  bctx.imageSmoothingQuality = "high";
  bctx.drawImage(inC, 0, 0, iw, ih, 0, 0, ow, oh);
  const bicData = bctx.getImageData(0, 0, ow, oh).data;

  // Detail-gain map: |esrgan − bicubic| amplified — WHERE the GAN synthesised detail beyond a naive
  // upscale (edges, texture). Flat regions stay dark.
  const delta = new Uint8ClampedArray(OUT * 4);
  for (let i = 0; i < OUT; i++) {
    const d =
      (Math.abs(esr[i * 4] - bicData[i * 4]) + Math.abs(esr[i * 4 + 1] - bicData[i * 4 + 1]) +
        Math.abs(esr[i * 4 + 2] - bicData[i * 4 + 2])) / 3;
    const v = Math.min(255, d * 4); // amplify ×4 for visibility
    delta[i * 4] = 20 + v * 0.2;
    delta[i * 4 + 1] = 20 + v * 0.5;
    delta[i * 4 + 2] = 40 + v * 0.85;
    delta[i * 4 + 3] = 255;
  }

  const sharpBic = sharpnessRGBA(bicData, ow, oh);
  const sharpEsr = sharpnessRGBA(esr, ow, oh);

  const esrBmp = await createImageBitmap(new ImageData(esr, ow, oh));
  const bicBmp = await createImageBitmap(new ImageData(new Uint8ClampedArray(bicData), ow, oh));
  const deltaBmp = await createImageBitmap(new ImageData(delta, ow, oh));

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    esrBmp,
    bicBmp,
    deltaBmp,
    inW: iw,
    inH: ih,
    outW: ow,
    outH: oh,
    scale: SCALE,
    sharpBic,
    sharpEsr,
    ms,
    infMs,
    device,
  }, [esrBmp, bicBmp, deltaBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await upscale(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
