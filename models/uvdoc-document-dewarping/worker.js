// UVDoc document-dewarping worker — ALL inference AND the dense output composite off the main thread
// via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no UVDoc model class and there is no
// document-unwarping pipeline task. The ONNX export is a clean image→image graph (it outputs the
// unwarped page directly, not a grid), so we run it with onnxruntime-web and hand-write the pre/post a
// pipeline would own: (1) resize the document photo (long side ≤ maxSide, rounded to a multiple of 32
// for the network's downsampling) and pack it as [1,3,H,W] float32 in [0,1] — plain RGB, no
// normalisation — and (2) clamp the flattened tensor back to an RGBA image. This is the isolated
// per-worker ORT-web escape hatch (like models/microdehaze-image-dehazing/worker.js) — onnxruntime-web
// is pinned HERE only, never in shared libs.
//
// Model: PaddlePaddle/UVDoc_onnx (inference.onnx, Apache-2.0, ~30 MB). UVDoc — the PaddleOCR document
// image unwarping model (based on the UVDoc paper: geometric + texture-consistent page rectification).
// Input "image" [1,3,H,W] float32 in [0,1] (dynamic H,W). Output "fetch_name_0" [1,3,H,W] float32 in
// [0,1] — the geometrically corrected (flattened, de-curled, de-skewed) page. Everything stays
// on-device: the document never leaves the tab.
//
// Runnability was proven FIRST in headless Chrome: on a synthetically curled+perspective-warped page the
// output flattened the page back to a clean rectangle (visual screenshot confirmed straightened text
// lines), RMSE(out,warped-input) 0.19 — a real geometric correction, not identity. Inference ~0.85 s
// @448² on single-thread WASM.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "PaddlePaddle/UVDoc_onnx";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/inference.onnx`;
const CACHE_NAME = "uvdoc-onnx-cache";

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/PaddlePaddle/UVDoc_onnx/") sees them → auto-init on a returning visit, honest Download on first
// visit, and the per-model "clear cache" control all work. Streams download progress.
async function fetchModelBytes() {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(MODEL_URL);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
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

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  const bytes = await fetchModelBytes();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  inputName = session.inputNames[0]; // "image"
  outputName = session.outputNames[0]; // "fetch_name_0"
  post({ type: "ready", device });
}

async function dewarp(id, bitmap, opts) {
  await ensureLoaded();
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 512;
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  // Round to a multiple of 32 so the encoder/decoder downsampling lines up.
  const w = Math.max(32, Math.round(iw * scale / 32) * 32);
  const h = Math.max(32, Math.round(ih * scale / 32) * 32);
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

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, h, w]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // [1,3,h,w] in [0,1] — the flattened page
  const infMs = Math.round(performance.now() - t0);

  // Correction magnitude: RMS pixel change (how much geometry the model moved).
  let ss = 0;
  const rgba = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    const r = Math.max(0, Math.min(1, out[i]));
    const g = Math.max(0, Math.min(1, out[N + i]));
    const b = Math.max(0, Math.min(1, out[2 * N + i]));
    rgba[i * 4] = r * 255;
    rgba[i * 4 + 1] = g * 255;
    rgba[i * 4 + 2] = b * 255;
    rgba[i * 4 + 3] = 255;
    const dr = r - feed[i], dg = g - feed[N + i], db = b - feed[2 * N + i];
    ss += (dr * dr + dg * dg + db * db) / 3;
  }
  const correction = Math.sqrt(ss / N);

  const outC = new OffscreenCanvas(w, h);
  outC.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
  const flatBmp = outC.transferToImageBitmap();

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    flatBmp,
    w,
    h,
    imgW: iw,
    imgH: ih,
    correction,
    ms,
    infMs,
    device,
  }, [flatBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await dewarp(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
