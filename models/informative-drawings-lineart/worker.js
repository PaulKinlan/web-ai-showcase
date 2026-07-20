// Informative-Drawings line-art worker — ALL inference AND the dense output composite off the main
// thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no informative-drawings model class and
// there is no `line-art`/`image-to-line-drawing` pipeline task. The ONNX export is a clean image→image
// graph, so we run it directly with onnxruntime-web and hand-write the pre/post a pipeline would own:
// (1) resize the photo (long side ≤ maxSide, rounded to a multiple of 8) and pack it as [1,3,H,W]
// float32 in [0,1] — plain ToTensor, NO ImageNet normalisation — and (2) turn the single-channel line
// map back into a drawing (dark ink on white paper) plus an accent-tinted overlay on the original photo.
// This is the isolated per-worker ORT-web escape hatch (like models/microdehaze-image-dehazing/worker.js)
// — onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: rocca/informative-drawings-line-art-onnx (model.onnx, ~17 MB). Informative Drawings —
// "Learning to generate line drawings that convey geometry and semantics" (Chan et al., CVPR 2022;
// upstream carolineec/informative-drawings, MIT). A generator that turns a photo into an artistic line
// drawing (this is the "anime"/line-art style checkpoint). Input "input" [1,3,H,W] float32 in [0,1]
// (dynamic H,W). Output "output" [1,1,H,W] float32 in [0,1] — a single-channel drawing where ~1 = white
// paper and low values = ink lines. Everything stays on-device: the image never leaves the tab.
//
// Runnability was proven FIRST in headless Chrome: on a city-street photo the output was a real sparse
// line drawing — 77% white paper, ~22% line/mid-tone structure grounded in the scene's edges (not noise,
// not a flat field). Inference ~3.7 s @512² on single-thread WASM.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "rocca/informative-drawings-line-art-onnx";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/model.onnx`;
const CACHE_NAME = "informative-drawings-onnx-cache";

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/rocca/informative-drawings-line-art-onnx/") sees them → auto-init on a returning visit, honest
// Download on first visit, and the per-model "clear cache" control all work. Streams download progress.
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
  inputName = session.inputNames[0]; // "input"
  outputName = session.outputNames[0]; // "output"
  post({ type: "ready", device });
}

async function trace(id, bitmap, opts) {
  await ensureLoaded();
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 512;
  const inkThreshold = opts?.inkThreshold ?? 0.6; // output value below this counts as an ink line
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  const w = Math.max(8, Math.round(iw * scale / 8) * 8);
  const h = Math.max(8, Math.round(ih * scale / 8) * 8);
  const t0 = performance.now();

  const c = new OffscreenCanvas(w, h);
  const cctx = c.getContext("2d", { willReadFrequently: true });
  cctx.drawImage(bitmap, 0, 0, iw, ih, 0, 0, w, h);
  bitmap.close?.();
  const rgbaSrc = cctx.getImageData(0, 0, w, h).data;
  const N = w * h;
  const feed = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    feed[i] = rgbaSrc[i * 4] / 255;
    feed[N + i] = rgbaSrc[i * 4 + 1] / 255;
    feed[2 * N + i] = rgbaSrc[i * 4 + 2] / 255;
  }

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, h, w]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // [1,1,h,w] in [0,1]; ~1 = paper, low = ink
  const infMs = Math.round(performance.now() - t0);

  // Line drawing (grayscale, ink on white paper) + an accent overlay (lines tinted onto the photo) +
  // ink coverage (fraction of pixels below the ink threshold).
  const draw = new Uint8ClampedArray(N * 4);
  const overlay = new Uint8ClampedArray(N * 4);
  let ink = 0;
  for (let i = 0; i < N; i++) {
    const v = Math.max(0, Math.min(1, out[i]));
    const g = v * 255;
    draw[i * 4] = g;
    draw[i * 4 + 1] = g;
    draw[i * 4 + 2] = g;
    draw[i * 4 + 3] = 255;
    // overlay: dim the photo, then paint indigo ink where the drawing is dark (1 − v = line strength)
    const strength = 1 - v;
    if (v < inkThreshold) ink++;
    const dim = 0.55;
    overlay[i * 4] = rgbaSrc[i * 4] * dim * (1 - strength) + 79 * strength;
    overlay[i * 4 + 1] = rgbaSrc[i * 4 + 1] * dim * (1 - strength) + 70 * strength;
    overlay[i * 4 + 2] = rgbaSrc[i * 4 + 2] * dim * (1 - strength) + 229 * strength;
    overlay[i * 4 + 3] = 255;
  }
  const inkFraction = ink / N;

  const dc = new OffscreenCanvas(w, h);
  dc.getContext("2d").putImageData(new ImageData(draw, w, h), 0, 0);
  const lineBmp = dc.transferToImageBitmap();
  const overlayBmp = await createImageBitmap(new ImageData(overlay, w, h));

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    lineBmp,
    overlayBmp,
    w,
    h,
    imgW: iw,
    imgH: ih,
    inkFraction,
    ms,
    infMs,
    device,
  }, [lineBmp, overlayBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await trace(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
