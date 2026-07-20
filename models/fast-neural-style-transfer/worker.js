// Fast Neural Style Transfer worker — ALL inference AND the dense output composite off the main thread
// via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no style-transfer model class and there
// is no neural-style pipeline task. These are the classic ONNX Model Zoo fast-neural-style networks
// (Johnson et al. 2016, "Perceptual Losses for Real-Time Style Transfer") — one small feed-forward
// generator per artistic style. We run each ONNX directly with onnxruntime-web and hand-write the
// pre/post a pipeline would own: (1) resize the photo to the network's FIXED 224×224 and pack it as
// **NCHW** [1,3,224,224] float32 in the **0–255** byte range (these exports take un-normalised bytes),
// and (2) clamp the 0–255 output back to an RGBA image, resized to the display aspect. This is the
// isolated per-worker ORT-web escape hatch (like models/animegan-cartoonization/worker.js) —
// onnxruntime-web is pinned HERE only, never in shared libs.
//
// Models: onnxmodelzoo/{candy,mosaic,udnie,pointilism,rain-princess}-9 (one ~6.6 MB ONNX per style,
// Apache-2.0). Input "input1" [1,3,224,224] float32 in [0,255] (FIXED size). Output "output1"
// [1,3,224,224] float32 in ~[0,255]. Styles load lazily (only the ones you pick are downloaded).
// Everything stays on-device: the image never leaves the tab.
//
// Runnability was proven FIRST in headless Chrome: on real photos each style produced a genuine painted
// restyle — normalized RMSE(out,in) ~0.32 (a strong, real style change, not identity, not a colour
// curve). Inference ~0.4 s @224² on single-thread WASM. Because the network is fixed at 224², the photo
// is stretched to 224² for inference and the 224² output is resized back to the display aspect.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const STYLES = {
  candy: { repo: "onnxmodelzoo/candy-9", file: "candy-9.onnx" },
  mosaic: { repo: "onnxmodelzoo/mosaic-9", file: "mosaic-9.onnx" },
  udnie: { repo: "onnxmodelzoo/udnie-9", file: "udnie-9.onnx" },
  pointilism: { repo: "onnxmodelzoo/pointilism-9", file: "pointilism-9.onnx" },
  "rain-princess": { repo: "onnxmodelzoo/rain-princess-9", file: "rain-princess-9.onnx" },
};
const DEFAULT_STYLE = "candy";
const NET = 224; // fixed network input size
const CACHE_NAME = "fast-neural-style-onnx-cache";

let ort = null;
let device = "wasm";
const sessions = new Map(); // style -> { session, inputName, outputName }

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/onnxmodelzoo/candy-9/") sees them → auto-init on a returning visit, honest Download on first visit,
// and the per-model "clear cache" control all work. Streams download progress.
async function fetchModelBytes(url, onChunk) {
  const cache = await caches.open(CACHE_NAME);
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const net = await fetch(url);
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
    if (total) onChunk?.((received / total) * 100);
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

async function ensureStyle(style) {
  if (sessions.has(style)) return sessions.get(style);
  if (!ort) {
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  }
  const s = STYLES[style] || STYLES[DEFAULT_STYLE];
  const url = `https://huggingface.co/${s.repo}/resolve/main/${s.file}`;
  const bytes = await fetchModelBytes(
    url,
    (p) => post({ type: "progress", p: { status: "progress", progress: p } }),
  );
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const rec = { session, inputName: session.inputNames[0], outputName: session.outputNames[0] };
  sessions.set(style, rec);
  return rec;
}

async function stylize(id, bitmap, opts) {
  const style = opts?.style || DEFAULT_STYLE;
  const rec = await ensureStyle(style);
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 640;
  const dispScale = Math.min(maxSide / Math.max(iw, ih), 1);
  const dw = Math.max(1, Math.round(iw * dispScale));
  const dh = Math.max(1, Math.round(ih * dispScale));
  const t0 = performance.now();

  // Network input: stretch to the fixed 224×224, pack NCHW [1,3,224,224] in [0,255].
  const net = new OffscreenCanvas(NET, NET);
  const nctx = net.getContext("2d", { willReadFrequently: true });
  nctx.drawImage(bitmap, 0, 0, iw, ih, 0, 0, NET, NET);
  bitmap.close?.();
  const nData = nctx.getImageData(0, 0, NET, NET).data;
  const N = NET * NET;
  const feed = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    feed[i] = nData[i * 4];
    feed[N + i] = nData[i * 4 + 1];
    feed[2 * N + i] = nData[i * 4 + 2];
  }

  const feeds = {};
  feeds[rec.inputName] = new ort.Tensor("float32", feed, [1, 3, NET, NET]);
  const results = await rec.session.run(feeds);
  const out = results[rec.outputName].data; // NCHW [1,3,224,224] in ~[0,255]
  const infMs = Math.round(performance.now() - t0);

  // 224² RGBA + measure how much the image changed (mean abs per-pixel delta in [0,1]).
  const rgba = new Uint8ClampedArray(N * 4);
  let deltaSum = 0;
  for (let i = 0; i < N; i++) {
    const r = Math.max(0, Math.min(255, out[i]));
    const g = Math.max(0, Math.min(255, out[N + i]));
    const b = Math.max(0, Math.min(255, out[2 * N + i]));
    rgba[i * 4] = r;
    rgba[i * 4 + 1] = g;
    rgba[i * 4 + 2] = b;
    rgba[i * 4 + 3] = 255;
    deltaSum +=
      (Math.abs(r - feed[i]) + Math.abs(g - feed[N + i]) + Math.abs(b - feed[2 * N + i])) /
      3 / 255;
  }
  const styleDelta = deltaSum / N;

  // Resize the 224² styled image to the display aspect (worker-side dense composite).
  const src = new OffscreenCanvas(NET, NET);
  src.getContext("2d").putImageData(new ImageData(rgba, NET, NET), 0, 0);
  const dispC = new OffscreenCanvas(dw, dh);
  const dctx = dispC.getContext("2d");
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, NET, NET, 0, 0, dw, dh);
  const styledBmp = dispC.transferToImageBitmap();

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    styledBmp,
    w: dw,
    h: dh,
    imgW: iw,
    imgH: ih,
    style,
    styleDelta,
    ms,
    infMs,
    device,
  }, [styledBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureStyle(e.data.style || DEFAULT_STYLE);
      post({ type: "ready", device });
    } else if (type === "run") {
      await stylize(e.data.id, e.data.bitmap, e.data.opts);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
