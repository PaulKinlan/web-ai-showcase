// Sapiens surface-normal worker — ALL inference off the main thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5 registers no `sapiens` model class
// (config model_type = "sapiens"), so `pipeline(...)` can't load it. The ONNX export is a clean
// image→image graph, so we run it directly with onnxruntime-web, hand-writing the preprocessing
// (ImageNet normalize to [1,3,1024,768]) and postprocessing (per-pixel L2-normalize the predicted
// vectors, then RGB-encode the normal map). This is the isolated per-worker ORT-web escape hatch
// (like models/yolov10-detection/worker.js) — onnxruntime-web is pinned HERE only.
//
// Model: onnx-community/sapiens-normal-0.3b (model_quantized.onnx, int8, ~444 MB). Sapiens is a
// human-centric ViT; it predicts a per-pixel SURFACE NORMAL — the 3D orientation (which way the
// surface faces), distinct from depth (how far away it is). Input pixel_values [1,3,1024,768],
// output predicted_normal [1,3,512,384] (raw XYZ vectors in camera space).
//
// NOTE ON SPEED: single-threaded WASM (GitHub Pages can't set COOP/COEP for threads) runs this large
// ViT at ~60–90 s per image. That is honest, real inference — the page says so up front. A WebGPU
// build would be far faster; this demo ships the universal CPU path so it runs everywhere.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "onnx-community/sapiens-normal-0.3b";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model_quantized.onnx`;
const CACHE_NAME = "sapiens-normal-onnx-cache";
const IN_W = 768, IN_H = 1024; // fixed network input
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225]; // ImageNet

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/onnx-community/sapiens-normal-0.3b/") sees them → auto-init on a returning visit, honest Download
// on first visit, and the per-model "clear cache" control all work. Streams download progress.
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

// Resize the source bitmap into the fixed 768×1024 network input and build the ImageNet-normalized
// CHW float tensor.
function preprocess(bitmap) {
  const canvas = new OffscreenCanvas(IN_W, IN_H);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, IN_W, IN_H);
  const { data } = ctx.getImageData(0, 0, IN_W, IN_H);
  const plane = IN_W * IN_H;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    chw[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    chw[plane + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    chw[2 * plane + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return chw;
}

// RGB-encode a normalized normal field into an RGBA ImageBitmap, OFF the main thread (the dense-output
// composite). Standard encoding: channel = (component * 0.5 + 0.5) * 255, so a camera-facing surface
// (n≈[0,0,1]) reads as the classic light-blue (128,128,255).
function encodeToBitmap(nrm, w, h) {
  if (typeof OffscreenCanvas === "undefined") return null;
  const off = new OffscreenCanvas(w, h);
  const ctx = off.getContext("2d");
  const img = ctx.createImageData(w, h);
  const px = img.data;
  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    px[i * 4] = (nrm[i] * 0.5 + 0.5) * 255;
    px[i * 4 + 1] = (nrm[plane + i] * 0.5 + 0.5) * 255;
    px[i * 4 + 2] = (nrm[2 * plane + i] * 0.5 + 0.5) * 255;
    px[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return off.transferToImageBitmap();
}

async function run(id, bitmap) {
  await ensureLoaded();
  const t0 = performance.now();
  const chw = preprocess(bitmap);
  bitmap.close?.();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", chw, [1, 3, IN_H, IN_W]);
  const results = await session.run(feeds);
  const out = results[outputName];
  const [, , H, W] = out.dims; // [1,3,512,384]
  const raw = out.data; // Float32Array(3*H*W), CHW, raw XYZ vectors
  const plane = H * W;
  // Per-pixel L2-normalize → unit normals (CHW layout preserved). Track the mean direction.
  const nrm = new Float32Array(3 * plane);
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < plane; i++) {
    const x = raw[i], y = raw[plane + i], z = raw[2 * plane + i];
    const m = Math.hypot(x, y, z) || 1;
    const nx = x / m, ny = y / m, nz = z / m;
    nrm[i] = nx;
    nrm[plane + i] = ny;
    nrm[2 * plane + i] = nz;
    mx += nx;
    my += ny;
    mz += nz;
  }
  const meanNormal = [mx / plane, my / plane, mz / plane];
  const bitmapOut = encodeToBitmap(nrm, W, H);
  const ms = Math.round(performance.now() - t0);
  const buf = nrm.buffer; // transfer the normal field for relighting / pixel-pick on the main thread
  const transfer = bitmapOut ? [buf, bitmapOut] : [buf];
  post({
    type: "result",
    id,
    width: W,
    height: H,
    normals: buf, // Float32 CHW, unit vectors
    bitmap: bitmapOut,
    meanNormal,
    dims: Array.from(out.dims),
    ms,
    device,
  }, transfer);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.bitmap);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
