// FaceNet face-embedding worker — ALL inference off the main thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5 has no FaceNet / InceptionResnetV1 model
// class and no `image-feature-extraction` path for this ONNX (model_type is a bare TF-exported graph,
// not a registered architecture). So we run the ONNX graph directly with onnxruntime-web, hand-writing
// the two pieces transformers.js would normally own: (1) the face-crop → 160×160 NHWC preprocessing
// with FaceNet "prewhitening" (per-image standardization), and (2) L2-normalizing the raw embedding.
// This is the isolated per-worker ORT-web escape hatch (like models/yolov10-detection/worker.js) —
// onnxruntime-web is pinned HERE only, never in shared lib/webai.js.
//
// Model: astaileyyoung/facenet-onnx (facenet.onnx, MIT, ~91 MB). InceptionResnetV1 trained on
// VGGFace2. Input: [1, 160, 160, 3] (NHWC, prewhitened). Output: [1, 128] face embedding. Two faces
// are "similar" when the cosine of their L2-normalized embeddings is high. EVERYTHING stays on-device —
// no image, crop, or embedding ever leaves the tab. This is face SIMILARITY, not identity or lookup.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "astaileyyoung/facenet-onnx";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/facenet.onnx`;
const CACHE_NAME = "facenet-onnx-cache";
const SIZE = 160; // network input side

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so the shared model-cache layer (which scans caches for
// "/astaileyyoung/facenet-onnx/") sees them → auto-init on a returning visit, honest Download on first
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

// Centre-square-crop an ImageBitmap and draw it into a 160×160 NHWC float tensor with FaceNet
// "prewhitening": subtract the per-image mean and divide by the per-image std (clamped so a flat
// image doesn't blow up). This is the exact standardization davidsandberg/facenet applies at inference.
function preprocess(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const side = Math.min(w, h);
  const sx = (w - side) / 2, sy = (h - side) / 2;
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, sx, sy, side, side, 0, 0, SIZE, SIZE);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const N = SIZE * SIZE * 3;
  const px = new Float32Array(N);
  let sum = 0, sum2 = 0;
  for (let i = 0; i < SIZE * SIZE; i++) {
    for (let c = 0; c < 3; c++) {
      const v = data[i * 4 + c];
      px[i * 3 + c] = v; // NHWC
      sum += v;
      sum2 += v * v;
    }
  }
  const mean = sum / N;
  const std = Math.sqrt(Math.max(sum2 / N - mean * mean, 0));
  const stdAdj = Math.max(std, 1 / Math.sqrt(N));
  for (let i = 0; i < N; i++) px[i] = (px[i] - mean) / stdAdj;
  return px;
}

async function embed(id, bitmap) {
  await ensureLoaded();
  const t0 = performance.now();
  const px = preprocess(bitmap);
  bitmap.close?.();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", px, [1, SIZE, SIZE, 3]);
  const results = await session.run(feeds);
  const raw = results[outputName].data; // Float32Array(128), NOT L2-normalized
  // L2-normalize so cosine similarity == dot product and the threshold is stable.
  let norm = 0;
  for (const v of raw) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  const emb = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) emb[i] = raw[i] / norm;
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, embedding: emb, dims: raw.length, ms, device }, [emb.buffer]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await embed(e.data.id, e.data.bitmap);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
