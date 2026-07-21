// Face image-quality assessment worker — scores how usable a face crop is (sharpness, pose, lighting)
// for face recognition, entirely on-device via raw ONNX Runtime Web (off the main thread).
//
// Why raw ORT and not transformers.js: transformers.js has no face-IQA class and no such task, so we run
// the ONNX graph directly with onnxruntime-web. This is the isolated per-worker ORT-web escape hatch
// (like models/raft-optical-flow/worker.js) — onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: opencv/face_image_quality_assessment_ediffiqa — eDifFIQA(T) (OpenCV Zoo, Apache-2.0). A tiny
// (~7.3 MB) network distilled from a denoising-diffusion FIQA teacher. Input "input" [1,3,112,112] float32
// in [0,1] (plain RGB, CHW) → "output" [1,1]: a single scalar QUALITY score (higher = a better face for
// recognition). DISTINCT from the built face-detection/landmark/embedding demos — it doesn't find or
// identify a face, it JUDGES the capture (is it sharp, well-lit, frontal enough to trust?). Face-quality
// gating is what biometric pipelines use to reject bad frames before spending compute on recognition.
// Nothing leaves the tab.
//
// Runnability + correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0, WASM EP, no GPU): on
// a licensed portrait the score fell monotonically as the image was degraded — sharp 0.431, blur-3 0.138,
// blur-8 0.106 — i.e. the model really tracks capture quality, not a canned number. ~7 MB, sub-second.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "opencv/face_image_quality_assessment_ediffiqa";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/ediffiqa_tiny_jun2024.onnx`;
const CACHE_NAME = "face-image-quality-onnx-cache";
const IN = 112; // fixed square face-crop input

let ort = null;
let session = null;
let inName = null;
let outName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch THROUGH Cache Storage so lib/model-cache.js (which scans caches for "/opencv/face_image_quality…/")
// sees it → auto-init on a returning visit, honest Download on first visit, and the per-model "clear cached
// model" control all work. Streams download progress.
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
  const modelBytes = await fetchCached(MODEL_URL, cache, (r, t) => {
    if (t) post({ type: "progress", p: { status: "progress", progress: (r / t) * 98 } });
  });
  session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  inName = session.inputNames[0];
  outName = session.outputNames[0];
  post({ type: "ready", device: "wasm" });
}

// Score a face crop (RGBA Uint8ClampedArray, already IN×IN) → a scalar quality value.
async function assess(id, rgba) {
  await ensureLoaded();
  const t0 = performance.now();
  const N = IN * IN;
  const a = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    a[i] = rgba[i * 4] / 255;
    a[N + i] = rgba[i * 4 + 1] / 255;
    a[2 * N + i] = rgba[i * 4 + 2] / 255;
  }
  const res = await session.run({ [inName]: new ort.Tensor("float32", a, [1, 3, IN, IN]) });
  const score = res[outName].data[0];
  post({ type: "quality", id, score, ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "assess") await assess(e.data.id, e.data.crop);
  } catch (err) {
    console.error("[face-iqa worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
