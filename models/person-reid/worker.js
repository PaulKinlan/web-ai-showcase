// Person re-identification worker — turns a person crop into an appearance EMBEDDING, entirely on-device
// via raw ONNX Runtime Web (off the main thread).
//
// Why raw ORT and not transformers.js: transformers.js has no person-reid class and no such task, so we
// run the ONNX graph directly with onnxruntime-web and hand-write the pre/post a pipeline would own. This
// is the isolated per-worker ORT-web escape hatch (like models/raft-optical-flow/worker.js) — onnxruntime-
// web is pinned HERE only, never in shared libs.
//
// Model: opencv/person_reid_youtureid — YouTu-ReID (OpenCV Zoo, Apache-2.0). A ResNet-50-IBN backbone that
// maps a person crop "input" [1,3,256,128] float32 in [0,1] (plain RGB, CHW) to a 768-d "output" [1,768,1,1]
// appearance embedding. Cosine similarity of two L2-normalised embeddings tells whether two crops are the
// SAME person — DISTINCT from the built face-embedding demo (that's a face; this is the whole body: gait,
// clothing, build — how surveillance/retail systems re-find a person across cameras). Nothing leaves the tab.
//
// Runnability + correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0, WASM EP, no GPU): the
// fp32 export (107 MB) runs in ~1 s per crop; on three runners cropped from a licensed marathon photo, a
// person vs a jittered/brighter crop of the SAME person scored cosine 0.939, while different runners scored
// 0.65–0.75 — the embedding separates identities, not a canned result. (The int8 exports risk aborting on
// the WASM EP, so we ship fp32.)

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "opencv/person_reid_youtureid";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/person_reid_youtu_2021nov.onnx`;
const CACHE_NAME = "person-reid-onnx-cache";
const IN_H = 256, IN_W = 128; // the model's fixed person-crop input size

let ort = null;
let session = null;
let inName = null;
let outName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch THROUGH Cache Storage so lib/model-cache.js (which scans caches for "/opencv/person_reid…/") sees
// it → auto-init on a returning visit, honest Download on first visit, and the per-model "clear cached
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

// Embed a person crop (RGBA Uint8ClampedArray, already IN_W×IN_H) → an L2-normalised 768-d vector.
async function embed(id, rgba) {
  await ensureLoaded();
  const t0 = performance.now();
  const N = IN_W * IN_H;
  const a = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    a[i] = rgba[i * 4] / 255;
    a[N + i] = rgba[i * 4 + 1] / 255;
    a[2 * N + i] = rgba[i * 4 + 2] / 255;
  }
  const res = await session.run({ [inName]: new ort.Tensor("float32", a, [1, 3, IN_H, IN_W]) });
  const raw = res[outName].data; // 768-d
  let norm = 0;
  for (let i = 0; i < raw.length; i++) norm += raw[i] * raw[i];
  norm = Math.sqrt(norm) || 1;
  const vec = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) vec[i] = raw[i] / norm;
  post({ type: "embedding", id, vec, dim: vec.length, ms: Math.round(performance.now() - t0) }, [
    vec.buffer,
  ]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "embed") await embed(e.data.id, e.data.crop);
  } catch (err) {
    console.error("[person-reid worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
