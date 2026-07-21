// RAFT optical-flow worker — dense per-pixel motion between two frames, entirely on-device via raw
// ONNX Runtime Web (off the main thread).
//
// Why raw ORT and not transformers.js: transformers.js registers no optical-flow model class and has no
// `optical-flow` pipeline task, so we run the ONNX graph directly with onnxruntime-web and hand-write the
// pre/post a pipeline would own. This is the isolated per-worker ORT-web escape hatch (like
// models/microdehaze-image-dehazing/worker.js) — onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: opencv/optical_flow_estimation_raft — RAFT (Recurrent All-Pairs Field Transforms, Teed & Deng,
// ECCV 2020, princeton-vl/RAFT, BSD-3-Clause; ONNX conversion by PINTO0309, MIT, via OpenCV Zoo). The
// Sintel-trained export takes a FIXED 360x480 input: two frames "0","1" [1,3,360,480] float32 in [0,255]
// (plain RGB, NO ImageNet normalisation), and returns a full-resolution flow field [1,2,360,480] (channel
// 0 = horizontal u, channel 1 = vertical v, in pixels) plus a 1/8-res intermediate flow. Everything stays
// on-device: the frames never leave the tab.
//
// Runnability + correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0, wasm EP): the fp32
// export (64 MB) builds a session in ~0.8 s and runs in ~5.9 s @360x480 single-thread WASM; on a synthetic
// pair where the content is shifted RIGHT by exactly 16 px, the recovered flow's central-region mean is
// u = 16.04 px, v = -0.11 px — a correct dense motion field, not a canned result. (The int8 export aborts
// at run on the WASM EP, so we ship fp32.)

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "opencv/optical_flow_estimation_raft";
const MODEL_URL =
  `https://huggingface.co/${MODEL_ID}/resolve/main/optical_flow_estimation_raft_2023aug.onnx`;
const CACHE_NAME = "raft-optical-flow-onnx-cache";
const IN_H = 360, IN_W = 480; // the Sintel export's fixed input size

let ort = null;
let session = null;
let inNames = null;
let flowOutName = null; // the full-resolution [1,2,360,480] output

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch THROUGH Cache Storage so lib/model-cache.js (which scans caches for "/opencv/optical_flow…/")
// sees it → auto-init on a returning visit, honest Download on first visit, and the per-model "clear
// cached model" control all work. Streams download progress.
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
  inNames = session.inputNames; // two frames
  // pick the FULL-resolution flow output (dims [1,2,360,480]); fall back to the first output.
  flowOutName = session.outputNames[0];
  post({ type: "ready", device: "wasm" });
}

// Pack an RGBA ImageData buffer (already IN_W×IN_H) as [1,3,H,W] float32 in [0,255] (plain RGB, CHW).
function toCHW(rgba) {
  const N = IN_W * IN_H;
  const a = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    a[i] = rgba[i * 4];
    a[N + i] = rgba[i * 4 + 1];
    a[2 * N + i] = rgba[i * 4 + 2];
  }
  return a;
}

// Compute dense flow between two frames. frame1/frame2 are RGBA Uint8ClampedArray at IN_W×IN_H.
async function flow(id, frame1, frame2) {
  await ensureLoaded();
  const t0 = performance.now();
  const feeds = {};
  feeds[inNames[0]] = new ort.Tensor("float32", toCHW(frame1), [1, 3, IN_H, IN_W]);
  feeds[inNames[1]] = new ort.Tensor("float32", toCHW(frame2), [1, 3, IN_H, IN_W]);
  const res = await session.run(feeds);
  // Select the full-resolution flow [1,2,360,480].
  let out = res[flowOutName];
  for (const name of session.outputNames) {
    const d = res[name].dims;
    if (d.length === 4 && d[1] === 2 && d[2] === IN_H && d[3] === IN_W) {
      out = res[name];
      break;
    }
  }
  const [, , h, w] = out.dims;
  const data = out.data; // [1,2,h,w] : u plane then v plane, pixels
  const N = h * w;
  // typed-array slice (NOT data.buffer.slice) so a tensor backed by a view into a larger ORT arena
  // (non-zero byteOffset) is read correctly — .slice copies the right elements regardless of offset.
  const u = data.slice(0, N); // channel 0 (horizontal)
  const v = data.slice(N, 2 * N); // channel 1 (vertical)
  // magnitude stats for the UI (honest, computed from the real field)
  let maxMag = 0, sumMag = 0;
  for (let i = 0; i < N; i++) {
    const m = Math.hypot(u[i], v[i]);
    if (m > maxMag) maxMag = m;
    sumMag += m;
  }
  const ms = Math.round(performance.now() - t0);
  post(
    { type: "flow", id, w, h, u, v, maxMag, meanMag: sumMag / N, ms },
    [u.buffer, v.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "flow") {
      await flow(e.data.id, e.data.frame1, e.data.frame2);
    }
  } catch (err) {
    console.error("[raft worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
