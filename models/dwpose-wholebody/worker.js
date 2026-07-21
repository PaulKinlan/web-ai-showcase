// Whole-body pose worker — estimates 133 body+feet+face+hand keypoints from a person crop entirely
// on-device via raw ONNX Runtime Web (off the main thread).
//
// Why raw ORT and not transformers.js: transformers.js has no RTMPose/DWPose (SimCC whole-body pose)
// class or task, so we run the ONNX graph directly with onnxruntime-web. This is the isolated per-worker
// ORT-web escape hatch (like models/raft-optical-flow/worker.js) — onnxruntime-web is pinned HERE only.
//
// Model: DWPose (dw-ll_ucoco_384) — a distilled RTMPose-l whole-body estimator from OpenMMLab's mmpose
// (Apache-2.0). Apache-2.0 permits redistribution, so the weights stay Apache-2.0 wherever mirrored; we
// fetch a faithful ONNX conversion and document that provenance. Input [1,3,384,288] float32 (a person
// crop, ImageNet-normalized RGB, top-down at W/H = 0.75) -> two SimCC heads simcc_x [1,133,576] +
// simcc_y [1,133,768] (a 1-D coordinate classification per axis, split ratio 2). DISTINCT from the built
// 17-keypoint body pose (VitPose/YOLOv8-pose): DWPose returns 133 keypoints in ONE pass — 17 body + 6 feet
// + 68 face + 42 hands (21 per hand). Nothing leaves the tab.
//
// Correctness proven FIRST in headless Chrome (onnxruntime-web 1.21.0, WASM EP, no GPU): on a ledgered
// runner crop the decoded keypoints are anatomically ordered top-to-bottom — nose y~21, shoulders y~53,
// hips y~155, ankles y~340 (in the 384-tall input) — with 130/133 keypoints above 0.3 confidence.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "yzd-v/DWPose"; // canonical DWPose model
// Apache-2.0 DWPose weights (dw-ll_ucoco_384.onnx), fetched from an Apache-2.0 mirror; license travels.
const MODEL_URL =
  "https://huggingface.co/Longcat2957/dwpose-onnx/resolve/main/dw-ll_ucoco_384.onnx";
const CACHE_NAME = "dwpose-onnx-cache";

export const IN_W = 288;
export const IN_H = 384;
export const N_KPTS = 133;
const SPLIT = 2; // SimCC split ratio
const MEAN = [123.675, 116.28, 103.53];
const STD = [58.395, 57.12, 57.375];

let ort = null;
let session = null;
let inName = null;
let xName = null;
let yName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch THROUGH Cache Storage so lib/model-cache.js (which scans caches for "/yzd-v/DWPose") sees it →
// auto-init on a returning visit, honest Download on first visit, and the "clear cached model" control all
// work. The cache key carries the model-id path so the scan matches. Streams download progress.
async function fetchCached(url, cache, onChunk) {
  const key = `https://huggingface.co/${MODEL_ID}/resolve/main/dw-ll_ucoco_384.onnx`;
  const hit = await cache.match(key);
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
    key,
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
  // Two SimCC outputs: x-axis has length IN_W*SPLIT (576), y-axis IN_H*SPLIT (768). Identify by a run so
  // we don't depend on output order (metadata dims aren't exposed until a run).
  const probe = await session.run({
    [inName]: new ort.Tensor("float32", new Float32Array(3 * IN_W * IN_H), [1, 3, IN_H, IN_W]),
  });
  xName = session.outputNames.find((n) => probe[n].dims[2] === IN_W * SPLIT);
  yName = session.outputNames.find((n) => probe[n].dims[2] === IN_H * SPLIT);
  post({ type: "ready", device: "wasm" });
}

// Estimate 133 keypoints from a person crop (RGBA Uint8ClampedArray at IN_W×IN_H) → [{x,y,c}] in input px.
async function estimate(id, rgba) {
  await ensureLoaded();
  const t0 = performance.now();
  const N = IN_W * IN_H;
  const a = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    a[i] = (rgba[i * 4] - MEAN[0]) / STD[0];
    a[N + i] = (rgba[i * 4 + 1] - MEAN[1]) / STD[1];
    a[2 * N + i] = (rgba[i * 4 + 2] - MEAN[2]) / STD[2];
  }
  const out = await session.run({ [inName]: new ort.Tensor("float32", a, [1, 3, IN_H, IN_W]) });
  const sx = out[xName], sy = out[yName];
  const LX = sx.dims[2], LY = sy.dims[2];
  const kpts = new Array(N_KPTS);
  for (let k = 0; k < N_KPTS; k++) {
    let bx = 0, vx = -1e9;
    for (let i = 0; i < LX; i++) {
      const v = sx.data[k * LX + i];
      if (v > vx) {
        vx = v;
        bx = i;
      }
    }
    let by = 0, vy = -1e9;
    for (let i = 0; i < LY; i++) {
      const v = sy.data[k * LY + i];
      if (v > vy) {
        vy = v;
        by = i;
      }
    }
    kpts[k] = { x: bx / SPLIT, y: by / SPLIT, c: Math.min(vx, vy) };
  }
  post({ type: "pose", id, kpts, w: IN_W, h: IN_H, ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "estimate") await estimate(e.data.id, e.data.crop);
  } catch (err) {
    console.error("[dwpose worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
