// YOLOv8-pose multi-model worker — a real two-model composition, off the main thread:
//   1. YOLOv8-pose (raw ONNX Runtime Web) finds every person + their 17 keypoints in one pass.
//   2. For each person we crop their box and run CLIP zero-shot image-classification (Transformers.js)
//      against a set of activity labels — so every skeleton gets a "what are they doing?" tag.
// Pose: Xenova/yolov8n-pose (~13 MB, fp32, raw ORT). CLIP: Xenova/clip-vit-base-patch16 (~340 MB, q8).
//
// This is the natural composition for a single-stage pose model: it already detects+poses everyone, so
// we chain its per-person crops into a second model that labels the action. Everything on-device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const POSE_ID = "Xenova/yolov8n-pose";
const POSE_URL = `https://huggingface.co/${POSE_ID}/resolve/main/onnx/model.onnx`;
const POSE_CACHE = "yolov8-pose-onnx-cache";
const CLIP_ID = "Xenova/clip-vit-base-patch16";
const NET = 640;

let ort = null, session = null, inputName = null, outputName = null;
let mod = null, classifier = null, RawImage = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function fetchPoseBytes() {
  const cache = await caches.open(POSE_CACHE);
  let resp = await cache.match(POSE_URL);
  if (!resp) {
    const net = await fetch(POSE_URL);
    if (!net.ok) throw new Error(`pose fetch failed (${net.status})`);
    const buf = await net.arrayBuffer();
    await cache.put(POSE_URL, new Response(buf));
    resp = await cache.match(POSE_URL);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (session && classifier) return;
  // Pose model (raw ORT).
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  post({ type: "progress", p: { status: "progress", progress: 5 } });
  const bytes = await fetchPoseBytes();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  // CLIP (transformers.js zero-shot).
  mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  mod.env.allowLocalModels = false;
  classifier = await mod.pipeline("zero-shot-image-classification", CLIP_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  const inter = w * h, areaA = (a[2] - a[0]) * (a[3] - a[1]), areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

async function run(id, bitmap, labels) {
  await ensureLoaded();
  const t0 = performance.now();
  const iw = bitmap.width, ih = bitmap.height;
  const scale = Math.min(NET / iw, NET / ih);
  const nw = Math.round(iw * scale), nh = Math.round(ih * scale);
  const padX = Math.floor((NET - nw) / 2), padY = Math.floor((NET - nh) / 2);

  // Full-res copy for cropping + a letterboxed copy for the network.
  const full = new OffscreenCanvas(iw, ih);
  full.getContext("2d").drawImage(bitmap, 0, 0);
  const oc = new OffscreenCanvas(NET, NET);
  const octx = oc.getContext("2d", { willReadFrequently: true });
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, NET, NET);
  octx.drawImage(bitmap, padX, padY, nw, nh);
  bitmap.close?.();
  const rgba = octx.getImageData(0, 0, NET, NET).data;
  const N = NET * NET, chw = new Float32Array(3 * N);
  for (let p = 0; p < N; p++) {
    chw[p] = rgba[p * 4] / 255;
    chw[N + p] = rgba[p * 4 + 1] / 255;
    chw[2 * N + p] = rgba[p * 4 + 2] / 255;
  }

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", chw, [1, 3, NET, NET]);
  const out = await session.run(feeds);
  const o = out[outputName];
  const na = o.dims[2], d = o.data;
  const poseMs = Math.round(performance.now() - t0);

  const cand = [];
  for (let i = 0; i < na; i++) {
    const conf = d[4 * na + i];
    if (conf < 0.3) continue;
    const cx = d[i], cy = d[na + i], w = d[2 * na + i], h = d[3 * na + i];
    const box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    const kx = new Float32Array(17), ky = new Float32Array(17), kv = new Float32Array(17);
    for (let k = 0; k < 17; k++) {
      kx[k] = d[(5 + k * 3) * na + i];
      ky[k] = d[(5 + k * 3 + 1) * na + i];
      kv[k] = d[(5 + k * 3 + 2) * na + i];
    }
    cand.push({ box, conf, kx, ky, kv });
  }
  cand.sort((a, b) => b.conf - a.conf);
  const keep = [];
  for (const c of cand) {
    if (keep.length >= 8) break;
    if (keep.every((k) => iou(c.box, k.box) <= 0.45)) keep.push(c);
  }

  const unpad = (x, pad) => (x - pad) / scale;
  const tc = performance.now();
  const persons = [];
  for (const c of keep) {
    const bx = Math.max(0, unpad(c.box[0], padX)), by = Math.max(0, unpad(c.box[1], padY));
    const bx2 = Math.min(iw, unpad(c.box[2], padX)), by2 = Math.min(ih, unpad(c.box[3], padY));
    const cw = Math.round(bx2 - bx), ch = Math.round(by2 - by);
    const keypoints = [], scores = [];
    for (let k = 0; k < 17; k++) {
      keypoints.push([unpad(c.kx[k], padX), unpad(c.ky[k], padY)]);
      scores.push(c.kv[k]);
    }
    let activity = null, actScore = 0;
    if (cw >= 16 && ch >= 16) {
      // crop the person box → RawImage → CLIP zero-shot
      const crop = new OffscreenCanvas(cw, ch);
      crop.getContext("2d").drawImage(full, bx, by, cw, ch, 0, 0, cw, ch);
      const cd = crop.getContext("2d").getImageData(0, 0, cw, ch);
      const img = new RawImage(new Uint8ClampedArray(cd.data), cw, ch, 4).rgb();
      const res = await classifier(img, labels);
      activity = res[0]?.label ?? null;
      actScore = res[0]?.score ?? 0;
    }
    persons.push({ box: [bx, by, cw, ch], score: c.conf, keypoints, scores, activity, actScore });
  }
  const clipMs = Math.round(performance.now() - tc);

  post({
    type: "result",
    id,
    persons,
    imageSize: [iw, ih],
    poseMs,
    clipMs,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.bitmap, e.data.labels);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
