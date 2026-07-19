// YOLO-World (open-vocabulary) detection worker — ALL inference off the main thread.
//
// Open-vocab detection means: detect ANY class you can NAME, with no fixed 80-COCO list. It is a
// TWO-MODEL composition, both running on-device:
//
//   1. A CLIP ViT-B/32 TEXT encoder (Transformers.js, Xenova/clip-vit-base-patch32, fp16) turns each
//      class name you type ("forklift", "traffic cone", "swan") into a 512-dim text embedding.
//   2. A YOLOv8-World-v2 vision graph (raw ONNX Runtime Web, Instemic/yolo-world-onnx) takes the image
//      PLUS those text embeddings as a dynamic `txt_feats` input and returns per-class boxes. The class
//      set is re-encoded at RUNTIME, so the vocabulary is whatever you type — nothing is baked in.
//
// Why raw ORT for the detector: transformers.js has no `yolo-world`/`yolov8` model class, so we run the
// ONNX graph directly (isolated version-pin escape hatch at the RUNTIME level — onnxruntime-web pinned
// in THIS worker only, never in shared lib/webai.js). The CLIP text tower uses the shared 3.7.5 pin.
//
// Detector export (Instemic/yolo-world-onnx · yolov8s-worldv2.onnx, 48.8 MB, AGPL-3.0):
//   inputs : images    float32[1,3,640,640]  (letterboxed, /255, RGB)
//            txt_feats  float32[1,N,512]      (L2-normalized CLIP ViT-B/32 text embeddings)
//   output : output0    float32[1,4+N,8400]   (channels 0-3 = box cx,cy,w,h in 640 space;
//                                              channels 4..4+N = per-class PROBABILITIES — the export
//                                              already applies sigmoid, so we do NOT sigmoid again)
// We apply per-class NMS and scale boxes back to the source image. Verified numerically against the
// cats sample (2×cat 0.92/0.91, remote 0.90) before build.

import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.5";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const YOLO_ID = "Instemic/yolo-world-onnx";
const YOLO_URL = `https://huggingface.co/${YOLO_ID}/resolve/main/yolov8s-worldv2.onnx`;
const YOLO_CACHE = "yolo-world-onnx-cache";
const CLIP_ID = "Xenova/clip-vit-base-patch32";
const SIZE = 640; // network input side
const FLOOR = 0.12; // run once at a low floor; the score slider filters the cached list upward.
const NMS_IOU = 0.5;

let ort = null;
let session = null;
let tokenizer = null;
let textModel = null;
let device = "wasm";
// Cache the last class-set's txt_feats so dragging the slider / re-detecting the SAME classes never
// re-encodes text. Re-encode only when the class list actually changes.
let cachedKey = null;
let cachedFeats = null; // { data: Float32Array, N }

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX detector THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/Instemic/yolo-world-onnx/") sees it → auto-init on return, honest Download first visit, working
// "clear cache" control. Streams download progress.
async function fetchYoloBytes() {
  const cache = await caches.open(YOLO_CACHE);
  let resp = await cache.match(YOLO_URL);
  if (!resp) {
    const net = await fetch(YOLO_URL);
    if (!net.ok || !net.body) throw new Error(`detector fetch failed (${net.status})`);
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
      YOLO_URL,
      new Response(buf, {
        headers: { "content-length": String(received), "content-type": "application/octet-stream" },
      }),
    );
    return buf;
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (session && textModel) return;
  // 1) CLIP text tower (Transformers.js manages its own Cache Storage under /Xenova/...).
  if (!textModel) {
    post({ type: "stage", stage: "text encoder" });
    tokenizer = await AutoTokenizer.from_pretrained(CLIP_ID);
    textModel = await CLIPTextModelWithProjection.from_pretrained(CLIP_ID, {
      dtype: "fp16",
      progress_callback: (p) => post({ type: "progress", p }),
    });
  }
  // 2) YOLO-World detector graph (raw ORT-web).
  if (!session) {
    post({ type: "stage", stage: "detector" });
    ort = await import(ORT_URL);
    ort.env.wasm.wasmPaths = ORT_WASM;
    ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-thread WASM.
    const bytes = await fetchYoloBytes();
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  }
  post({ type: "ready", device });
}

// Encode class names → L2-normalized txt_feats [1, N, 512]. Cached by the class-list key.
async function encodeClasses(classes) {
  const key = classes.join("");
  if (cachedKey === key && cachedFeats) return cachedFeats;
  const inputs = tokenizer(classes, { padding: true, truncation: true });
  const out = await textModel(inputs);
  const emb = out.text_embeds; // [N, 512]
  const N = emb.dims[0], D = emb.dims[1];
  const src = emb.data;
  const data = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let d = 0; d < D; d++) s += src[i * D + d] ** 2;
    const inv = 1 / (Math.sqrt(s) || 1);
    for (let d = 0; d < D; d++) data[i * D + d] = src[i * D + d] * inv;
  }
  cachedKey = key;
  cachedFeats = { data, N, D };
  return cachedFeats;
}

// Letterbox an ImageBitmap into a 640×640 CHW float tensor (top-left pad, /255, RGB — matches the
// Ultralytics export's preprocessing; boxes scale back with /scale, no offset).
function preprocess(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const scale = SIZE / Math.max(w, h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(bitmap, 0, 0, nw, nh);
  const { data } = ctx.getImageData(0, 0, SIZE, SIZE);
  const plane = SIZE * SIZE;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    chw[i] = data[i * 4] / 255;
    chw[plane + i] = data[i * 4 + 1] / 255;
    chw[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return { chw, scale, w, h };
}

function iou(a, b) {
  const x1 = Math.max(a.box.xmin, b.box.xmin), y1 = Math.max(a.box.ymin, b.box.ymin);
  const x2 = Math.min(a.box.xmax, b.box.xmax), y2 = Math.min(a.box.ymax, b.box.ymax);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a.box.xmax - a.box.xmin) * (a.box.ymax - a.box.ymin);
  const areaB = (b.box.xmax - b.box.xmin) * (b.box.ymax - b.box.ymin);
  const u = areaA + areaB - inter;
  return u > 0 ? inter / u : 0;
}

// Decode [1, 4+N, anchors] → per-class NMS'd detections in SOURCE pixel coords. Class channels are
// already probabilities (do NOT sigmoid). One box gets the argmax class over the N prompts.
function decode(tensor, classes, scale, w, h) {
  const arr = tensor.data;
  const N = classes.length;
  const anchors = tensor.dims[2];
  const cand = [];
  for (let a = 0; a < anchors; a++) {
    let best = FLOOR, bestC = -1;
    for (let c = 0; c < N; c++) {
      const v = arr[(4 + c) * anchors + a];
      if (v > best) {
        best = v;
        bestC = c;
      }
    }
    if (bestC < 0) continue;
    const cx = arr[a] / scale, cy = arr[anchors + a] / scale;
    const bw = arr[2 * anchors + a] / scale, bh = arr[3 * anchors + a] / scale;
    cand.push({
      label: classes[bestC],
      classId: bestC,
      score: best,
      box: {
        xmin: Math.max(0, cx - bw / 2),
        ymin: Math.max(0, cy - bh / 2),
        xmax: Math.min(w, cx + bw / 2),
        ymax: Math.min(h, cy + bh / 2),
      },
    });
  }
  // Per-class NMS.
  cand.sort((x, y) => y.score - x.score);
  const keep = [];
  for (const d of cand) {
    let ok = true;
    for (const k of keep) {
      if (k.classId === d.classId && iou(k, d) > NMS_IOU) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(d);
  }
  return keep;
}

async function run(id, bitmap, classes) {
  await ensureLoaded();
  const tText0 = performance.now();
  const feats = await encodeClasses(classes);
  const textMs = Math.round(performance.now() - tText0);

  const tDet0 = performance.now();
  const { chw, scale, w, h } = preprocess(bitmap);
  bitmap.close?.();
  const feeds = {
    images: new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]),
    txt_feats: new ort.Tensor("float32", feats.data, [1, feats.N, feats.D]),
  };
  const results = await session.run(feeds);
  const detections = decode(results[session.outputNames[0]], classes, scale, w, h);
  const detMs = Math.round(performance.now() - tDet0);

  post({
    type: "result",
    id,
    detections,
    classes,
    textMs,
    detMs,
    ms: textMs + detMs,
    device,
    imgW: w,
    imgH: h,
    reusedText: cachedKey === classes.join("") && textMs < 3,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.bitmap, e.data.classes);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
