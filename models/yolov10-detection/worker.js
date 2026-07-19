// YOLOv10 object-detection worker — ALL inference off the main thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5 has no `yolov10` model class
// (`pipeline("object-detection","onnx-community/yolov10n")` throws "Unsupported model type: yolov10").
// So we run the ONNX graph directly with onnxruntime-web, hand-writing the two pieces transformers.js
// would normally own: (1) letterbox preprocessing to 640×640, and (2) decoding YOLOv10's one-to-one
// head. This is the isolated version-pin escape hatch applied to the runtime itself — pinned here in
// THIS worker only (onnxruntime-web@1.21.0), never in shared lib/webai.js.
//
// The point of YOLOv10 (vs DETR / YOLOS / RT-DETR): it is NMS-FREE. Older detectors emit thousands of
// overlapping boxes and rely on Non-Max-Suppression to dedupe them. YOLOv10 trains a "one-to-one" head
// that emits at most one box per object, already deduped, so the ONNX graph returns a fixed [1, 300, 6]
// tensor — top-300 detections, each [x1, y1, x2, y2, score, classId] in 640-letterbox space, sorted by
// score. No NMS pass, lower latency. We just scale the boxes back to the source image.
//
// Model: onnx-community/yolov10n (model.onnx, fp32, ~9 MB). 80 COCO classes.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "onnx-community/yolov10n";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model.onnx`;
const CACHE_NAME = "yolov10-onnx-cache";
const SIZE = 640; // network input side

// COCO 80 classes, in class-id order (index === classId the model emits).
const COCO = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so the shared model-cache layer (which scans caches for
// "/onnx-community/yolov10n/") sees them → auto-init on a returning visit, honest Download on first
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
    // Store a fresh Response (with content-length) so future visits are download-free + cache-scannable.
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
  post({ type: "ready", device });
}

// Letterbox an ImageBitmap into a 640×640 CHW float tensor (rescale 1/255, no mean/std — matches the
// repo's preprocessor_config: do_rescale, do_normalize=false, longest_edge 640, top-left pad).
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
    chw[i] = data[i * 4] / 255; // R
    chw[plane + i] = data[i * 4 + 1] / 255; // G
    chw[2 * plane + i] = data[i * 4 + 2] / 255; // B
  }
  return { chw, scale, w, h };
}

// Decode the [1, N, 6] one-to-one head → detections in the SOURCE image's pixel coordinates.
function decode(tensor, scale, w, h, floor) {
  const arr = tensor.data;
  const N = tensor.dims[1], stride = tensor.dims[2]; // stride === 6
  const out = [];
  for (let i = 0; i < N; i++) {
    const score = arr[i * stride + 4];
    if (score < floor) continue; // rows are score-sorted; could break, but N is only 300.
    const cls = arr[i * stride + 5] | 0;
    const xmin = Math.max(0, Math.min(w, arr[i * stride] / scale));
    const ymin = Math.max(0, Math.min(h, arr[i * stride + 1] / scale));
    const xmax = Math.max(0, Math.min(w, arr[i * stride + 2] / scale));
    const ymax = Math.max(0, Math.min(h, arr[i * stride + 3] / scale));
    out.push({
      label: COCO[cls] || String(cls),
      classId: cls,
      score,
      box: { xmin, ymin, xmax, ymax },
    });
  }
  return out;
}

const FLOOR = 0.05; // run at a low floor; the UI slider filters the cached list upward, no re-run.

async function run(id, bitmap) {
  await ensureLoaded();
  const t0 = performance.now();
  const { chw, scale, w, h } = preprocess(bitmap);
  bitmap.close?.();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]);
  const results = await session.run(feeds);
  const outName = session.outputNames[0];
  const detections = decode(results[outName], scale, w, h, FLOOR);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, detections, ms, device, imgW: w, imgH: h });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.bitmap);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
