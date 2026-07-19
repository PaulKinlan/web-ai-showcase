// YOLO11 object-detection worker — ALL inference off the main thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js has no `yolo11` model class
// (`pipeline("object-detection", …)` throws "Unsupported model type: yolo11"). So we run the ONNX
// graph directly with onnxruntime-web, hand-writing the two pieces a pipeline would own: (1) letterbox
// preprocessing to 640×640, and (2) decoding YOLO11's detection head. This is the isolated
// version-pin escape hatch applied to the runtime itself — pinned here in THIS worker only
// (onnxruntime-web@1.21.0), never in shared lib/webai.js. Precedent: models/yolov10-detection/worker.js.
//
// The point of YOLO11 vs YOLOv10: YOLO11 is the OPPOSITE design. YOLOv10 trains an NMS-free one-to-one
// head and hands back a tidy [1, 300, 6] tensor already deduplicated. YOLO11 keeps the classic dense
// head: the graph returns a raw [1, 84, 8400] tensor — 8400 candidate anchors, each with 4 box coords
// (cx, cy, w, h in 640-letterbox pixels) and 80 class scores — and it is FULL OF near-duplicate boxes.
// To turn that into usable detections you must (a) argmax the 80 class scores per anchor, (b) threshold,
// and (c) run Non-Max-Suppression to collapse the duplicates. This worker does (a)+(b) and returns the
// raw candidate list; the NMS step (c) lives client-side in yolo11.js so the page can make it
// interactive (slide the IoU threshold and watch duplicates collapse) — the whole story of the demo.
//
// Model: webnn/yolo11n (onnx/yolo11n.onnx, fp32, ~10 MB). 80 COCO classes.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "webnn/yolo11n";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/yolo11n.onnx`;
const CACHE_NAME = "yolo11-onnx-cache";
const SIZE = 640; // network input side

// COCO 80 classes, in class-id order (index === classId).
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
// "/webnn/yolo11n/") sees them → auto-init on a returning visit, honest Download on first visit, and the
// per-model "clear cache" control all work. Streams download progress.
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
  post({ type: "ready", device });
}

// Letterbox an ImageBitmap into a 640×640 CHW float tensor (rescale 1/255, RGB). Ultralytics-style:
// scale longest edge to 640, pad the rest with gray 114, image at top-left so decode is a plain /scale.
function preprocess(bitmap) {
  const w = bitmap.width, h = bitmap.height;
  const scale = SIZE / Math.max(w, h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const canvas = new OffscreenCanvas(SIZE, SIZE);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
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

// Decode the raw [1, 84, 8400] dense head → candidate detections in the SOURCE image's pixel
// coordinates. Layout is channel-major: value(channel c, anchor a) = data[c * 8400 + a]. Channels
// 0..3 are cx, cy, w, h (640-letterbox pixels, already DFL-decoded + anchor-added by the export);
// channels 4..83 are the 80 per-class scores (already sigmoid-activated). We take the best class per
// anchor and keep anything above the low floor — deliberately NOT deduplicated. NMS happens later,
// client-side, so the raw duplicate flood is inspectable.
function decode(tensor, scale, w, h, floor) {
  const arr = tensor.data;
  const nc = tensor.dims[1]; // 84
  const na = tensor.dims[2]; // 8400
  const numClasses = nc - 4; // 80
  const out = [];
  for (let a = 0; a < na; a++) {
    let best = 0, bestScore = -1;
    for (let c = 0; c < numClasses; c++) {
      const s = arr[(4 + c) * na + a];
      if (s > bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (bestScore < floor) continue;
    const cx = arr[a], cy = arr[na + a], bw = arr[2 * na + a], bh = arr[3 * na + a];
    const xmin = Math.max(0, Math.min(w, (cx - bw / 2) / scale));
    const ymin = Math.max(0, Math.min(h, (cy - bh / 2) / scale));
    const xmax = Math.max(0, Math.min(w, (cx + bw / 2) / scale));
    const ymax = Math.max(0, Math.min(h, (cy + bh / 2) / scale));
    out.push({
      label: COCO[best] || String(best),
      classId: best,
      score: bestScore,
      box: { xmin, ymin, xmax, ymax },
    });
  }
  out.sort((p, q) => q.score - p.score);
  return out;
}

const FLOOR = 0.05; // run at a low floor; the UI filters + runs NMS on the cached candidates, no re-run.

async function run(id, bitmap) {
  await ensureLoaded();
  const t0 = performance.now();
  const { chw, scale, w, h } = preprocess(bitmap);
  bitmap.close?.();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]);
  const results = await session.run(feeds);
  const outName = session.outputNames[0];
  const raw = decode(results[outName], scale, w, h, FLOOR);
  const ms = Math.round(performance.now() - t0);
  // rawDetections = candidates above floor BEFORE NMS (the duplicate flood); client runs NMS.
  post({ type: "result", id, rawDetections: raw, ms, device, imgW: w, imgH: h });
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
