// YOLOv8-pose worker — ALL inference AND the letterbox preprocessing off the main thread via raw
// ONNX Runtime Web. One forward pass returns EVERY person in the frame: a box, an objectness score,
// and 17 COCO keypoints (x, y, visibility) — the network is a SINGLE-STAGE detector+pose head, so it
// finds and poses everyone at once (a different design from the top-down ViTPose demo, which needs a
// person box per crop, and from MediaPipe's BlazePose).
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5/4.2 registers no `yolov8`/`yolo` model
// class, and there is no single-stage pose pipeline task — `pipeline(...)` throws "Unsupported model
// type". So we run the ONNX graph directly with onnxruntime-web and hand-write the two pieces a
// pipeline would own: (1) letterbox to 640x640 (aspect-preserving pad, /255, CHW, RGB) and (2) decode
// the [1, 56, 8400] head — 4 box + 1 conf + 51 keypoint values per anchor — filter by confidence,
// run Non-Max-Suppression, and map boxes+keypoints back to source pixels. This is the isolated
// per-worker ORT-web escape hatch (precedent: models/yolov10-detection, models/ddcolor-*): onnxruntime-
// web is pinned HERE only, never in shared lib/webai.js.
//
// Model: Xenova/yolov8n-pose (onnx/model.onnx, fp32, ~13 MB, AGPL-3.0 — same YOLO family/license as
// the built yolov10 + yolo11 detection demos). Everything stays on-device: the image never leaves the
// tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "Xenova/yolov8n-pose";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/onnx/model.onnx`;
const CACHE_NAME = "yolov8-pose-onnx-cache";
const NET = 640; // fixed network input side

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/Xenova/yolov8n-pose/") sees them → auto-init on a returning visit, honest Download on first visit,
// and the per-model "clear cache" control all work. Streams download progress to the shared loader.
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
    const blob = new Blob(chunks);
    await cache.put(
      MODEL_URL,
      new Response(blob, { headers: { "content-length": String(received) } }),
    );
    resp = await cache.match(MODEL_URL);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  const bytes = await fetchModelBytes();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  device = "wasm";
  post({ type: "ready", device });
}

// IoU of two [x1,y1,x2,y2] boxes.
function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const w = Math.max(0, x2 - x1), h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter + 1e-6);
}

async function run(id, bitmap, opts) {
  await ensureLoaded();
  const t0 = performance.now();
  const iw = bitmap.width, ih = bitmap.height;
  const confThr = opts?.conf ?? 0.25;
  const iouThr = opts?.iou ?? 0.45;
  const maxPeople = opts?.maxPeople ?? 20;

  // 1) Letterbox to 640x640 (aspect-preserving), grey pad — do it in the worker on an OffscreenCanvas.
  const scale = Math.min(NET / iw, NET / ih);
  const nw = Math.round(iw * scale), nh = Math.round(ih * scale);
  const padX = Math.floor((NET - nw) / 2), padY = Math.floor((NET - nh) / 2);
  const oc = new OffscreenCanvas(NET, NET);
  const octx = oc.getContext("2d", { willReadFrequently: true });
  octx.fillStyle = "rgb(114,114,114)";
  octx.fillRect(0, 0, NET, NET);
  octx.drawImage(bitmap, padX, padY, nw, nh);
  bitmap.close?.();
  const rgba = octx.getImageData(0, 0, NET, NET).data;
  const N = NET * NET;
  const chw = new Float32Array(3 * N);
  for (let p = 0; p < N; p++) {
    chw[p] = rgba[p * 4] / 255;
    chw[N + p] = rgba[p * 4 + 1] / 255;
    chw[2 * N + p] = rgba[p * 4 + 2] / 255;
  }

  // 2) Inference.
  const tInf = performance.now();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", chw, [1, 3, NET, NET]);
  const out = await session.run(feeds);
  const o = out[outputName]; // [1, 56, 8400]
  const infMs = Math.round(performance.now() - tInf);

  const dims = o.dims; // [1, 56, na]
  const ch = dims[1], na = dims[2];
  const d = o.data;
  const nk = (ch - 5) / 3; // 17 keypoints

  // 3) Decode: gather candidates over confThr, in 640-letterbox space.
  const cand = [];
  for (let i = 0; i < na; i++) {
    const conf = d[4 * na + i];
    if (conf < confThr) continue;
    const cx = d[0 * na + i], cy = d[1 * na + i], w = d[2 * na + i], h = d[3 * na + i];
    const box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2];
    const kx = new Float32Array(nk), ky = new Float32Array(nk), kv = new Float32Array(nk);
    for (let k = 0; k < nk; k++) {
      kx[k] = d[(5 + k * 3) * na + i];
      ky[k] = d[(5 + k * 3 + 1) * na + i];
      kv[k] = d[(5 + k * 3 + 2) * na + i];
    }
    cand.push({ box, conf, kx, ky, kv });
  }
  cand.sort((a, b) => b.conf - a.conf);

  // 4) Non-Max-Suppression (single "person" class).
  const keep = [];
  for (const c of cand) {
    if (keep.length >= maxPeople) break;
    let ok = true;
    for (const k of keep) {
      if (iou(c.box, k.box) > iouThr) {
        ok = false;
        break;
      }
    }
    if (ok) keep.push(c);
  }

  // 5) Map boxes + keypoints from 640-letterbox space back to source pixels.
  const unpad = (x, pad) => (x - pad) / scale;
  const persons = keep.map((c) => {
    const bx = unpad(c.box[0], padX), by = unpad(c.box[1], padY);
    const bx2 = unpad(c.box[2], padX), by2 = unpad(c.box[3], padY);
    const keypoints = [], scores = [];
    for (let k = 0; k < nk; k++) {
      keypoints.push([unpad(c.kx[k], padX), unpad(c.ky[k], padY)]);
      scores.push(c.kv[k]);
    }
    return {
      box: [bx, by, bx2 - bx, by2 - by],
      score: c.conf,
      keypoints,
      scores,
    };
  });

  // 6) "See inside" — the objectness field over the largest anchor grid (stride 8 → 80x80 = the first
  //    6400 anchors), so you can literally see WHERE the network detects person-ness before NMS.
  const GRID = 80;
  const gridConf = new Float32Array(GRID * GRID);
  for (let i = 0; i < GRID * GRID && i < na; i++) gridConf[i] = d[4 * na + i];

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    persons,
    imageSize: [iw, ih],
    gridConf,
    gridSize: GRID,
    candidates: cand.length,
    ms,
    infMs,
    device,
  }, [gridConf.buffer]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
