// Scene text DETECTION worker — ALL inference + DB post-processing off the main thread via raw ONNX
// Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5 registers no model class for a PaddleOCR
// DB (Differentiable Binarization) text detector — the ONNX is a bare exported graph (input "x",
// output "sigmoid_0.tmp_0"), not a registered architecture, and there is no text-detection pipeline
// task. So we run the ONNX graph directly with onnxruntime-web and hand-write the two pieces a
// pipeline would own: (1) resize-to-multiple-of-32 + ImageNet-normalise NCHW preprocessing, and
// (2) the DB decode — threshold the probability map, label connected components, and fit a
// minimum-area oriented box to each text region. This is the isolated per-worker ORT-web escape hatch
// (like models/face-embedding/worker.js) — onnxruntime-web is pinned HERE only, never in shared
// lib/webai.js.
//
// Model: breezedeus/cnstd-ppocr-en_PP-OCRv3_det (en_PP-OCRv3_det_infer.onnx, Apache-2.0, ~2.4 MB).
// PaddleOCR PP-OCRv3 detection (DBNet-style). Input: [1,3,H,W] NCHW, H/W multiples of 32. Output:
// [1,1,H,W] per-pixel text PROBABILITY map. We locate WHERE text is (region boxes/polygons) — we do
// NOT read it. Everything stays on-device: no image or box ever leaves the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "breezedeus/cnstd-ppocr-en_PP-OCRv3_det";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/en_PP-OCRv3_det_infer.onnx`;
const CACHE_NAME = "ppocr-det-onnx-cache";

// ImageNet normalisation used by PaddleOCR detection.
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so the shared model-cache layer (which scans caches for
// "/breezedeus/cnstd-ppocr-en_PP-OCRv3_det/") sees them → auto-init on a returning visit, honest
// Download on first visit, and the per-model "clear cache" control all work. Streams download progress.
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

// Resize an ImageBitmap so its longest side ≤ `limit` and both sides are multiples of 32 (DBNet
// requirement), then build a normalised NCHW Float32 tensor. Returns { data, nw, nh }.
function preprocess(bitmap, limit) {
  const w = bitmap.width, h = bitmap.height;
  const scale = Math.min(limit / Math.max(w, h), 1);
  const nw = Math.max(32, Math.round((w * scale) / 32) * 32);
  const nh = Math.max(32, Math.round((h * scale) / 32) * 32);
  const canvas = new OffscreenCanvas(nw, nh);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, w, h, 0, 0, nw, nh);
  const { data } = ctx.getImageData(0, 0, nw, nh);
  const N = nw * nh;
  const out = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * N + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c]; // NCHW
    }
  }
  return { data: out, nw, nh };
}

// ── DB decode: probability map → oriented text-region polygons ──────────────────────────────────
// Label 8-connected components of the thresholded map, then fit a minimum-area (possibly rotated)
// rectangle to each so slanted text gets a tight box, not a loose axis-aligned one.

function convexHull(pts) {
  pts.sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

// Rotating-calipers minimum-area rectangle over a component's boundary points → 4 corners.
function minAreaRect(pts) {
  const hull = convexHull(pts);
  if (hull.length < 3) return null;
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i], b = hull[(i + 1) % hull.length];
    const len = Math.hypot(b.x - a.x, b.y - a.y) || 1;
    const ux = (b.x - a.x) / len, uy = (b.y - a.y) / len; // edge direction
    const vx = -uy, vy = ux; // normal
    let minu = Infinity, maxu = -Infinity, minv = Infinity, maxv = -Infinity;
    for (const p of hull) {
      const du = p.x * ux + p.y * uy, dv = p.x * vx + p.y * vy;
      if (du < minu) minu = du;
      if (du > maxu) maxu = du;
      if (dv < minv) minv = dv;
      if (dv > maxv) maxv = dv;
    }
    const area = (maxu - minu) * (maxv - minv);
    if (!best || area < best.area) best = { area, ux, uy, vx, vy, minu, maxu, minv, maxv };
  }
  const c = best;
  const corner = (u, v) => ({ x: c.ux * u + c.vx * v, y: c.uy * u + c.vy * v });
  return [
    corner(c.minu, c.minv),
    corner(c.maxu, c.minv),
    corner(c.maxu, c.maxv),
    corner(c.minu, c.maxv),
  ];
}

// Expand a polygon outward from its centroid (DB "unclip") so the box wraps the glyph strokes, not
// just the high-probability core.
function unclip(poly, ratio) {
  let cx = 0, cy = 0;
  for (const p of poly) {
    cx += p.x;
    cy += p.y;
  }
  cx /= poly.length;
  cy /= poly.length;
  return poly.map((p) => ({ x: cx + (p.x - cx) * ratio, y: cy + (p.y - cy) * ratio }));
}

function decode(prob, nw, nh, thresh, boxThresh, sx, sy) {
  const N = nw * nh;
  const bin = new Uint8Array(N);
  for (let i = 0; i < N; i++) bin[i] = prob[i] > thresh ? 1 : 0;
  const seen = new Uint8Array(N);
  const stack = new Int32Array(N);
  const regions = [];
  for (let start = 0; start < N; start++) {
    if (!bin[start] || seen[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    seen[start] = 1;
    const pts = [];
    let scoreSum = 0;
    while (sp > 0) {
      const idx = stack[--sp];
      const x = idx % nw, y = (idx / nw) | 0;
      pts.push({ x, y });
      scoreSum += prob[idx];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= nh || nx < 0 || nx >= nw) continue;
          const nidx = ny * nw + nx;
          if (bin[nidx] && !seen[nidx]) {
            seen[nidx] = 1;
            stack[sp++] = nidx;
          }
        }
      }
    }
    if (pts.length < 8) continue; // drop specks
    const score = scoreSum / pts.length;
    if (score < boxThresh) continue;
    let poly = minAreaRect(pts);
    if (!poly) continue;
    // side lengths (in model space) — drop degenerate slivers
    const w1 = Math.hypot(poly[1].x - poly[0].x, poly[1].y - poly[0].y);
    const h1 = Math.hypot(poly[3].x - poly[0].x, poly[3].y - poly[0].y);
    if (Math.min(w1, h1) < 3) continue;
    poly = unclip(poly, 1.6);
    // map to original image pixels
    const scaled = poly.map((p) => ({ x: p.x * sx, y: p.y * sy }));
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of scaled) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    const dxv = poly[1].x - poly[0].x, dyv = poly[1].y - poly[0].y;
    const angle = Math.atan2(dyv, dxv) * 180 / Math.PI;
    regions.push({
      points: scaled,
      box: { x0, y0, x1, y1 },
      score,
      angle: ((angle % 180) + 180) % 180,
      area: pts.length,
    });
  }
  regions.sort((a, b) => b.score - a.score);
  return regions;
}

async function detect(id, bitmap, opts) {
  await ensureLoaded();
  const limit = opts?.limit || 736;
  const thresh = opts?.thresh ?? 0.3;
  const boxThresh = opts?.boxThresh ?? 0.5;
  const imgW = bitmap.width, imgH = bitmap.height;
  const t0 = performance.now();
  const { data, nw, nh } = preprocess(bitmap, limit);
  bitmap.close?.();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", data, [1, 3, nh, nw]);
  const results = await session.run(feeds);
  const prob = results[outputName].data; // Float32Array(nw*nh)
  const infMs = Math.round(performance.now() - t0);
  const regions = decode(prob, nw, nh, thresh, boxThresh, imgW / nw, imgH / nh);
  // Downscale the probability map to a Uint8 heatmap buffer (transfer, zero-copy).
  const heat = new Uint8ClampedArray(nw * nh);
  for (let i = 0; i < prob.length; i++) {
    heat[i] = Math.round(Math.max(0, Math.min(1, prob[i])) * 255);
  }
  const ms = Math.round(performance.now() - t0);
  post(
    { type: "result", id, regions, heat, mapW: nw, mapH: nh, imgW, imgH, ms, infMs, device },
    [heat.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await detect(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
