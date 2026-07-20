// CRAFT text DETECTION worker — ALL inference + word-box decode off the main thread via raw ONNX
// Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5 registers no model class for CRAFT
// (Character-Region Awareness For Text detection) and there is no text-detection pipeline task. The
// ONNX is a bare exported graph (input "input", outputs "score_map"/"feature_map"), not a registered
// architecture, so we run it directly with onnxruntime-web and hand-write the two pieces a pipeline
// would own: (1) aspect-preserving resize-to-multiple-of-32 + ImageNet-normalise NCHW preprocessing,
// and (2) the CRAFT decode — threshold the REGION map (per-pixel "is this a character?") and the
// AFFINITY map (per-pixel "are two characters part of the same word?"), OR them into a word mask,
// label 8-connected components, and fit a minimum-area (rotated) rectangle to each. This is the
// isolated per-worker ORT-web escape hatch (like models/scene-text-detection/worker.js and
// models/face-embedding/worker.js) — onnxruntime-web is pinned HERE only, never in shared lib/webai.js.
//
// Model: inference4j/craft-mlt-25k (model.onnx, MIT, ~83 MB) — the CLOVA AI CRAFT detector (VGG16-BN
// backbone) as used by EasyOCR. Input: [1,3,H,W] NCHW. Output "score_map": [1, H/2, W/2, 2] —
// channel 0 = REGION score (character heat), channel 1 = AFFINITY score (character-link heat). We
// locate WHERE text is (word polygons) — we do NOT read it. Everything stays on-device: no image or
// box ever leaves the tab. CRAFT is a distinct architecture from the DBNet detector: it reasons about
// characters + the links between them, rather than a single differentiable-binarization map.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "inference4j/craft-mlt-25k";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/model.onnx`;
const CACHE_NAME = "craft-mlt-onnx-cache";

// ImageNet normalisation (CRAFT's normalizeMeanVariance).
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
// "/inference4j/craft-mlt-25k/") sees them → auto-init on a returning visit, honest Download on first
// visit, and the per-model "clear cache" control all work. Streams download progress.
async function fetchModelBytes() {
  const cache = await caches.open(CACHE_NAME);
  const resp = await cache.match(MODEL_URL);
  if (resp) return new Uint8Array(await resp.arrayBuffer());
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

async function ensureLoaded() {
  if (session) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1; // no cross-origin isolation on GitHub Pages → single-threaded WASM.
  const bytes = await fetchModelBytes();
  session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  inputName = session.inputNames[0];
  // "score_map" is the [1,H/2,W/2,2] region+affinity head; "feature_map" is the intermediate feature
  // tensor we don't use. Pick score_map explicitly.
  outputName = session.outputNames.includes("score_map") ? "score_map" : session.outputNames[0];
  post({ type: "ready", device });
}

// Resize an ImageBitmap so its longest side ≤ `limit`, both sides multiples of 32, then build a
// normalised NCHW Float32 tensor. Returns { data, nw, nh }.
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

// ── geometry helpers (shared with the DBNet decode) ────────────────────────────────────────────────
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
    const ux = (b.x - a.x) / len, uy = (b.y - a.y) / len;
    const vx = -uy, vy = ux;
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

// Expand a polygon outward from its centroid (CRAFT "enlarge") so the word box wraps the glyph strokes,
// not just the high-score core.
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

// ── CRAFT decode: region + affinity heatmaps → word polygons ────────────────────────────────────────
// text pixels = REGION > lowText; link pixels = AFFINITY > linkThresh. A word is a connected component
// of (text OR link) that contains at least one strong character core (REGION > textThreshold). The
// affinity map bridges the gaps between characters so the whole word becomes one component — that
// character-linking is exactly what makes CRAFT distinct from a DBNet probability map.
function decode(region, aff, mapW, mapH, params, sx, sy) {
  const { textThreshold, linkThreshold, lowText } = params;
  const N = mapW * mapH;
  const comb = new Uint8Array(N); // text OR link
  const textBin = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    const t = region[i] > lowText ? 1 : 0;
    const l = aff[i] > linkThreshold ? 1 : 0;
    textBin[i] = t;
    comb[i] = (t || l) ? 1 : 0;
  }
  const seen = new Uint8Array(N);
  const stack = new Int32Array(N);
  const regions = [];
  for (let start = 0; start < N; start++) {
    if (!comb[start] || seen[start]) continue;
    let sp = 0;
    stack[sp++] = start;
    seen[start] = 1;
    const pts = [];
    let maxRegion = 0;
    let regionSum = 0, regionCount = 0;
    while (sp > 0) {
      const idx = stack[--sp];
      const x = idx % mapW, y = (idx / mapW) | 0;
      pts.push({ x, y });
      if (region[idx] > maxRegion) maxRegion = region[idx];
      if (textBin[idx]) {
        regionSum += region[idx];
        regionCount++;
      }
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= mapH || nx < 0 || nx >= mapW) continue;
          const nidx = ny * mapW + nx;
          if (comb[nidx] && !seen[nidx]) {
            seen[nidx] = 1;
            stack[sp++] = nidx;
          }
        }
      }
    }
    if (pts.length < 10) continue; // drop specks
    // A real word must contain a confident character core; pure-affinity blobs are rejected.
    if (maxRegion < textThreshold) continue;
    let poly = minAreaRect(pts);
    if (!poly) continue;
    const w1 = Math.hypot(poly[1].x - poly[0].x, poly[1].y - poly[0].y);
    const h1 = Math.hypot(poly[3].x - poly[0].x, poly[3].y - poly[0].y);
    if (Math.min(w1, h1) < 2) continue;
    poly = unclip(poly, 1.35);
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
    const score = regionCount ? regionSum / regionCount : maxRegion;
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

function toHeat(arr, N) {
  const heat = new Uint8ClampedArray(N);
  for (let i = 0; i < N; i++) heat[i] = Math.round(Math.max(0, Math.min(1, arr[i])) * 255);
  return heat;
}

async function detect(id, bitmap, opts) {
  await ensureLoaded();
  const limit = opts?.limit || 800;
  const params = {
    textThreshold: opts?.textThreshold ?? 0.7,
    linkThreshold: opts?.linkThreshold ?? 0.4,
    lowText: opts?.lowText ?? 0.4,
  };
  const imgW = bitmap.width, imgH = bitmap.height;
  const t0 = performance.now();
  const { data, nw, nh } = preprocess(bitmap, limit);
  bitmap.close?.();
  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", data, [1, 3, nh, nw]);
  const results = await session.run(feeds);
  const out = results[outputName];
  const dims = out.dims; // [1, mapH, mapW, 2]
  const mapH = dims[1], mapW = dims[2];
  const raw = out.data;
  const M = mapW * mapH;
  const region = new Float32Array(M);
  const affinity = new Float32Array(M);
  // Layout: NHWC with last dim = 2 (region, affinity) interleaved.
  for (let i = 0; i < M; i++) {
    region[i] = raw[i * 2];
    affinity[i] = raw[i * 2 + 1];
  }
  const infMs = Math.round(performance.now() - t0);
  const regions = decode(region, affinity, mapW, mapH, params, imgW / mapW, imgH / mapH);
  const heat = toHeat(region, M);
  const heatAff = toHeat(affinity, M);
  const ms = Math.round(performance.now() - t0);
  post(
    {
      type: "result",
      id,
      regions,
      heat,
      heatAff,
      mapW,
      mapH,
      imgW,
      imgH,
      ms,
      infMs,
      device,
    },
    [heat.buffer, heatAff.buffer],
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
