// ALIKED + LightGlue local-feature MATCHING worker — ALL inference AND preprocessing off the main
// thread via raw ONNX Runtime Web. Two forward passes of ALIKED (one per image) detect + describe up
// to 1000 local keypoints each; one forward pass of LightGlue (a learned attention-based matcher) finds
// the correspondences — the pairs of keypoints that land on the SAME physical point across the two
// photos. This is a distinct capability from the pose/landmark demos: it answers "are these two images
// of the same scene, and where do they overlap?".
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5 exposes no keypoint-detection PIPELINE and
// no learned-matcher class, so we run the two ONNX graphs directly with onnxruntime-web and hand-write
// the glue a pipeline would own: (1) resize each image to the fixed 640x640 ALIKED input (RGB, /255,
// CHW), (2) add the batch dim LightGlue expects, and (3) map the normalized keypoints back to pixels.
// Isolated per-worker ORT-web pin (precedent: models/yolov8-pose, models/ddcolor-*): onnxruntime-web is
// pinned HERE only, never in shared lib/webai.js.
//
// Models (bukuroo/ALIKED-LightGlue-ONNX, all fp32): aliked-n16rot-top1k-640.onnx (~6 MB — ALIKED-N16rot,
// rotation-robust, top-1000 keypoints, 640px) + lightglue_for_aliked.onnx (~45 MB). ALIKED input
// "image" [1,3,640,640]; outputs keypoints [1000,2] (normalized to [-1,1]), descriptors [1000,128],
// scores [1000]. LightGlue inputs kpts0/kpts1 [1,N,2] + desc0/desc1 [1,N,128]; outputs matches0 [M,2]
// (int64 index pairs into kpts0/kpts1) + mscores0 [M]. Everything stays on-device: the images never
// leave the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "bukuroo/ALIKED-LightGlue-ONNX";
const EXT_FILE = "aliked-n16rot-top1k-640.onnx";
const LG_FILE = "lightglue_for_aliked.onnx";
const base = `https://huggingface.co/${MODEL_ID}/resolve/main/`;
const CACHE_NAME = "aliked-lightglue-onnx-cache";
const NET = 640; // fixed ALIKED input side

let ort = null;
let extSess = null;
let lgSess = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch an ONNX file THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/bukuroo/ALIKED-LightGlue-ONNX/") sees it -> auto-init on a returning visit, honest Download on
// first visit, per-model clear-cache. Streams combined download progress.
async function fetchBytes(file, weight, offset) {
  const cache = await caches.open(CACHE_NAME);
  const url = base + file;
  let resp = await cache.match(url);
  if (!resp) {
    const net = await fetch(url);
    if (!net.ok || !net.body) throw new Error(`fetch failed for ${file} (${net.status})`);
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
        const pct = offset + (received / total) * weight;
        post({ type: "progress", p: { status: "progress", progress: pct } });
      }
    }
    await cache.put(
      url,
      new Response(new Blob(chunks), { headers: { "content-length": String(received) } }),
    );
    resp = await cache.match(url);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (extSess && lgSess) return;
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  // Extractor first (~6 MB → weight 12%), then the matcher (~45 MB → weight 88%).
  const eb = await fetchBytes(EXT_FILE, 12, 0);
  extSess = await ort.InferenceSession.create(eb, { executionProviders: ["wasm"] });
  const lb = await fetchBytes(LG_FILE, 88, 12);
  lgSess = await ort.InferenceSession.create(lb, { executionProviders: ["wasm"] });
  device = "wasm";
  post({ type: "ready", device });
}

// Resize a bitmap to 640x640 (stretch) and build the CHW float32 tensor ALIKED expects.
function toTensor(bitmap) {
  const oc = new OffscreenCanvas(NET, NET);
  const cx = oc.getContext("2d", { willReadFrequently: true });
  cx.drawImage(bitmap, 0, 0, NET, NET);
  const d = cx.getImageData(0, 0, NET, NET).data;
  const N = NET * NET;
  const chw = new Float32Array(3 * N);
  for (let p = 0; p < N; p++) {
    chw[p] = d[p * 4] / 255;
    chw[N + p] = d[p * 4 + 1] / 255;
    chw[2 * N + p] = d[p * 4 + 2] / 255;
  }
  return new ort.Tensor("float32", chw, [1, 3, NET, NET]);
}

// Run ALIKED on one image. Returns { keypoints (Tensor [K,2]), descriptors (Tensor [K,128]),
// scores (Float32Array), pxKeypoints ([[x,y]] in 640-space), K }.
async function extract(bitmap) {
  const t = toTensor(bitmap);
  const r = await extSess.run({ [extSess.inputNames[0]]: t });
  const kp = r.keypoints, desc = r.descriptors, sc = r.scores;
  const K = kp.dims[0];
  const px = new Array(K);
  const kd = kp.data;
  for (let i = 0; i < K; i++) {
    // normalized [-1,1] over the 640 input -> pixel in 640-space
    px[i] = [(kd[i * 2] + 1) * NET / 2, (kd[i * 2 + 1] + 1) * NET / 2];
  }
  return { keypoints: kp, descriptors: desc, scores: sc, pxKeypoints: px, K };
}

async function run(id, bitmapA, bitmapB, opts) {
  await ensureLoaded();
  const t0 = performance.now();
  const wA = bitmapA.width, hA = bitmapA.height;
  const wB = bitmapB.width, hB = bitmapB.height;
  const fA = await extract(bitmapA);
  const fB = await extract(bitmapB);
  const extMs = Math.round(performance.now() - t0);

  // LightGlue expects a batch dim on every input.
  const B = (tn) => new ort.Tensor(tn.type, tn.data, [1, ...tn.dims]);
  const tm = performance.now();
  const out = await lgSess.run({
    kpts0: B(fA.keypoints),
    kpts1: B(fB.keypoints),
    desc0: B(fA.descriptors),
    desc1: B(fB.descriptors),
  });
  const matchMs = Math.round(performance.now() - tm);

  const matchesT = out.matches0, scoresT = out.mscores0;
  const M = matchesT.dims[0];
  const md = matchesT.data; // int64 (BigInt) index pairs
  const ms = scoresT ? scoresT.data : null;
  const minScore = opts?.minScore ?? 0;

  // Map matched keypoints to EACH image's native pixel space (we stretched to 640, so scale back).
  const sxA = wA / NET, syA = hA / NET, sxB = wB / NET, syB = hB / NET;
  const pairs = [];
  for (let i = 0; i < M; i++) {
    const a = Number(md[i * 2]), b = Number(md[i * 2 + 1]);
    const score = ms ? Number(ms[i]) : 1;
    if (score < minScore) continue;
    const pa = fA.pxKeypoints[a], pb = fB.pxKeypoints[b];
    pairs.push({
      a: [pa[0] * sxA, pa[1] * syA],
      b: [pb[0] * sxB, pb[1] * syB],
      score,
    });
  }

  // All detected keypoints (native px) for the "see inside" overlay.
  const allA = fA.pxKeypoints.map((p) => [p[0] * sxA, p[1] * syA]);
  const allB = fB.pxKeypoints.map((p) => [p[0] * sxB, p[1] * syB]);

  bitmapA.close?.();
  bitmapB.close?.();
  post({
    type: "result",
    id,
    pairs,
    keypointsA: allA,
    keypointsB: allB,
    numKeypointsA: fA.K,
    numKeypointsB: fB.K,
    numMatches: pairs.length,
    rawMatches: M,
    sizeA: [wA, hA],
    sizeB: [wB, hB],
    extMs,
    matchMs,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.bitmapA, e.data.bitmapB, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
