// ALIKED + LightGlue multi-model worker — a real two-stage image-retrieval composition, off the main
// thread: the standard "retrieve then verify" pipeline used in visual place recognition.
//   1. DINOv2 (Transformers.js image-feature-extraction) embeds the query + every gallery image and
//      ranks them by cosine similarity — a fast GLOBAL shortlist ("which look alike?").
//   2. ALIKED + LightGlue (raw ONNX Runtime Web) geometrically VERIFIES the top candidate — do concrete
//      local features actually correspond? — turning a soft similarity into a hard yes/no.
// DINOv2: Xenova/dinov2-small (~90 MB). ALIKED + LightGlue: bukuroo/ALIKED-LightGlue-ONNX (~51 MB).
// Everything on-device.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MATCH_ID = "bukuroo/ALIKED-LightGlue-ONNX";
const mbase = `https://huggingface.co/${MATCH_ID}/resolve/main/`;
const CACHE_NAME = "aliked-lightglue-onnx-cache";
const DINO_ID = "Xenova/dinov2-small";
const NET = 640;

let ort = null, extSess = null, lgSess = null;
let mod = null, embedder = null, device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function fetchBytes(file) {
  const cache = await caches.open(CACHE_NAME);
  const url = mbase + file;
  let resp = await cache.match(url);
  if (!resp) {
    const net = await fetch(url);
    if (!net.ok) throw new Error(`fetch failed ${file} (${net.status})`);
    const buf = await net.arrayBuffer();
    await cache.put(url, new Response(buf));
    resp = await cache.match(url);
  }
  return new Uint8Array(await resp.arrayBuffer());
}

async function ensureLoaded() {
  if (extSess && lgSess && embedder) return;
  mod = await import(TRANSFORMERS_URL);
  mod.env.allowLocalModels = false;
  embedder = await mod.pipeline("image-feature-extraction", DINO_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  ort = await import(ORT_URL);
  ort.env.wasm.wasmPaths = ORT_WASM;
  ort.env.wasm.numThreads = 1;
  post({ type: "progress", p: { status: "progress", progress: 60 } });
  extSess = await ort.InferenceSession.create(await fetchBytes("aliked-n16rot-top1k-640.onnx"), {
    executionProviders: ["wasm"],
  });
  post({ type: "progress", p: { status: "progress", progress: 75 } });
  lgSess = await ort.InferenceSession.create(await fetchBytes("lightglue_for_aliked.onnx"), {
    executionProviders: ["wasm"],
  });
  device = "wasm";
  post({ type: "ready", device });
}

function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

async function embed(bitmap) {
  const ri = await bitmapToRawImage(bitmap);
  const out = await embedder(ri);
  const dims = out.dims, dim = dims.at(-1), seq = dims.length === 3 ? dims[1] : 1;
  const data = out.data;
  // mean-pool the token embeddings → a single global descriptor
  const v = new Float64Array(dim);
  for (let t = 0; t < seq; t++) for (let k = 0; k < dim; k++) v[k] += data[t * dim + k];
  for (let k = 0; k < dim; k++) v[k] /= seq;
  return v;
}
async function bitmapToRawImage(bitmap) {
  const s = 224;
  const oc = new OffscreenCanvas(s, s);
  const cx = oc.getContext("2d", { willReadFrequently: true });
  cx.drawImage(bitmap, 0, 0, s, s);
  const d = cx.getImageData(0, 0, s, s);
  return new mod.RawImage(new Uint8ClampedArray(d.data), s, s, 4).rgb();
}

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
async function verify(bmA, bmB) {
  const a = await extSess.run({ [extSess.inputNames[0]]: toTensor(bmA) });
  const b = await extSess.run({ [extSess.inputNames[0]]: toTensor(bmB) });
  const B = (tn) => new ort.Tensor(tn.type, tn.data, [1, ...tn.dims]);
  const out = await lgSess.run({
    kpts0: B(a.keypoints),
    kpts1: B(b.keypoints),
    desc0: B(a.descriptors),
    desc1: B(b.descriptors),
  });
  const M = out.matches0.dims[0];
  const ms = out.mscores0 ? out.mscores0.data : null;
  let strong = 0;
  for (let i = 0; i < M; i++) if (!ms || Number(ms[i]) >= 0.3) strong++;
  return { matches: M, strong };
}

async function run(id, queryBitmap, galleryBitmaps) {
  await ensureLoaded();
  const t0 = performance.now();
  const qv = await embed(queryBitmap);
  const ranked = [];
  const galleryCopies = [];
  for (let i = 0; i < galleryBitmaps.length; i++) {
    // need the bitmap twice (embed + possibly verify); embed consumes nothing (RawImage copy)
    const gv = await embed(galleryBitmaps[i]);
    ranked.push({ index: i, sim: cosine(qv, gv) });
  }
  ranked.sort((a, b) => b.sim - a.sim);
  const retrMs = Math.round(performance.now() - t0);
  post({ type: "ranked", id, ranked });

  // Verify the top candidate geometrically.
  const top = ranked[0];
  const tv = performance.now();
  const v = await verify(queryBitmap, galleryBitmaps[top.index]);
  const verifyMs = Math.round(performance.now() - tv);

  post({
    type: "done",
    id,
    ranked,
    top: top.index,
    topSim: top.sim,
    verify: v,
    verified: v.strong >= 20,
    retrMs,
    verifyMs,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.query, e.data.gallery);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
