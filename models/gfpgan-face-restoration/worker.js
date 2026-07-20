// GFPGAN face-restoration worker — ALL inference AND the dense output composite off the main thread via
// raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js has no GFPGAN (StyleGAN2-based restorer) model
// class and no face-restoration pipeline task. The ONNX export is a clean image→image graph, so we run
// it directly with onnxruntime-web and hand-write the pre/post: (1) resize the input face to the
// network's fixed 512×512 and pack it as [1,3,512,512] float32 normalised to [-1,1] (GFPGAN's
// convention: (x/255 − 0.5) / 0.5 — NOT plain [0,1]), and (2) map the restored tensor back from [-1,1]
// to an RGBA image, resized to the display aspect. This is the isolated per-worker ORT-web escape hatch
// (like NAFNet / DDColor) — onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: Neus/GFPGANv1.4 (GFPGANv1.4.onnx, ~340 MB) — an ONNX export of GFPGAN v1.4 (Wang et al., CVPR
// 2021), the blind face-restoration network. Input "input" [1,3,512,512] float32 in [-1,1]; output
// [1,3,512,512] float32 in [-1,1] (output node is numerically named, read via session.outputNames[0]).
// GFPGAN weights are Apache-2.0 upstream (TencentARC/GFPGAN). Everything stays on-device: the image
// never leaves the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "Neus/GFPGANv1.4";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/GFPGANv1.4.onnx`;
const CACHE_NAME = "gfpgan-onnx-cache";
const NET = 512; // fixed network input size

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/Neus/GFPGANv1.4/") sees them → auto-init on a returning visit, honest Download on first visit, and
// the per-model "clear cache" control all work. Streams download progress.
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

// Laplacian variance of channel 0 of a [3,H,W] planar buffer (values in [0,1]) — the standard sharpness
// / detail measure. A degraded face has little high-frequency detail; restoration adds it.
function sharpness(plane, w, h) {
  let s = 0, ss = 0, n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const lap = 4 * plane[i] - plane[i - 1] - plane[i + 1] - plane[i - w] - plane[i + w];
      s += lap;
      ss += lap * lap;
      n++;
    }
  }
  const m = s / n;
  return ss / n - m * m;
}

async function restore(id, bitmap, opts) {
  await ensureLoaded();
  const iw = bitmap.width, ih = bitmap.height;
  const maxSide = opts?.maxSide ?? 768;
  const dispScale = Math.min(maxSide / Math.max(iw, ih), 1);
  const dw = Math.max(1, Math.round(iw * dispScale));
  const dh = Math.max(1, Math.round(ih * dispScale));
  const t0 = performance.now();

  // Network input: stretch the face to 512×512, pack [1,3,512,512] float32 normalised to [-1,1].
  const net = new OffscreenCanvas(NET, NET);
  const nctx = net.getContext("2d", { willReadFrequently: true });
  nctx.imageSmoothingQuality = "high";
  nctx.drawImage(bitmap, 0, 0, iw, ih, 0, 0, NET, NET);
  bitmap.close?.();
  const nData = nctx.getImageData(0, 0, NET, NET).data;
  const N = NET * NET;
  const feed = new Float32Array(3 * N);
  const inPlane = new Float32Array(3 * N); // [0,1] copy for sharpness + delta
  for (let i = 0; i < N; i++) {
    for (let ch = 0; ch < 3; ch++) {
      const v = nData[i * 4 + ch] / 255;
      feed[ch * N + i] = (v - 0.5) / 0.5; // → [-1,1]
      inPlane[ch * N + i] = v;
    }
  }
  const inSharp = sharpness(inPlane, NET, NET);

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, NET, NET]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // Float32Array(3*512*512), [-1,1]
  const infMs = Math.round(performance.now() - t0);

  // Restored RGBA at 512×512 + a "restoration delta" map (|out-in| amplified) showing WHERE the model
  // rebuilt detail — eyes, mouth, skin texture, hair.
  const outPlane = new Float32Array(3 * N);
  const rgba = new Uint8ClampedArray(N * 4);
  const delta = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    let d = 0;
    for (let ch = 0; ch < 3; ch++) {
      const o = Math.max(0, Math.min(1, (out[ch * N + i] + 1) / 2)); // [-1,1] → [0,1]
      outPlane[ch * N + i] = o;
      rgba[i * 4 + ch] = o * 255;
      d += Math.abs(o - inPlane[ch * N + i]);
    }
    rgba[i * 4 + 3] = 255;
    const v = Math.min(255, (d / 3) * 255 * 3); // amplify ×3 for visibility
    delta[i * 4] = 20 + v * 0.2;
    delta[i * 4 + 1] = 20 + v * 0.5;
    delta[i * 4 + 2] = 40 + v * 0.85;
    delta[i * 4 + 3] = 255;
  }
  const outSharp = sharpness(outPlane, NET, NET);

  // Resize the 512×512 restored face to the display aspect (worker-side dense composite).
  const src = new OffscreenCanvas(NET, NET);
  src.getContext("2d").putImageData(new ImageData(rgba, NET, NET), 0, 0);
  const dispC = new OffscreenCanvas(dw, dh);
  const dctx = dispC.getContext("2d");
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, NET, NET, 0, 0, dw, dh);
  const restoredBmp = dispC.transferToImageBitmap();
  const deltaBmp = await createImageBitmap(new ImageData(delta, NET, NET));

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    restoredBmp,
    deltaBmp,
    w: dw,
    h: dh,
    imgW: iw,
    imgH: ih,
    inSharp,
    outSharp,
    ms,
    infMs,
    device,
  }, [restoredBmp, deltaBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await restore(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
