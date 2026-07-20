// LaMa image-inpainting worker — ALL inference AND the dense output composite off the main thread via
// raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js registers no `lama`/`fourier-conv` model class
// and there is no `inpainting` pipeline task. The ONNX export is a clean (image, mask)→image graph, so
// we run it directly with onnxruntime-web and hand-write the pre/post a pipeline would own: (1) resize
// the image AND the user's mask to the network's fixed 512×512, pack the image as [1,3,512,512] float32
// in [0,1] and the mask as [1,1,512,512] float32 (1 = hole to fill, 0 = keep), (2) the network returns
// [1,3,512,512] in the [0,255] range — we clamp it, and COMPOSITE only the masked region back over the
// original pixels (final = orig·(1−mask) + lama·mask) so untouched areas are pixel-exact, then resize to
// the display aspect. This is the isolated per-worker ORT-web escape hatch (like
// models/nafnet-image-deblurring/worker.js) — onnxruntime-web is pinned HERE only, never in shared libs.
//
// Model: Carve/LaMa-ONNX (lama_fp32.onnx, apache-2.0, ~208 MB). LaMa (Suvorov et al., WACV 2022) —
// "Resolution-robust Large Mask Inpainting with Fourier Convolutions". Inputs: "image" [1,3,512,512]
// float32 [0,1], "mask" [1,1,512,512] float32 {0,1}. Output "output" [1,3,512,512] float32 [0,255].
// Everything stays on-device: the image, the mask, and every filled pixel never leave the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "Carve/LaMa-ONNX";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/lama_fp32.onnx`;
const CACHE_NAME = "lama-inpaint-onnx-cache";
const NET = 512; // fixed network input size

let ort = null;
let session = null;
let device = "wasm";
let inputName = "image";
let maskName = "mask";
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/Carve/LaMa-ONNX/") sees them → auto-init on a returning visit, honest Download on first visit, and
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
  // Bind by declared names, tolerant of export naming.
  const ins = session.inputNames;
  inputName = ins.find((n) => /image|img|input/i.test(n)) || ins[0];
  maskName = ins.find((n) => /mask/i.test(n)) || ins[1];
  outputName = session.outputNames[0];
  post({ type: "ready", device });
}

// message: { id, imageBmp, maskBmp } — maskBmp is an alpha/white-on-black mask at any size; white (or
// opaque) marks the region to fill. opts: { maxSide }.
async function inpaint(id, imageBmp, maskBmp, opts) {
  await ensureLoaded();
  const iw = imageBmp.width, ih = imageBmp.height;
  const maxSide = opts?.maxSide ?? 900;
  const dispScale = Math.min(maxSide / Math.max(iw, ih), 1);
  const dw = Math.max(1, Math.round(iw * dispScale));
  const dh = Math.max(1, Math.round(ih * dispScale));
  const t0 = performance.now();
  const N = NET * NET;

  // Image → fixed 512×512, packed [1,3,512,512] in [0,1].
  const imgC = new OffscreenCanvas(NET, NET);
  const ictx = imgC.getContext("2d", { willReadFrequently: true });
  ictx.drawImage(imageBmp, 0, 0, iw, ih, 0, 0, NET, NET);
  const iData = ictx.getImageData(0, 0, NET, NET).data;
  const imgFeed = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    imgFeed[i] = iData[i * 4] / 255;
    imgFeed[N + i] = iData[i * 4 + 1] / 255;
    imgFeed[2 * N + i] = iData[i * 4 + 2] / 255;
  }

  // Mask → fixed 512×512, [1,1,512,512] float32 {0,1}. White/opaque = hole. We read luminance+alpha so
  // both a white-on-black mask and a transparent-brush mask work.
  const mC = new OffscreenCanvas(NET, NET);
  const mctx = mC.getContext("2d", { willReadFrequently: true });
  mctx.drawImage(maskBmp, 0, 0, maskBmp.width, maskBmp.height, 0, 0, NET, NET);
  const mData = mctx.getImageData(0, 0, NET, NET).data;
  const maskFeed = new Float32Array(N);
  let holePx = 0;
  for (let i = 0; i < N; i++) {
    const lum = (mData[i * 4] + mData[i * 4 + 1] + mData[i * 4 + 2]) / 3;
    const a = mData[i * 4 + 3];
    const on = (a > 20 && lum > 128) ? 1 : 0;
    maskFeed[i] = on;
    holePx += on;
  }
  imageBmp.close?.();
  maskBmp.close?.();

  if (holePx === 0) throw new Error("The mask is empty — paint (or drop) a region to fill first.");

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", imgFeed, [1, 3, NET, NET]);
  feeds[maskName] = new ort.Tensor("float32", maskFeed, [1, 1, NET, NET]);
  const results = await session.run(feeds);
  const out = results[outputName].data; // Float32Array(3*512*512) in [0,255]
  const infMs = Math.round(performance.now() - t0);

  // Composite: keep original pixels outside the mask exactly; use LaMa's output inside it. Also build a
  // "fill map" showing exactly the region the model synthesised.
  const rgba = new Uint8ClampedArray(N * 4);
  const fillMap = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    const m = maskFeed[i];
    const lr = Math.max(0, Math.min(255, out[i]));
    const lg = Math.max(0, Math.min(255, out[N + i]));
    const lb = Math.max(0, Math.min(255, out[2 * N + i]));
    const or = iData[i * 4], og = iData[i * 4 + 1], ob = iData[i * 4 + 2];
    rgba[i * 4] = m ? lr : or;
    rgba[i * 4 + 1] = m ? lg : og;
    rgba[i * 4 + 2] = m ? lb : ob;
    rgba[i * 4 + 3] = 255;
    // fill map: filled region tinted indigo over a dimmed original
    fillMap[i * 4] = m ? 90 : or * 0.35;
    fillMap[i * 4 + 1] = m ? 70 : og * 0.35;
    fillMap[i * 4 + 2] = m ? 230 : ob * 0.35;
    fillMap[i * 4 + 3] = 255;
  }

  // Resize the 512×512 result to the display aspect (worker-side dense composite).
  const src = new OffscreenCanvas(NET, NET);
  src.getContext("2d").putImageData(new ImageData(rgba, NET, NET), 0, 0);
  const dispC = new OffscreenCanvas(dw, dh);
  const dctx = dispC.getContext("2d");
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, NET, NET, 0, 0, dw, dh);
  const resultBmp = dispC.transferToImageBitmap();
  const fillBmp = await createImageBitmap(new ImageData(fillMap, NET, NET));

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    resultBmp,
    fillBmp,
    w: dw,
    h: dh,
    imgW: iw,
    imgH: ih,
    holeFraction: holePx / N,
    ms,
    infMs,
    device,
  }, [resultBmp, fillBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await inpaint(e.data.id, e.data.imageBmp, e.data.maskBmp, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
