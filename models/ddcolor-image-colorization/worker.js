// DDColor image-colorization worker — ALL inference AND the dense Lab→RGB composite off the main
// thread via raw ONNX Runtime Web.
//
// Why raw ORT and not transformers.js: transformers.js 3.7.5/4.2 registers no `ddcolor` model class
// (DDColor is a ConvNeXt encoder + a multi-scale colour-query decoder — a bespoke architecture), and
// there is no `image-colorization` pipeline task. The ONNX export is a clean image→image graph, so we
// run it directly with onnxruntime-web and hand-write the two pieces a pipeline would own: (1) build a
// luminance-only input (rebuild a grayscale RGB from the L channel, values [0,1], NO ImageNet
// normalisation) at the fixed 512×512 network size, and (2) take the predicted chroma, upsample it to
// the display resolution, recombine with the ORIGINAL full-res luminance, and convert Lab→RGB. This is
// the isolated per-worker ORT-web escape hatch (like models/scene-text-detection/worker.js) —
// onnxruntime-web is pinned HERE only, never in shared lib/webai.js.
//
// Model: edgetools/ddcolor (ddcolor-tiny-fp16.onnx, Apache-2.0, ~130 MB). DDColor-tiny (ICCV 2023).
// Input "input" [1,3,512,512] float32 — a grayscale RGB rebuilt from L. Output "output" [1,2,512,512]
// float32 — the predicted **ab chroma** in OpenCV's float Lab convention (L∈[0,100], a,b∈~[-127,127]).
// DDColor predicts CHROMA ONLY; the luminance is the source image's, so structure/detail is preserved
// exactly and only the colour is invented. Everything stays on-device: the image never leaves the tab.

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs";
const ORT_WASM = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
const MODEL_ID = "edgetools/ddcolor";
const MODEL_URL = `https://huggingface.co/${MODEL_ID}/resolve/main/ddcolor-tiny-fp16.onnx`;
const CACHE_NAME = "ddcolor-onnx-cache";
const NET = 512; // fixed network input size

let ort = null;
let session = null;
let device = "wasm";
let inputName = null;
let outputName = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// ── sRGB ↔ CIE-Lab (D65, OpenCV float convention: L 0..100, a,b ~ -127..127) ──────────────────────
function srgb2lin(c) {
  return c > 0.04045 ? Math.pow((c + 0.055) / 1.055, 2.4) : c / 12.92;
}
function lin2srgb(c) {
  c = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  return Math.max(0, Math.min(1, c)) * 255;
}
const fLab = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
const fInv = (t) => {
  const t3 = t * t * t;
  return t3 > 0.008856 ? t3 : (t - 16 / 116) / 7.787;
};
function rgb2L(r, g, b) {
  const R = srgb2lin(r / 255), G = srgb2lin(g / 255), B = srgb2lin(b / 255);
  const y = R * 0.2126 + G * 0.7152 + B * 0.0722;
  return 116 * fLab(y) - 16;
}
function lab2rgb(Lc, a, bb, out, o) {
  let y = (Lc + 16) / 116, x = a / 500 + y, z = y - bb / 200;
  x = 0.95047 * fInv(x);
  y = fInv(y);
  z = 1.08883 * fInv(z);
  const r = x * 3.2406 - y * 1.5372 - z * 0.4986;
  const g = -x * 0.9689 + y * 1.8758 + z * 0.0415;
  const b = x * 0.0557 - y * 0.2040 + z * 1.0570;
  out[o] = lin2srgb(r);
  out[o + 1] = lin2srgb(g);
  out[o + 2] = lin2srgb(b);
  out[o + 3] = 255;
}

// Fetch the ONNX weights THROUGH Cache Storage so lib/model-cache.js (which scans caches for
// "/edgetools/ddcolor/") sees them → auto-init on a returning visit, honest Download on first visit,
// and the per-model "clear cache" control all work. Streams download progress.
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

// Bilinear-sample a single channel of a src grid (sw×sh) at fractional (fx,fy).
function sample(ch, sw, sh, fx, fy) {
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1);
  const dx = fx - x0, dy = fy - y0;
  const a = ch[y0 * sw + x0], b = ch[y0 * sw + x1];
  const c = ch[y1 * sw + x0], d = ch[y1 * sw + x1];
  return a * (1 - dx) * (1 - dy) + b * dx * (1 - dy) + c * (1 - dx) * dy + d * dx * dy;
}

// std of a Float32Array (for the honesty readout — how much chroma the model actually invented).
function stdOf(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  const m = s / arr.length;
  let q = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    q += d * d;
  }
  return Math.sqrt(q / arr.length);
}

async function colorize(id, bitmap, opts) {
  await ensureLoaded();
  const sat = opts?.saturation ?? 1; // chroma multiplier (wild demo)
  const maxSide = opts?.maxSide ?? 1024; // cap display resolution for the full-res composite
  const iw = bitmap.width, ih = bitmap.height;
  const scale = Math.min(maxSide / Math.max(iw, ih), 1);
  const dw = Math.max(1, Math.round(iw * scale));
  const dh = Math.max(1, Math.round(ih * scale));
  const t0 = performance.now();

  // Display-resolution luminance (preserved exactly) + a grayscale RGBA for the "before" view.
  const disp = new OffscreenCanvas(dw, dh);
  const dctx = disp.getContext("2d", { willReadFrequently: true });
  dctx.drawImage(bitmap, 0, 0, dw, dh);
  const dData = dctx.getImageData(0, 0, dw, dh).data;
  const Lfull = new Float32Array(dw * dh);
  const gray = new Uint8ClampedArray(dw * dh * 4);
  for (let i = 0; i < dw * dh; i++) {
    const L = rgb2L(dData[i * 4], dData[i * 4 + 1], dData[i * 4 + 2]);
    Lfull[i] = L;
    // grayscale RGB from L alone (a=b=0) — exactly what the model "sees".
    lab2rgb(L, 0, 0, gray, i * 4);
  }

  // Network input: 512×512 grayscale RGB rebuilt from L (values [0,1], NCHW, no normalisation).
  // Rescale the display-res grayscale down to the fixed 512×512 network size.
  const grayCanvas = new OffscreenCanvas(dw, dh);
  grayCanvas.getContext("2d").putImageData(new ImageData(gray, dw, dh), 0, 0);
  const net2 = new OffscreenCanvas(NET, NET);
  const n2 = net2.getContext("2d");
  n2.drawImage(grayCanvas, 0, 0, dw, dh, 0, 0, NET, NET);
  const nData = n2.getImageData(0, 0, NET, NET).data;
  const feed = new Float32Array(3 * NET * NET);
  const N = NET * NET;
  for (let i = 0; i < N; i++) {
    feed[i] = nData[i * 4] / 255;
    feed[N + i] = nData[i * 4 + 1] / 255;
    feed[2 * N + i] = nData[i * 4 + 2] / 255;
  }

  const feeds = {};
  feeds[inputName] = new ort.Tensor("float32", feed, [1, 3, NET, NET]);
  const results = await session.run(feeds);
  const ab = results[outputName].data; // Float32Array(2*512*512): a then b
  const infMs = Math.round(performance.now() - t0);
  const aCh = ab.subarray(0, N), bCh = ab.subarray(N, 2 * N);
  const aStd = stdOf(aCh), bStd = stdOf(bCh);

  bitmap.close?.();

  // ── Dense composite (worker-side): upsample ab to display res, recombine with full-res L. ──────
  const color = new Uint8ClampedArray(dw * dh * 4);
  const sx = NET / dw, sy = NET / dh;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const i = y * dw + x;
      const fx = Math.min(NET - 1, x * sx), fy = Math.min(NET - 1, y * sy);
      const a = sample(aCh, NET, NET, fx, fy) * sat;
      const b = sample(bCh, NET, NET, fx, fy) * sat;
      lab2rgb(Lfull[i], a, b, color, i * 4);
    }
  }

  // Chroma-only "see inside" view at network res: fix L≈70, show the invented colour field alone.
  const chroma = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) lab2rgb(72, aCh[i], bCh[i], chroma, i * 4);

  const colorBmp = await createImageBitmap(new ImageData(color, dw, dh));
  const grayOutBmp = await createImageBitmap(new ImageData(gray, dw, dh));
  const chromaBmp = await createImageBitmap(new ImageData(chroma, NET, NET));
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    colorBmp,
    grayBmp: grayOutBmp,
    chromaBmp,
    w: dw,
    h: dh,
    imgW: iw,
    imgH: ih,
    ms,
    infMs,
    aStd,
    bStd,
    device,
  }, [colorBmp, grayOutBmp, chromaBmp]);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await colorize(e.data.id, e.data.bitmap, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
