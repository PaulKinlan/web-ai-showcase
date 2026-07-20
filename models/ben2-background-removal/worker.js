// BEN2 background-removal / matting worker — runs ALL segmentation off the main thread, and does the
// dense per-pixel cutout composite HERE (OffscreenCanvas + transferToImageBitmap) so the UI thread
// never touches a 1024×1024 pixel loop (invariant 15).
//
// Model: onnx-community/BEN2-ONNX (base_model PramaLLC/BEN2), task: image-segmentation. BEN2 is a
// confidence-guided matting network — it predicts a soft ALPHA MATTE (foreground confidence per pixel)
// aimed at crisp edges and fine detail (hair, fur). transformers.js 3.7.5 registers "ben" as a custom
// image-segmentation architecture. The ONNX export is fp16-only, so we load fp16 on both WebGPU (fast)
// and the WASM fallback (works, slower) — no q8 build exists upstream.
//
// I/O (verified on-device): input tensor "pixel_values" [1,3,1024,1024]; output tensor "alphas"
// [1,1,1024,1024] with values in 0..1. We scale to 0..255, resize to the original photo, and read the
// alpha channel.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/BEN2-ONNX";
let model = null;
let processor = null;
let RawImage = null;
let device = "wasm";
const dtype = "fp16"; // the only exported precision for BEN2

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Build the transparent-subject cutout as an RGBA ImageBitmap inside the worker: stamp the alpha matte
// into the decoded photo's alpha channel via an OffscreenCanvas, then transferToImageBitmap(). Returns
// null (main-thread per-pixel fallback) if OffscreenCanvas/ImageData aren't available in this worker.
function buildCutout(image, alpha, w, h) {
  try {
    if (typeof OffscreenCanvas === "undefined" || typeof ImageData === "undefined") return null;
    const src = image.data;
    const ch = image.channels ?? ((src.length / (w * h)) | 0);
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      const o = i * 4, s = i * ch;
      rgba[o] = src[s];
      rgba[o + 1] = ch >= 2 ? src[s + 1] : src[s];
      rgba[o + 2] = ch >= 3 ? src[s + 2] : src[s];
      rgba[o + 3] = alpha[i];
    }
    const oc = new OffscreenCanvas(w, h);
    oc.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
    return oc.transferToImageBitmap();
  } catch {
    return null;
  }
}

async function realDevice() {
  if ("gpu" in navigator) {
    try {
      if (await navigator.gpu.requestAdapter()) return "webgpu";
    } catch { /* fall through */ }
  }
  return "wasm";
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoModel, AutoProcessor } = mod;
  RawImage = mod.RawImage;
  device = await realDevice();
  model = await AutoModel.from_pretrained(MODEL_ID, {
    device,
    dtype,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device, dtype });
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const processed = await processor(image);
  const pixel_values = processed.pixel_values ?? processed.input ??
    Object.values(processed).find((v) => v && v.dims);
  const modelOut = await model({ pixel_values });
  // BEN2's ONNX graph names its output "alphas"; fall back to the first tensor if a build differs.
  let out = modelOut.alphas ?? modelOut.output ?? modelOut.logits;
  if (!out) {
    const first = Object.values(modelOut).find((v) => v && v.dims);
    if (!first) {
      throw new Error("model returned no tensor; keys=" + Object.keys(modelOut).join(","));
    }
    out = first;
  }
  // Single-channel foreground confidence (0..1): [B,1,H,W] or [1,H,W]. Reduce to [1,H,W].
  let t = out;
  while (t.dims.length > 3) t = t[0];
  let maskImg = RawImage.fromTensor(t.mul(255).to("uint8"));
  if (maskImg?.then) maskImg = await maskImg;
  maskImg = await maskImg.resize(image.width, image.height); // resize IS async — must await
  const w = maskImg.width, h = maskImg.height;
  const ch = maskImg.channels ?? (maskImg.data.length / (w * h)) | 0;
  let alpha;
  if (ch === 1) {
    alpha = new Uint8Array(maskImg.data); // own copy so its buffer is transferable
  } else {
    alpha = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) alpha[i] = maskImg.data[i * ch];
  }

  // "See inside" numbers: foreground coverage, soft-edge pixel count, and a 16-bin alpha histogram.
  let fg = 0, soft = 0;
  const hist = new Array(16).fill(0);
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i];
    if (a >= 128) fg++;
    if (a > 25 && a < 230) soft++;
    hist[Math.min(15, a >> 4)]++;
  }

  const ms = Math.round(performance.now() - t0);

  // Composite the subject cutout (photo pixels + alpha matte) HERE, off the main thread, and hand back
  // a ready-to-draw RGBA ImageBitmap. The main thread then only paints a backdrop + drawImage(cutout).
  const cutout = buildCutout(image, alpha, w, h);

  const transfer = [alpha.buffer];
  if (cutout) transfer.push(cutout);
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      alpha: alpha.buffer,
      cutout,
      coverage: fg / alpha.length,
      softEdge: soft,
      hist,
      ms,
      device,
      dtype,
    },
    transfer,
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
