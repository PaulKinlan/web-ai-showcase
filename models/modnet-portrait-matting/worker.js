// MODNet portrait-matting worker — runs ALL matting off the main thread. One forward pass returns a
// soft foreground ALPHA MATTE at the input resolution (0 = background … 1 = subject), which the page
// stamps into the photo's alpha channel. Where RMBG-1.4 does general dichotomous foreground/background
// segmentation (IS-Net), MODNet is a trimap-free MATTING network: it predicts fine, continuous alpha
// along wispy edges — hair, fur, motion — which is the hard part of a believable cutout.
//
// Model: Xenova/modnet (task: image-segmentation → a single-channel alpha matte). The ONNX graph takes
// one input tensor named "input" and returns one output tensor named "output" (both verified from the
// live session's inputNames/outputNames). It has a DYNAMIC input size, so any aspect ratio runs — no
// fixed 1024² crop. WebGPU fp32 with an honest WASM (q8, ~7 MB) fallback. We load it through the
// documented AutoModel + AutoProcessor path (the real API), reading the matte tensor directly.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let model = null;
let processor = null;
let RawImage = null;
let device = "wasm";
let dtype = "q8";
let inputName = "input";

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
  // fp32 on the GPU for the cleanest matte; the small q8 build keeps the CPU fallback ~7 MB.
  dtype = device === "webgpu" ? "fp32" : "q8";
  model = await AutoModel.from_pretrained("Xenova/modnet", {
    device,
    dtype,
    config: { model_type: "custom" },
    progress_callback: (p) => post({ type: "progress", p }),
  });
  // Read the real ONNX IO names off the live session so we never invent a signature.
  try {
    const sess = model.sessions?.model ?? Object.values(model.sessions ?? {})[0];
    if (sess?.inputNames?.length) inputName = sess.inputNames[0];
  } catch { /* keep default "input" */ }
  processor = await AutoProcessor.from_pretrained("Xenova/modnet");
  post({ type: "ready", device, dtype });
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const processed = await processor(image);
  const pixel_values = processed.pixel_values ?? processed.input ?? Object.values(processed)[0];
  const modelOut = await model({ [inputName]: pixel_values });
  // MODNet's ONNX graph names its output "output"; fall back to the first real tensor if a build differs.
  let out = modelOut.output ?? modelOut.matte ?? modelOut.alphas;
  if (!out) {
    const first = Object.values(modelOut).find((v) => v && v.dims);
    if (!first) {
      throw new Error("model returned no tensor; keys=" + Object.keys(modelOut).join(","));
    }
    out = first;
  }
  // A single-channel alpha map (0–1) shaped [B,1,H,W] or [1,H,W]; index down to [1,H,W] before imaging.
  let t = out;
  while (t.dims.length > 3) t = t[0];
  let maskImg = RawImage.fromTensor(t.mul(255).to("uint8"));
  if (maskImg?.then) maskImg = await maskImg; // fromTensor is sync in most builds; await defensively
  maskImg = await maskImg.resize(image.width, image.height); // resize IS async — must await
  const w = maskImg.width, h = maskImg.height;
  const ch = maskImg.channels ?? (maskImg.data.length / (w * h)) | 0;
  let alpha;
  if (ch === 1) {
    alpha = maskImg.data instanceof Uint8Array ? maskImg.data : Uint8Array.from(maskImg.data);
  } else {
    alpha = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) alpha[i] = maskImg.data[i * ch];
  }

  // "See inside" numbers: how much is kept, how soft the boundary is, and the alpha distribution.
  let fg = 0, soft = 0;
  const hist = new Array(16).fill(0);
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i];
    if (a >= 128) fg++;
    if (a > 25 && a < 230) soft++; // partial alpha = the model's hair/fur/edge detail
    hist[Math.min(15, a >> 4)]++;
  }

  const ms = Math.round(performance.now() - t0);

  // Composite the subject cutout (photo pixels + alpha matte) HERE, off the main thread, and hand back
  // a ready-to-draw RGBA ImageBitmap. The main thread then only paints a backdrop + drawImage(cutout) —
  // no getImageData/per-pixel/putImageData on the UI thread (that dense loop was the INP cost @1080p).
  // Degrade to null where OffscreenCanvas is unavailable; the page keeps its per-pixel fallback.
  const cutout = buildCutout(image, alpha, w, h);

  const transfer = [buf];
  if (cutout) transfer.push(cutout);
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      alpha: buf,
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
