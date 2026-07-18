// RMBG-1.4 background-removal worker — runs ALL segmentation off the main thread. One forward pass
// returns the foreground alpha matte at the original resolution, plus the numbers "See inside" needs
// (coverage, soft-edge pixels, alpha histogram).
//
// Model: briaai/RMBG-1.4 (task: image-segmentation → a single foreground matte). WebGPU fp32 with an
// honest WASM (q8) fallback. Uses the documented AutoModel + AutoProcessor path from the model card —
// the real API, not an invented one. The explicit processor config mirrors BRIA's reference usage.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let model = null;
let processor = null;
let RawImage = null;
let device = "wasm";
let dtype = "q8";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
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
  dtype = device === "webgpu" ? "fp32" : "q8"; // fp32 on GPU; q8 build keeps the CPU fallback small
  model = await AutoModel.from_pretrained("briaai/RMBG-1.4", {
    device,
    dtype,
    config: { model_type: "custom" },
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained("briaai/RMBG-1.4", {
    config: {
      do_normalize: true,
      do_pad: false,
      do_rescale: true,
      do_resize: true,
      image_mean: [0.5, 0.5, 0.5],
      image_std: [1, 1, 1],
      resample: 2,
      rescale_factor: 0.00392156862745098,
      size: { width: 1024, height: 1024 },
    },
  });
  post({ type: "ready", device, dtype });
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const processed = await processor(image);
  const pixel_values = processed.pixel_values ?? processed.input ?? Object.values(processed)[0];
  const modelOut = await model({ input: pixel_values });
  // RMBG's ONNX graph names its output "output"; fall back to the first tensor if a build differs.
  let out = modelOut.output ?? modelOut.logits ?? modelOut.alphas;
  if (!out) {
    const first = Object.values(modelOut).find((v) => v && v.dims);
    if (!first) {
      throw new Error("model returned no tensor; keys=" + Object.keys(modelOut).join(","));
    }
    out = first;
  }
  // A single-channel foreground probability map (0–1). It can come back as [B,1,H,W] or [1,H,W];
  // index down to a [1,H,W] tensor before turning it into an image.
  let t = out;
  while (t.dims.length > 3) t = t[0];
  let maskImg = RawImage.fromTensor(t.mul(255).to("uint8"));
  if (maskImg?.then) maskImg = await maskImg; // fromTensor is sync in most builds, await defensively
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

  // "See inside" numbers: how much is kept, how soft the edge is, and the alpha distribution.
  let fg = 0, soft = 0;
  const hist = new Array(16).fill(0);
  for (let i = 0; i < alpha.length; i++) {
    const a = alpha[i];
    if (a >= 128) fg++;
    if (a > 25 && a < 230) soft++;
    hist[Math.min(15, a >> 4)]++;
  }

  const ms = Math.round(performance.now() - t0);
  const buf = alpha.buffer.slice(0);
  post(
    {
      type: "result",
      id,
      width: w,
      height: h,
      alpha: buf,
      coverage: fg / alpha.length,
      softEdge: soft,
      hist,
      ms,
      device,
      dtype,
    },
    [buf],
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
