// CLIPSeg worker — text-prompted image segmentation OFF the main thread, so the control UI stays
// responsive while the model runs. Model: Xenova/clipseg-rd64-refined (CLIPSegForImageSegmentation),
// WASM backend, q8 (model_quantized.onnx). ~140 MB, cached by transformers.js after first load.
//
// How CLIPSeg actually works (the real API — not invented; verified against transformers.js 3.7.5):
//   • AutoProcessor for this repo exposes an IMAGE processor only (no tokenizer bundled), so we load
//     the text tokenizer separately with AutoTokenizer.
//   • Tokenise the N phrases (padded) → input_ids / attention_mask  ([N, L]).
//   • Run the image processor once → pixel_values [1, 3, 352, 352]; TILE it to [N, 3, 352, 352] so
//     every phrase is scored against the same image in one batched forward pass.
//   • model({ input_ids, attention_mask, pixel_values }) → logits [N, 352, 352] — a RAW per-pixel
//     score map per phrase at the model's 352×352 working resolution.
// We return the raw logits (Float32) per prompt; sigmoid + thresholding happen on the page so the
// "see inside" heatmap and the threshold slider stay live without re-running the model.

import { pickDevice, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/clipseg-rd64-refined";
let tokenizer = null;
let imageProcessor = null;
let model = null;
let RawImageRef = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoProcessor, CLIPSegForImageSegmentation, RawImage, env } = await import(
    TRANSFORMERS_URL
  );
  env.allowLocalModels = false;
  RawImageRef = RawImage;
  // CLIPSeg runs happily on WASM; q8 quantised. Prefer the honest device (wasm here) — no WebGPU need.
  device = await pickDevice("wasm");
  const onProgress = (p) => post({ type: "progress", p });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, { progress_callback: onProgress });
  const processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: onProgress,
  });
  imageProcessor = processor.image_processor || processor;
  model = await CLIPSegForImageSegmentation.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device,
    progress_callback: onProgress,
  });
  post({ type: "ready", device });
}

// Tile pixel_values [1,C,H,W] → [n,C,H,W] (batch the same image against every phrase).
function tile(pv, n) {
  if (pv.dims[0] === n) return pv;
  const [, C, H, W] = pv.dims;
  const per = C * H * W;
  const data = new pv.data.constructor(per * n);
  for (let i = 0; i < n; i++) data.set(pv.data, i * per);
  return new pv.constructor(pv.type, data, [n, C, H, W]);
}

async function run(id, imageURL, prompts) {
  await ensureLoaded();
  if (!Array.isArray(prompts) || prompts.length === 0) throw new Error("no prompts given");
  const t0 = performance.now();

  const image = (await RawImageRef.read(imageURL)).rgb();
  const textInputs = tokenizer(prompts, { padding: true, truncation: true });
  const imgInputs = await imageProcessor(image);
  const pixel_values = tile(imgInputs.pixel_values, prompts.length);

  const out = await model({
    input_ids: textInputs.input_ids,
    attention_mask: textInputs.attention_mask,
    pixel_values,
  });

  const logits = out.logits;
  const dims = logits.dims;
  const mapH = dims[dims.length - 2];
  const mapW = dims[dims.length - 1];
  const per = mapW * mapH;
  const src = logits.data; // Float32Array length prompts*per (or per when a single prompt is squeezed)
  const n = Math.max(1, Math.round(src.length / per));

  const maps = [];
  const transfer = [];
  for (let p = 0; p < prompts.length; p++) {
    const slice = new Float32Array(per);
    const base = (p % n) * per;
    slice.set(src.subarray(base, base + per));
    maps.push({ prompt: prompts[p], data: slice.buffer });
    transfer.push(slice.buffer);
  }

  post({
    type: "result",
    id,
    mapW,
    mapH,
    imageW: image.width,
    imageH: image.height,
    maps,
    ms: Math.round(performance.now() - t0),
    device,
  }, transfer);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.prompts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
