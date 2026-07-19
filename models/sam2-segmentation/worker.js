// SAM 2.1 point/box-prompt worker — ALL inference off the main thread.
//
// SAM 2 (Segment Anything 2, Meta) is the newer promptable segmenter: a Hiera hierarchical vision
// backbone that emits a multi-scale feature pyramid, feeding a prompt encoder + mask decoder. Like
// SAM 1 it has two stages — an expensive image encoder (run ONCE per image, cached) and a cheap
// prompt/mask decoder (run per click/box). We cache the image embeddings so every extra prompt is
// fast. The same architecture powers SAM 2 video (memory attention across frames); here we run the
// image path.
//
// Model:   onnx-community/sam2.1-hiera-tiny-ONNX  (task: mask-generation / SAM 2)
// Classes: Sam2Model + Sam2Processor  (present in @huggingface/transformers >= 4.x, NOT in 3.7.5)
// dtype:   vision_encoder fp16 (~67 MB) + prompt_encoder_mask_decoder fp32 (crisp masks)
// Backend: WebAssembly — runs anywhere, no WebGPU required.
//
// Verified (headless Chrome, WASM): get_image_embeddings -> {image_embeddings.0/1/2}, forward returns
// {iou_scores, pred_masks, object_score_logits}; post_process_masks -> [1,3,H,W]; 3 candidate masks
// with real predicted-IoU scores. Point AND box prompts both return masks.

// SAM 2 lives in Transformers.js v4+ (Sam2Model). The shared webai.js is pinned to 3.7.5, which has
// no Sam2 class, so this worker pins v4 locally — a per-model choice, not a shared-file change.
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

let model = null;
let processor = null;
let Tensor = null;
let RawImage = null;
const device = "wasm";

// Per-image cache: the vision-encoder pyramid + the sizes post_process_masks needs. Clicks reuse it.
let cache = null; // { embeddings, imageInputs }

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { Sam2Model, Sam2Processor, AutoProcessor, RawImage: RI, Tensor: T, env } = mod;
  if (!Sam2Model) {
    throw new Error(
      "This build of Transformers.js has no Sam2Model class (SAM 2 needs v4+). Update the pinned version.",
    );
  }
  env.allowLocalModels = false;
  Tensor = T;
  RawImage = RI;
  // fp16 vision encoder keeps the download small (~67 MB); the mask decoder stays fp32 for crisp masks.
  model = await Sam2Model.from_pretrained("onnx-community/sam2.1-hiera-tiny-ONNX", {
    dtype: { vision_encoder: "fp16", prompt_encoder_mask_decoder: "fp32" },
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await (Sam2Processor || AutoProcessor).from_pretrained(
    "onnx-community/sam2.1-hiera-tiny-ONNX",
  );
  post({ type: "ready", device });
}

// Compute + cache the vision-encoder embeddings (Hiera multi-scale pyramid) for one image.
async function embed(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const imageInputs = await processor(image);
  const embeddings = await model.get_image_embeddings(imageInputs);
  cache = { embeddings, imageInputs };
  const originalSize = imageInputs.original_sizes[0]; // [height, width]
  // How many pyramid levels the encoder produced (a SAM 2 vs SAM 1 distinction, for "see inside").
  const embKeys = Object.keys(embeddings).filter((k) => k.startsWith("image_embeddings"));
  post({
    type: "embedded",
    id,
    originalSize,
    embeddingLevels: embKeys.length,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

// Decode candidate masks from point prompts and/or a box prompt.
//   points: [{x,y,label}]  normalized [0,1], label 1 = foreground / 0 = background
//   box:    [x1,y1,x2,y2]  normalized [0,1], or null
async function segment(id, points, box) {
  if (!cache) throw new Error("Call embed(image) before segment().");
  const t0 = performance.now();
  const reshaped = cache.imageInputs.reshaped_input_sizes[0]; // [rH, rW] — SAM 2 pads to 1024×1024
  const [rH, rW] = reshaped;

  const args = { ...cache.embeddings };
  if (points && points.length) {
    const flatPoints = [];
    const flatLabels = [];
    for (const p of points) {
      flatPoints.push(p.x * rW, p.y * rH);
      flatLabels.push(BigInt(p.label ?? 1));
    }
    args.input_points = new Tensor("float32", flatPoints, [1, 1, points.length, 2]);
    args.input_labels = new Tensor("int64", flatLabels, [1, 1, points.length]);
  }
  if (box && box.length === 4) {
    args.input_boxes = new Tensor(
      "float32",
      [box[0] * rW, box[1] * rH, box[2] * rW, box[3] * rH],
      [1, 1, 4],
    );
  }

  const outputs = await model(args);

  const masks = await processor.post_process_masks(
    outputs.pred_masks,
    cache.imageInputs.original_sizes,
    cache.imageInputs.reshaped_input_sizes,
  );
  const maskTensor = masks[0]; // dims [1, numMasks, H, W], bool
  const [, numMasks, H, W] = maskTensor.dims;
  const scores = Array.from(outputs.iou_scores.data); // predicted-IoU per candidate
  const objScore = outputs.object_score_logits
    ? Array.from(outputs.object_score_logits.data)[0]
    : null;

  // Extract every candidate plane + its area so the UI can switch candidates without re-inference.
  const src = maskTensor.data;
  const planes = [];
  const areas = [];
  for (let m = 0; m < numMasks; m++) {
    const plane = new Uint8Array(H * W);
    const off = m * H * W;
    let area = 0;
    for (let i = 0; i < H * W; i++) {
      plane[i] = src[off + i] ? 1 : 0;
      if (plane[i]) area++;
    }
    planes.push(plane);
    areas.push(area);
  }
  let best = 0;
  for (let i = 1; i < numMasks; i++) if (scores[i] > scores[best]) best = i;

  post(
    {
      type: "result",
      id,
      width: W,
      height: H,
      masks: planes,
      areas,
      scores,
      objScore,
      bestIndex: best,
      numMasks,
      points: points || [],
      box: box || null,
      ms: Math.round(performance.now() - t0),
      device,
    },
    planes.map((p) => p.buffer),
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "embed") await embed(e.data.id, e.data.image);
    else if (type === "segment") await segment(e.data.id, e.data.points, e.data.box);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
