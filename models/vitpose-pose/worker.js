// ViTPose worker — runs the transformer pose estimator OFF the main thread so the control UI stays
// responsive (invariant 3). One forward pass per image returns everything the pages need: the decoded
// COCO keypoints + confidences for each person box, AND the raw heatmaps so "See inside" can show the
// literal heatmap → arg-max decode.
//
// Model: onnx-community/vitpose-base-simple (VitPoseForPoseEstimation), WASM backend, q8.
// ViTPose is TOP-DOWN: it needs a person box. With no detector we pass a single full-image box; the
// multi-model page passes real person boxes from an object detector.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/vitpose-base-simple";
let model = null;
let processor = null;
let RawImage = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoModel, AutoImageProcessor, env } = mod;
  RawImage = mod.RawImage;
  // Let the library own its Cache Storage; do not fight the service worker (invariant 4).
  env.allowLocalModels = false;
  model = await AutoModel.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoImageProcessor.from_pretrained(MODEL_ID);
  device = "wasm";
  post({ type: "ready", device });
}

async function run(id, imageURL, boxes) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  // boxes: [x, y, w, h] person boxes. Default to one full-image box.
  const personBoxes = (boxes && boxes.length) ? boxes : [[0, 0, image.width, image.height]];
  const boxesArg = [personBoxes]; // one image → its list of person boxes

  const inputs = await processor(image, { boxes: boxesArg });
  const output = await model(inputs);
  const heatmaps = output.heatmaps; // dims [numBoxes, 17, H, W]
  const decoded = processor.post_process_pose_estimation(heatmaps, boxesArg)[0];

  const persons = decoded.map((p) => ({
    keypoints: p.keypoints.map((kp) => [Number(kp[0]), Number(kp[1])]),
    scores: Array.from(p.scores).map((s) => Number(s)),
    bbox: p.bbox ? Array.from(p.bbox).map((v) => Number(v)) : null,
  }));

  // Raw heatmaps for person 0 → the "see inside" surface (17 × H × W).
  const hd = heatmaps.dims; // [numBoxes, 17, H, W]
  const H = hd[2], W = hd[3];
  const perPerson = 17 * H * W;
  const src = heatmaps.data;
  const first = new Float32Array(perPerson);
  for (let i = 0; i < perPerson; i++) first[i] = src[i];

  const ms = Math.round(performance.now() - t0);
  post(
    {
      type: "result",
      id,
      persons,
      imageSize: [image.width, image.height],
      heatmapDims: [17, H, W],
      heatmaps: first,
      ms,
      device,
    },
    [first.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.boxes);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
