// ViTPose multi-model worker — a real two-model composition, off the main thread:
//   1. YOLOS-tiny (object-detection) finds every person and returns pixel boxes.
//   2. ViTPose estimates a pose for EACH person box (top-down, one crop per person).
// Detector: Xenova/yolos-tiny (~30 MB, q8). Pose: onnx-community/vitpose-base-simple (~87 MB, q8).

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const POSE_ID = "onnx-community/vitpose-base-simple";
const DET_ID = "Xenova/yolos-tiny";
let detector = null;
let model = null;
let processor = null;
let RawImage = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (model && detector) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoModel, AutoImageProcessor, env } = mod;
  RawImage = mod.RawImage;
  env.allowLocalModels = false;
  // Detector first (smaller), then the pose model — one progress stream for the shared loader.
  const det = await loadPipeline({
    task: "object-detection",
    model: DET_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  detector = det.pipe;
  model = await AutoModel.from_pretrained(POSE_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoImageProcessor.from_pretrained(POSE_ID);
  device = "wasm";
  post({ type: "ready", device });
}

// Cap the number of people we pose-estimate per image so a crowded photo stays responsive.
const MAX_PEOPLE = 8;

async function run(id, imageURL, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);

  // 1) Detect people.
  const detections = await detector(image, { threshold: threshold ?? 0.5 });
  const people = detections
    .filter((d) => d.label === "person")
    .sort((a, b) => b.score - a.score)
    .slice(0, MAX_PEOPLE);
  const detMs = Math.round(performance.now() - t0);

  if (people.length === 0) {
    post({ type: "result", id, persons: [], detMs, poseMs: 0, ms: detMs, device });
    return;
  }

  // 2) ViTPose is top-down: its image processor resizes the WHOLE image (it does not crop by box), so
  //    to estimate each person independently we crop their box ourselves, run the model on the crop,
  //    then offset the keypoints back into image-space.
  const tp = performance.now();
  const persons = [];
  for (const d of people) {
    const b = d.box;
    const x = Math.max(0, Math.round(b.xmin));
    const y = Math.max(0, Math.round(b.ymin));
    const x2 = Math.min(image.width, Math.round(b.xmax));
    const y2 = Math.min(image.height, Math.round(b.ymax));
    const cw = x2 - x, ch = y2 - y;
    if (cw < 8 || ch < 8) continue;
    const crop = await image.crop([x, y, x2, y2]); // [left, top, right, bottom]
    const box = [[[0, 0, cw, ch]]];
    const inputs = await processor(crop, { boxes: box });
    const output = await model(inputs);
    const [p] = processor.post_process_pose_estimation(output.heatmaps, box)[0];
    persons.push({
      keypoints: p.keypoints.map((kp) => [Number(kp[0]) + x, Number(kp[1]) + y]),
      scores: Array.from(p.scores).map((s) => Number(s)),
      detBox: [x, y, cw, ch],
      detScore: Number(d.score),
    });
  }
  const poseMs = Math.round(performance.now() - tp);

  post({
    type: "result",
    id,
    persons,
    imageSize: [image.width, image.height],
    detMs,
    poseMs,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image, e.data.threshold);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
