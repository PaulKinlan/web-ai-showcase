// Facial-expression-recognition worker — ALL inference off the main thread so the affective-UX
// control panel never janks. Like the ViT page, we run the LOW-LEVEL forward pass (processor → model)
// rather than the convenience pipeline, so a single pass yields everything both surfaces need: the raw
// pre-softmax logits, the full softmax over the 7 emotion classes, the human-readable labels, and the
// confidence diagnostics (entropy, top-1 margin) the "see inside" panel visualises.
//
// Model: Xenova/facial_emotions_image_detection (task: image-classification), WASM backend, q8. A ViT-
// B/16 fine-tune of dima806/facial_emotions_image_detection over 7 FER emotions:
// angry, disgust, fear, happy, neutral, sad, surprise. Nothing leaves the device — the frame/photo is
// classified locally and never uploaded or stored.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/facial_emotions_image_detection";
const TASK = "image-classification";

let pipe = null;
let device = "wasm";
let RawImage = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const mod = await import(TRANSFORMERS_URL);
  RawImage = mod.RawImage;
  const loaded = await loadPipeline({
    task: TASK,
    model: MODEL,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Shannon entropy normalised to [0,1] against log(numClasses). High = the model is unsure which
// emotion; low = one emotion dominates.
function normEntropy(probs) {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h / Math.log(probs.length);
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const inputs = await pipe.processor(image);
  const output = await pipe.model(inputs);

  const logits = Array.from(output.logits.data); // [7] raw, pre-softmax
  const probs = softmax(logits);
  const id2label = pipe.model.config.id2label || {};

  const all = probs.map((p, i) => ({
    label: id2label[i] ?? `class ${i}`,
    prob: p,
    logit: logits[i],
    index: i,
  })).sort((a, b) => b.prob - a.prob);

  const sorted = [...probs].sort((a, b) => b - a);
  const margin = sorted[0] - (sorted[1] ?? 0);
  const entropy = normEntropy(probs);

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, all, entropy, margin, numClasses: probs.length, ms, device });
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
