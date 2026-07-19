// Age-estimation worker — ALL inference off the main thread so the control panel never janks. Like the
// ViT / facial-expression pages we run the LOW-LEVEL forward pass (processor → model) rather than the
// convenience pipeline, so a single pass yields everything both surfaces need: the raw pre-softmax
// logits, the full softmax over the 9 age buckets, the human-readable bucket labels, a single
// expected-age point estimate (probability-weighted bucket midpoints), and the confidence diagnostics
// (entropy, top-1 margin) the "see inside" panel visualises.
//
// Model: jdp8/vit-age-classifier (task: image-classification), WASM backend, q8. An ONNX export of
// nateraw/vit-age-classifier — a ViT-B/16 fine-tune over 9 age buckets:
//   0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, more than 70.
// This is an IMPERFECT estimate of *apparent* age from pixels — it carries demographic bias and is not
// identity or a birth date. Nothing leaves the device: the frame/photo is classified locally and never
// uploaded or stored.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "jdp8/vit-age-classifier";
const TASK = "image-classification";

// Representative midpoint (years) for each bucket, used to fold the distribution into ONE expected-age
// number. "more than 70" is open-ended; 75 is a conservative representative, flagged in the UI.
const BUCKET_MIDPOINT = {
  "0-2": 1,
  "3-9": 6,
  "10-19": 15,
  "20-29": 25,
  "30-39": 35,
  "40-49": 45,
  "50-59": 55,
  "60-69": 65,
  "more than 70": 75,
};

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

// Shannon entropy normalised to [0,1] against log(numClasses). High = the model is unsure which bucket;
// low = one bucket dominates.
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

  const logits = Array.from(output.logits.data); // [9] raw, pre-softmax
  const probs = softmax(logits);
  const id2label = pipe.model.config.id2label || {};

  const all = probs.map((p, i) => {
    const label = id2label[i] ?? `class ${i}`;
    return { label, prob: p, logit: logits[i], index: i, midpoint: BUCKET_MIDPOINT[label] ?? null };
  });
  // Keep age order (young → old) for the bars; a separate sorted copy drives the verdict + diagnostics.
  const sorted = [...all].sort((a, b) => b.prob - a.prob);

  // Expected age = Σ prob·midpoint — a single point estimate that respects the whole distribution,
  // not just the argmax bucket. Honest: it's a soft signal, not a measured age.
  const expectedAge = all.reduce((s, t) => s + t.prob * (t.midpoint ?? 0), 0);

  const p = [...probs].sort((a, b) => b - a);
  const margin = p[0] - (p[1] ?? 0);
  const entropy = normEntropy(probs);

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    all, // age-ordered [{label, prob, logit, index, midpoint}]
    top: sorted[0],
    expectedAge,
    entropy,
    margin,
    numClasses: probs.length,
    ms,
    device,
  });
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
