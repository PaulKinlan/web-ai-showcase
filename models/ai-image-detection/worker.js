// AI-generated-image detection worker — all inference off the main thread so the control UI stays
// responsive. Model: onnx-community/SMOGY-Ai-images-detector-ONNX (SwinForImageClassification, task:
// image-classification), WASM, q8. Two classes: `artificial` (AI-generated / synthetic) and `human`
// (a real photograph). This is SINGLE-label softmax — the two scores sum to 1.
//
// We run the LOW-LEVEL forward (processor → model → logits) instead of the convenience pipeline, so a
// single pass yields the raw logits AND the softmax probabilities the "see inside" surface needs.
//
// HONESTY: this is an imperfect provenance SIGNAL, not proof of origin. It was trained on a particular
// distribution of real photos and AI images; it will produce false positives and false negatives, and
// simple edits (crop, recompress, add noise) can flip it. The demo frames the number as triage
// evidence, never a verdict.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/SMOGY-Ai-images-detector-ONNX";
const AI_LABEL = "artificial";
const REAL_LABEL = "human";
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
    task: "image-classification",
    model: MODEL_ID,
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

// Forward one image → { classes:[{label,prob,logit}], ai, real }.
async function scoreImage(imageURL) {
  const image = await RawImage.read(imageURL);
  const inputs = await pipe.processor(image);
  const output = await pipe.model(inputs);
  const logits = Array.from(output.logits.data); // [2] raw, pre-softmax
  const probs = softmax(logits);
  const id2label = pipe.model.config.id2label || { 0: AI_LABEL, 1: REAL_LABEL };
  const classes = probs.map((p, i) => ({
    label: id2label[i] ?? `class ${i}`,
    prob: p,
    logit: logits[i],
  }));
  const byLabel = Object.fromEntries(classes.map((c) => [c.label, c.prob]));
  return { classes, ai: byLabel[AI_LABEL] ?? 0, real: byLabel[REAL_LABEL] ?? 0 };
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreImage(imageURL);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, ...r, ms, device });
}

// Batch screen many images (each is a data URL). Returns per-item flag against a threshold.
async function screen(id, images, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const items = [];
  for (const { key, url } of images) {
    const r = await scoreImage(url);
    items.push({ key, ai: r.ai, real: r.real, flagged: r.ai >= threshold });
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "screen", id, items, ms, device });
}

// Provenance GATE for multi-model composition: should this image be flagged as likely-AI upstream?
async function gate(id, imageURL, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreImage(imageURL);
  const ms = Math.round(performance.now() - t0);
  post({ type: "gate", id, flagged: r.ai >= threshold, ai: r.ai, real: r.real, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.image);
    else if (type === "screen") await screen(e.data.id, e.data.images, e.data.threshold);
    else if (type === "gate") await gate(e.data.id, e.data.image, e.data.threshold);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
