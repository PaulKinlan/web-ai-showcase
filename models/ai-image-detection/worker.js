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
let loadPromise = null;

// How many forwards may be in flight at once. The engine is single-threaded WASM (ort.env.wasm.numThreads
// = 1 on this repo, and this worker pins backend "wasm"), so extra concurrency buys no throughput; it
// only multiplies live decoded buffers and tensors in the WASM heap. Bound it and queue the rest FIFO.
const MAX_IN_FLIGHT_FORWARDS = 2;
let forwardsInFlight = 0;
const forwardQueue = [];

function post(msg) {
  self.postMessage(msg);
}

/**
 * Load the model at most once, even under concurrent requests.
 *
 * `run`/`screen`/`gate` all call this, the message listener is async, and the client's
 * AIDetectEngine._call posts immediately with no queue, so two messages arriving before the load
 * finishes used to each start a full `loadPipeline` — a duplicate download and a second set of model
 * weights, with a second `ready` message. models/ai-image-detection/wild/index.html issues two
 * classify() calls together, so it reached that state without any unusual user timing.
 * Share one in-flight promise; if it rejects, clear it so a later request can retry rather than
 * replaying a permanently rejected memo.
 */
function ensureLoaded() {
  if (!loadPromise) {
    loadPromise = (async () => {
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
    })().catch((error) => {
      loadPromise = null; // failed load must be retryable
      throw error;
    });
  }
  return loadPromise;
}

/**
 * Run `task` once a forward slot is free, FIFO. `task` is a closure over an image URL that the worker
 * has already received, so queueing adds no new payload to keep alive — only the number of forwards
 * that may touch the WASM heap at once is bounded.
 */
function withForwardSlot(task) {
  return new Promise((resolve, reject) => {
    forwardQueue.push({ task, resolve, reject });
    pumpForwardQueue();
  });
}

function pumpForwardQueue() {
  while (forwardsInFlight < MAX_IN_FLIGHT_FORWARDS && forwardQueue.length > 0) {
    const { task, resolve, reject } = forwardQueue.shift();
    forwardsInFlight++;
    Promise.resolve()
      .then(task)
      .then(resolve, reject)
      .finally(() => {
        // A rejection must free its slot and keep draining: one bad image may not wedge the queue.
        forwardsInFlight--;
        pumpForwardQueue();
      });
  }
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// Forward one image → { classes:[{label,prob,logit}], ai, real }. Goes through the bounded FIFO queue
// so run/screen/gate can never exceed MAX_IN_FLIGHT_FORWARDS concurrent forwards between them.
async function scoreImageUnqueued(imageURL) {
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

function scoreImage(imageURL) {
  return withForwardSlot(() => scoreImageUnqueued(imageURL));
}

async function run(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await scoreImage(imageURL);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, ...r, ms, device });
}

// Batch screen many images (each is a data URL). Returns per-item flag against a threshold.
// Results are written by index so the bound never reorders them, and one response is posted per
// request id, as the client expects.
async function screen(id, images, threshold) {
  await ensureLoaded();
  const t0 = performance.now();
  const items = new Array(images.length);
  for (let i = 0; i < images.length; i++) {
    const { key, url } = images[i];
    const r = await scoreImage(url);
    items[i] = { key, ai: r.ai, real: r.real, flagged: r.ai >= threshold };
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
