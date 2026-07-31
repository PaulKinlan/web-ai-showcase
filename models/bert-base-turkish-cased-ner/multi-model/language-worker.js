// XLM-RoBERTa language-identification worker — inference off the main thread.
// Exact stage: onnx-community/xlm-roberta-base-language-detection-ONNX @
// 919c87aa2749131ae1ab709931a16bf1cc9774ea, q8 onnx/model_quantized.onnx,
// 278,836,241 bytes, SHA-256 e3a2f1b44ea6a76683e4655127531b05c6b568fe643bd39bddf7f7c62ab182c9.
// This is the papluca/xlm-roberta-base-language-detection model exported for browser ONNX inference.
// It classifies 20 languages and returns the full softmax distribution.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/xlm-roberta-base-language-detection-ONNX";
const REVISION = "919c87aa2749131ae1ab709931a16bf1cc9774ea";
const MODEL_BYTES = 278836241;
const MODEL_SHA256 = "e3a2f1b44ea6a76683e4655127531b05c6b568fe643bd39bddf7f7c62ab182c9";
const N_LABELS = 20;

let pipe = null;
let device = "wasm";
const cancelled = new Set();
let activeJob = null;
let queuedJob = null; // one running + one waiting, latest request wins

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "text-classification",
    model: "onnx-community/xlm-roberta-base-language-detection-ONNX",
    revision: "919c87aa2749131ae1ab709931a16bf1cc9774ea",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({
    type: "ready",
    device,
    revision: REVISION,
    modelBytes: MODEL_BYTES,
    modelSha256: MODEL_SHA256,
  });
}

async function execute(job) {
  await ensureLoaded();
  if (cancelled.delete(job.id)) return;
  const t0 = performance.now();
  const scores = await pipe(job.text, { top_k: N_LABELS });
  if (!cancelled.delete(job.id)) {
    post({
      type: "result",
      id: job.id,
      text: job.text,
      scores,
      ms: Math.round(performance.now() - t0),
      device,
    });
  }
}

async function drain() {
  if (activeJob || !queuedJob) return;
  activeJob = queuedJob;
  queuedJob = null;
  try {
    await execute(activeJob);
  } catch (error) {
    if (!cancelled.delete(activeJob.id)) {
      post({ type: "error", id: activeJob.id, message: String(error?.message ?? error) });
    }
  } finally {
    activeJob = null;
    void drain();
  }
}

function enqueue(job) {
  if (queuedJob) cancelled.delete(queuedJob.id);
  queuedJob = job;
  void drain();
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "load") await ensureLoaded();
    else if (data.type === "cancel") cancelled.add(data.id);
    else if (data.type === "run") enqueue(data);
  } catch (error) {
    post({ type: "error", id: data?.id, message: String(error?.message ?? error) });
  }
});
