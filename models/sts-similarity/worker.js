// STS-B semantic textual similarity worker — inference off the main thread so the UI stays smooth.
// Model: cross-encoder/stsb-TinyBERT-L4 (task: text-classification, num_labels=1 → regression), WASM, fp32.
//
// This is a CROSS-ENCODER regressor. Unlike a bi-encoder (which embeds each sentence separately and
// compares vectors) it reads BOTH sentences together — [CLS] sentence-A [SEP] sentence-B [SEP] — and
// emits ONE number: a graded similarity. It was trained on STS-B, whose human labels run 0–5, normalised
// to [0,1]; so the calibrated similarity is sigmoid(logit) and the human-scale score is sigmoid(logit)·5.
// We read the RAW regression logit straight off the model, then expose logit → sigmoid → 0–5 so you can
// see exactly how the single output becomes the gauge. No pipeline squashing hidden from view.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "cross-encoder/stsb-TinyBERT-L4";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "text-classification",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "fp32", // this repo ships a standard fp32 ONNX (~58 MB); its int8 builds use non-standard names
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Score one or more sentence pairs in a single batched forward pass. Each pair is [a, b].
async function score(id, pairs) {
  await ensureLoaded();
  const t0 = performance.now();
  const tok = pipe.tokenizer;
  const model = pipe.model;

  const aTexts = pairs.map((p) => p[0]);
  const bTexts = pairs.map((p) => p[1]);
  // Joint encoding: each A is paired with its B as [CLS] A [SEP] B [SEP].
  const enc = tok(aTexts, { text_pair: bTexts, padding: true, truncation: true });
  const out = await model(enc);
  const logitsT = out.logits;
  const nCols = logitsT.dims[logitsT.dims.length - 1]; // 1 for this regression head
  const flat = Array.from(logitsT.data, Number);

  const results = pairs.map((pair, i) => {
    const logit = flat[i * nCols];
    const sim = sigmoid(logit); // calibrated 0–1
    return {
      a: pair[0],
      b: pair[1],
      logit,
      sim, // 0–1
      score5: sim * 5, // human STS-B scale 0–5
    };
  });

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, results, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await score(e.data.id, e.data.pairs);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
