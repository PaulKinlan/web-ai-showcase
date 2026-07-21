// Code language identification worker — inference off the main thread so typing stays smooth.
// Model: onnx-community/CodeBERTa-language-id-ONNX (task: text-classification), WASM, q8 (~84 MB).
// CodeBERTa (a RoBERTa pre-trained on CodeSearchNet) fine-tuned to name the programming LANGUAGE of a
// snippet — one of go, java, javascript, php, python, ruby. DISTINCT from the built code models
// (codebert-fill-mask predicts masked tokens, code-embeddings does retrieval, starcoder-fim completes code):
// this classifies WHICH LANGUAGE the code is. Apache-2.0. We import the SHARED loader from lib/webai.js —
// no invented API. Nothing leaves the tab.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/CodeBERTa-language-id-ONNX";
const TASK = "text-classification";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
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

// Classify a code snippet → the full 6-language probability distribution (top_k = 6).
async function classify(id, code) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(code, { top_k: 6 });
  const labels = (Array.isArray(out) ? out : [out]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "classify") await classify(d.id, d.code);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
