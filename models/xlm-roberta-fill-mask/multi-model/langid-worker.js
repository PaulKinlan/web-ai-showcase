// Language-identification worker for the XLM-RoBERTa multi-model demo. Off the main thread.
// Model: onnx-community/xlm-roberta-base-language-detection-ONNX (task: text-classification), WASM q8.
// Fittingly, the language detector is ALSO an XLM-RoBERTa — same architecture as the fill-mask model,
// fine-tuned to classify which of 20 languages a piece of text is in.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/xlm-roberta-base-language-detection-ONNX";
let classifier = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (classifier) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  classifier = await pipeline("text-classification", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "wasm" });
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await classifier(text, { top_k: 3 });
  const scores = (Array.isArray(out) ? out : [out]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "langid", id, scores, ms: Math.round(performance.now() - t0), device: "wasm" });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "classify") await classify(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
