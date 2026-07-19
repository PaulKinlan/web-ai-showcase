// XLM-RoBERTa language-identification worker — inference off the main thread.
// Model: onnx-community/xlm-roberta-base-language-detection-ONNX (text-classification), WASM, q8.
// This is the papluca/xlm-roberta-base-language-detection model exported to ONNX for the browser.
//
// XLM-RoBERTa is a multilingual encoder trained on 100 languages; this fine-tune classifies text into
// one of 20 languages. Because every language shares ONE embedding space, the model reads the actual
// linguistic signal (script, morphology, function words), not a per-language keyword list — so it can
// label short snippets and stays honest (spreads probability) on ambiguous or code-switched input. We
// return the FULL softmax distribution so the page can show every language's probability, not just the top.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/xlm-roberta-base-language-detection-ONNX";
const N_LABELS = 20;

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
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  // top_k = N_LABELS returns the full distribution, sorted high→low.
  const out = await pipe(text, { top_k: N_LABELS });
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text, scores: out, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
