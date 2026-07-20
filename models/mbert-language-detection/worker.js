// mBERT language-detection worker — inference off the main thread.
// Model: onnx-community/language_detection-ONNX (text-classification), WASM, q8. This is a BERT
// (BertForSequenceClassification) fine-tune — base alexneakameni/language_detection — exported to ONNX.
//
// A DISTINCT model from the built XLM-RoBERTa "language-identification" demo: a different architecture
// (multilingual BERT, not XLM-RoBERTa) with a MUCH wider label set — 201 languages using FLORES-200
// codes (e.g. fra_Latn, deu_Latn, zho_Hans) rather than 20 ISO-639-1 codes. The classification head
// produces a real logit per class ([1, 201]); we softmax + return the FULL distribution so the page can
// show every language's probability, not just the top. Because it reads the actual linguistic signal
// (script, morphology, function words) it stays honest — spreading probability on short/ambiguous text.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/language_detection-ONNX";
const N_LABELS = 201;

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
