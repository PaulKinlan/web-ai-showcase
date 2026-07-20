// DiT document-image-classification worker — runs ALL inference off the main thread so the control UI
// stays responsive. Like the ViT / Food demos it runs the LOW-LEVEL forward pass (processor → model)
// instead of the convenience pipeline, so a single pass yields everything the surfaces need: the raw
// pre-softmax logits, the full softmax distribution over ALL 16 document classes (small enough to show
// in full), the human-readable class names, and confidence diagnostics (entropy, top-1 margin).
//
// Model: Xenova/dit-base-finetuned-rvlcdip (task: image-classification), WASM backend, q8
// (model_quantized.onnx). DiT (Document Image Transformer) is a BEiT-architecture ViT *self-supervised
// pre-trained on 42M document images* (masked image modelling over a discrete VAE codebook), then
// fine-tuned on RVL-CDIP — 16 document types (letter, form, email, handwritten, advertisement,
// scientific report, scientific publication, specification, file folder, news article, budget, invoice,
// presentation, questionnaire, resume, memo). Because it is pre-trained on documents (not natural
// photos), it reads page LAYOUT — margins, tables, letterheads, columns — rather than object texture.
//
// Protocol (shared with lib/classify-ui.js ClassifierEngine):
//   → { type: "load" } / { type: "run", id, image, topK }
//   ← { type: "progress", p } / { type: "ready", device }
//   ← { type: "result", id, top, all, entropy, margin, numClasses, ms, device }
//   ← { type: "error", id?, message }

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/dit-base-finetuned-rvlcdip";

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
    backend: "wasm", // BEiT/DiT q8 runs on plain WebAssembly — no WebGPU required
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

// Shannon entropy of the distribution, normalised to [0,1] against log(numClasses). High = uncertain.
function normEntropy(probs) {
  let h = 0;
  for (const p of probs) if (p > 0) h -= p * Math.log(p);
  return h / Math.log(probs.length);
}

async function run(id, imageURL, topK) {
  await ensureLoaded();
  const t0 = performance.now();
  const image = await RawImage.read(imageURL);
  const inputs = await pipe.processor(image);
  const output = await pipe.model(inputs);

  const logits = Array.from(output.logits.data); // [16] raw, pre-softmax
  const probs = softmax(logits);
  const id2label = pipe.model.config.id2label || {};

  // The FULL distribution — only 16 classes, so we return every one (sorted) for the "see inside" view.
  const all = probs
    .map((p, i) => ({ label: id2label[i] ?? `class ${i}`, prob: p, logit: logits[i], index: i }))
    .sort((a, b) => b.prob - a.prob);

  const k = Math.max(1, Math.min(16, topK || 5));
  const top = all.slice(0, k);

  // Confidence diagnostics: how peaked is the distribution?
  const sorted = [...probs].sort((a, b) => b - a);
  const margin = sorted[0] - (sorted[1] ?? 0); // top-1 vs top-2 gap
  const entropy = normEntropy(probs);

  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, top, all, entropy, margin, numClasses: probs.length, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.topK);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
