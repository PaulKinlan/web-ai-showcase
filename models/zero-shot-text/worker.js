// DeBERTa-v3 zero-shot text classification worker — inference off the main thread.
// Model: Xenova/nli-deberta-v3-xsmall (task: zero-shot-classification), WASM, q8.
//
// Zero-shot via NLI (natural language inference): to score whether a TEXT belongs to a LABEL, the model
// is asked "does the text ENTAIL the hypothesis 'This example is {label}.'?" It outputs three logits —
// contradiction / neutral / entailment — and the entailment probability becomes the label's score. No
// training, no fixed label set: you invent the labels at run time. We use the pipeline for the ranked
// scores and reuse the same model instance (pipe.model + pipe.tokenizer) to expose the raw 3-way NLI
// logits per hypothesis for the "See inside" surface.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/nli-deberta-v3-xsmall";
const HYPOTHESIS_TEMPLATE = "This example is {}.";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "zero-shot-classification",
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
  const exp = arr.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map((x) => x / sum);
}

// Raw 3-way NLI logits for (text, "This example is {label}.") using the shared model instance.
async function nliLogits(text, label) {
  const tok = pipe.tokenizer;
  const model = pipe.model;
  const hypothesis = HYPOTHESIS_TEMPLATE.replace("{}", label);
  const enc = tok(text, { text_pair: hypothesis, padding: true, truncation: true });
  const out = await model(enc);
  return Array.from(out.logits.data, Number); // [contradiction, neutral, entailment] per config.id2label
}

async function classify(id, text, labels, multiLabel) {
  await ensureLoaded();
  const t0 = performance.now();

  const out = await pipe(text, labels, {
    hypothesis_template: HYPOTHESIS_TEMPLATE,
    multi_label: !!multiLabel,
  });
  // pipeline returns { sequence, labels (sorted), scores (aligned) }.
  const scored = out.labels.map((label, i) => ({ label, score: out.scores[i] }));

  // See-inside: the raw NLI 3-way distribution for each label, so you can watch entailment win/lose.
  const id2label = pipe.model.config.id2label ||
    { 0: "contradiction", 1: "neutral", 2: "entailment" };
  const classNames = Object.keys(id2label).sort((a, b) => a - b).map((k) => id2label[k]);
  const nli = [];
  for (const { label } of scored) {
    const logits = await nliLogits(text, label);
    nli.push({ label, logits, probs: softmax(logits) });
  }

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    multiLabel: !!multiLabel,
    scored,
    nli,
    classNames,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") {
      await classify(e.data.id, e.data.text, e.data.labels, e.data.multiLabel);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
