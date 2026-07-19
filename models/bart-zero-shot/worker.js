// BART-large-MNLI zero-shot text classification worker — inference off the main thread.
// Model: Xenova/bart-large-mnli (task: zero-shot-classification), WASM, q4f16 (model_q4f16.onnx ~309 MB).
//
// DTYPE NOTE: we ship q4f16, NOT the usual q8. Measured in-browser, the q8 (int8) build of this model is
// badly degraded — it returns near-uniform scores and frequently picks the wrong label (e.g. "the phone
// overheats" → "billing problem"). q4f16 (4-bit weights + fp16 compute) restores full discrimination
// (e.g. → "hardware fault") at a SMALLER download than q8, and still runs on WASM. This is the smallest
// dtype that genuinely works for this model.
//
// This is THE canonical zero-shot classifier: facebook/bart-large fine-tuned on the MultiNLI entailment
// dataset. Zero-shot works via natural-language inference (NLI): to score whether a TEXT belongs to a
// LABEL, the model is asked "does the text ENTAIL the hypothesis 'This example is {label}.'?" It outputs
// three logits — contradiction / neutral / entailment — and the entailment probability becomes the
// label's score. No training, no fixed label set: you invent the labels at run time.
//
// We use the pipeline for the ranked scores and reuse the same model instance (pipe.model +
// pipe.tokenizer) to expose the raw 3-way NLI logits per hypothesis for the "See inside" surface.
//
// Contrast mode: on demand we ALSO load Xenova/nli-deberta-v3-xsmall (~70 MB, the DeBERTa page's model)
// so you can run the SAME text + labels through both zero-shot backbones and compare their rankings.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const BART_ID = "Xenova/bart-large-mnli";
const DEBERTA_ID = "Xenova/nli-deberta-v3-xsmall";
const HYPOTHESIS_TEMPLATE = "This example is {}.";

let bart = null; // { pipe, device }
let deberta = null; // { pipe, device } — lazy, contrast only

function post(msg) {
  self.postMessage(msg);
}

async function ensureBart() {
  if (bart) return;
  const loaded = await loadPipeline({
    task: "zero-shot-classification",
    model: BART_ID,
    backend: "wasm",
    dtype: "q4f16",
    onProgress: (p) => post({ type: "progress", p }),
  });
  bart = loaded;
  post({ type: "ready", device: loaded.device });
}

async function ensureDeberta() {
  if (deberta) return;
  const loaded = await loadPipeline({
    task: "zero-shot-classification",
    model: DEBERTA_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "cmpProgress", p }),
  });
  deberta = loaded;
  post({ type: "cmpReady", device: loaded.device });
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map((x) => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0) || 1;
  return exp.map((x) => x / sum);
}

// Class names in id order, e.g. ["contradiction","neutral","entailment"].
function classNamesOf(pipe) {
  const id2label = pipe.model.config.id2label ||
    { 0: "contradiction", 1: "neutral", 2: "entailment" };
  return Object.keys(id2label).sort((a, b) => a - b).map((k) => id2label[k]);
}

// Raw 3-way NLI logits for (text, "This example is {label}.") using a shared model instance.
async function nliLogits(pipe, text, label) {
  const tok = pipe.tokenizer;
  const model = pipe.model;
  const hypothesis = HYPOTHESIS_TEMPLATE.replace("{}", label);
  const enc = tok(text, { text_pair: hypothesis, padding: true, truncation: true });
  const out = await model(enc);
  return Array.from(out.logits.data, Number);
}

async function rank(pipe, text, labels, multiLabel) {
  const out = await pipe(text, labels, {
    hypothesis_template: HYPOTHESIS_TEMPLATE,
    multi_label: !!multiLabel,
  });
  return out.labels.map((label, i) => ({ label, score: out.scores[i] }));
}

async function classify(id, text, labels, multiLabel) {
  await ensureBart();
  const t0 = performance.now();
  const scored = await rank(bart.pipe, text, labels, multiLabel);

  // See-inside: the raw NLI 3-way distribution for each label, so you can watch entailment win/lose.
  const classNames = classNamesOf(bart.pipe);
  const nli = [];
  for (const { label } of scored) {
    const logits = await nliLogits(bart.pipe, text, label);
    nli.push({ label, logits, probs: softmax(logits) });
  }

  post({
    type: "result",
    id,
    text,
    multiLabel: !!multiLabel,
    scored,
    nli,
    classNames,
    ms: Math.round(performance.now() - t0),
    device: bart.device,
  });
}

// Run the SAME text + labels through BOTH backbones for a side-by-side contrast.
async function compare(id, text, labels, multiLabel) {
  await ensureBart();
  await ensureDeberta();
  const tB = performance.now();
  const bartScored = await rank(bart.pipe, text, labels, multiLabel);
  const msBart = Math.round(performance.now() - tB);
  const tD = performance.now();
  const debScored = await rank(deberta.pipe, text, labels, multiLabel);
  const msDeb = Math.round(performance.now() - tD);
  post({
    type: "cmpResult",
    id,
    text,
    multiLabel: !!multiLabel,
    bart: { scored: bartScored, ms: msBart, device: bart.device, params: "406M", model: BART_ID },
    deberta: {
      scored: debScored,
      ms: msDeb,
      device: deberta.device,
      params: "22M",
      model: DEBERTA_ID,
    },
  });
}

// Classify with a chosen backbone only ("bart" | "deberta") — used by the multi-model cascade page,
// where BART routes to a coarse bucket and DeBERTa does the fast fine-grained sub-classification.
async function classifyWith(id, which, text, labels, multiLabel) {
  let pipe, device;
  if (which === "deberta") {
    await ensureDeberta();
    pipe = deberta.pipe;
    device = deberta.device;
  } else {
    await ensureBart();
    pipe = bart.pipe;
    device = bart.device;
  }
  const t0 = performance.now();
  const scored = await rank(pipe, text, labels, multiLabel);
  post({
    type: "result",
    id,
    which,
    text,
    multiLabel: !!multiLabel,
    scored,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureBart();
    else if (type === "run") {
      await classify(e.data.id, e.data.text, e.data.labels, e.data.multiLabel);
    } else if (type === "loadCompare") await ensureDeberta();
    else if (type === "runWith") {
      await classifyWith(e.data.id, e.data.which, e.data.text, e.data.labels, e.data.multiLabel);
    } else if (type === "compare") {
      await compare(e.data.id, e.data.text, e.data.labels, e.data.multiLabel);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
