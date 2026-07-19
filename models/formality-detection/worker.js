// Formality (register) detection worker — inference off the main thread so the meter stays smooth.
// Model: Deepchecks/roberta_base_formality_ranker_onnx (an ONNX export of s-nlp/roberta-base-
// formality-ranker), task: text-classification, WASM. Labels: {informal, formal}.
//
// NOTE ON LOADING: this repo ships its single fp32 graph as `model_optimized.onnx` at the repo ROOT
// (not the usual onnx/model.onnx layout), so we call pipeline() DIRECTLY with `subfolder: ""` +
// `model_file_name: "model_optimized"`. That's why this worker imports transformers.js itself instead
// of the shared loadPipeline helper (which assumes the standard onnx/ layout + a q8 variant). There is
// no quantised export, so this loads fp32 (~260 MB). Everything still runs locally; nothing is uploaded.
//
// Operations:
//   run       → classify one text → formal vs informal probabilities.
//   attribute → OCCLUSION attribution: re-score with each word removed to reveal which words carry the
//               formal / informal register signal (a real, model-grounded saliency; N+1 forward passes).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Deepchecks/roberta_base_formality_ranker_onnx";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  pipe = await pipeline("text-classification", MODEL_ID, {
    subfolder: "",
    model_file_name: "model_optimized",
    dtype: "fp32",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}

// Recover the "formal" probability from a result row (topk may return only the winning label).
function formalProb(row) {
  const arr = Array.isArray(row) ? row : [row];
  const f = arr.find((r) => r.label === "formal");
  if (f) return f.score;
  const i = arr.find((r) => r.label === "informal");
  return i ? 1 - i.score : 0.5;
}

// Log-odds of the formal class — keeps occlusion deltas legible even when a score is near-saturated.
function logit(p) {
  const c = Math.min(Math.max(p, 1e-6), 1 - 1e-6);
  return Math.log(c / (1 - c));
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 2 });
  const formal = formalProb(out);
  post({
    type: "result",
    id,
    text,
    formal,
    informal: 1 - formal,
    label: formal >= 0.5 ? "formal" : "informal",
    ms: Math.round(performance.now() - t0),
    device,
  });
}

// Split into re-joinable words; trailing punctuation rides with the word.
function tokenize(text) {
  return text.split(/(\s+)/).filter((t) => t.trim().length > 0);
}

async function attribute(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const words = tokenize(text);
  if (words.length === 0) {
    post({
      type: "attr",
      id,
      text,
      words: [],
      attributions: [],
      formal: 0.5,
      label: "neutral",
      ms: 0,
      device,
    });
    return;
  }
  // Batch: the full text plus one variant per word with that word removed.
  const variants = [text, ...words.map((_, i) => words.filter((_, j) => j !== i).join(" "))];
  const out = await pipe(variants, { top_k: 2 });
  const fullFormal = formalProb(out[0]);
  const fullLogit = logit(fullFormal);
  // attribution_i = logit_formal(full) - logit_formal(without word i): positive ⇒ removing the word
  // dropped the formal score, so that word pushed FORMAL; negative ⇒ it pushed INFORMAL.
  const attributions = words.map((_, i) => fullLogit - logit(formalProb(out[i + 1])));
  post({
    type: "attr",
    id,
    text,
    words,
    attributions,
    formal: fullFormal,
    label: fullFormal >= 0.5 ? "formal" : "informal",
    ms: Math.round(performance.now() - t0),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "attribute") await attribute(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
