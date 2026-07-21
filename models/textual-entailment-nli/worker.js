// Natural-language-inference (textual entailment) worker — runs inference off the main thread.
//
// Model: onnx-community/nli-deberta-v3-xsmall-ONNX (a cross-encoder NLI model; task text-classification),
// WASM backend, q8 (model_quantized.onnx). It reads a PAIR of sentences — a premise and a hypothesis — and
// predicts their logical relationship: entailment / neutral / contradiction. DISTINCT from the built
// zero-shot-classification demos, which apply an NLI model to LABEL a single text (premise + "This example
// is {label}" hypotheses): here the hypothesis is a free-form sentence, so the demo shows the atomic
// reasoning behind fact-checking, RAG grounding ("does this evidence support the claim?"), and contradiction
// detection. The weights are Apache-2.0 (base cross-encoder/nli-deberta-v3-xsmall, itself a fine-tune of
// Apache-2.0 microsoft/deberta-v3-xsmall); Apache-2.0 permits redistribution, so they stay Apache-2.0 in
// the onnx-community conversion despite its blank license field.
//
// WHY THE LOW-LEVEL PATH (not pipeline()): the transformers.js text-classification pipeline encodes only a
// single text, so a sentence PAIR is ignored (verified: all pairs gave the same output). We tokenize the
// pair explicitly with tokenizer(premise, { text_pair: hypothesis }) and run the model, then softmax the
// logits over the three classes.
//
// Correctness proven FIRST in headless Chrome (transformers.js 3.7.5, WASM, q8): "A man is playing a guitar
// on stage." / "A person is making music." -> entailment 0.98; / "The man is asleep in bed." ->
// contradiction 1.00; / "The man is a famous musician." -> neutral 1.00; "Eiffel Tower is in Paris." /
// "Eiffel Tower is in Berlin." -> contradiction 1.00. Nothing leaves the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/nli-deberta-v3-xsmall-ONNX";

let tokenizer = null;
let model = null;
let id2label = null;

function post(msg) {
  self.postMessage(msg);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoTokenizer, AutoModelForSequenceClassification, env } = mod;
  env.allowLocalModels = false;
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForSequenceClassification.from_pretrained(MODEL_ID, {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  id2label = model.config.id2label;
  post({ type: "ready", device: "wasm" });
}

// Classify a premise/hypothesis pair → { scores:[{label,prob}], top, ms }.
async function run(id, premise, hypothesis) {
  await ensureLoaded();
  const t0 = performance.now();
  const inputs = await tokenizer(premise, { text_pair: hypothesis });
  const { logits } = await model(inputs);
  const probs = softmax(Array.from(logits.data));
  const scores = probs
    .map((prob, i) => ({ label: id2label[i], prob }))
    .sort((a, b) => b.prob - a.prob);
  post({ type: "result", id, scores, top: scores[0], ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.premise, e.data.hypothesis);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
