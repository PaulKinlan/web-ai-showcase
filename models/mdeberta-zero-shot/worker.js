// mDeBERTa-v3 multilingual zero-shot text classification worker — inference off the main thread.
// Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (task: zero-shot-classification), WASM, q8.
// No version pin — model_type "deberta-v2" loads on the existing zero-shot-classification pipeline in
// transformers.js 3.7.5.
//
// Zero-shot via NLI, but MULTILINGUAL. mDeBERTa-v3-base was pretrained on 100 languages (CC100) and
// fine-tuned on MNLI (English) + XNLI (15 languages) natural-language-inference data. To score whether a
// TEXT belongs to a LABEL, the model is asked "does the text ENTAIL the hypothesis '{template with
// label}'?" It outputs three logits — entailment / neutral / contradiction — and the entailment
// probability becomes the label's score. Because the encoder is multilingual, the TEXT, the LABELS, and
// the hypothesis TEMPLATE can each be in (almost) any language — including cross-lingual mixes (English
// labels on Japanese text). We use the pipeline for the ranked scores and reuse the same model instance
// (pipe.model + pipe.tokenizer) to expose the raw 3-way NLI logits per hypothesis for "See inside".

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli";
const DEFAULT_TEMPLATE = "This example is {}.";

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

// Raw 3-way NLI logits for (text, hypothesis) using the shared model instance.
async function nliLogits(text, hypothesis) {
  const tok = pipe.tokenizer;
  const model = pipe.model;
  const enc = tok(text, { text_pair: hypothesis, padding: true, truncation: true });
  const out = await model(enc);
  return Array.from(out.logits.data, Number); // order per config.id2label
}

async function classify(id, text, labels, multiLabel, template) {
  await ensureLoaded();
  const t0 = performance.now();
  const hypothesisTemplate = template && template.includes("{}") ? template : DEFAULT_TEMPLATE;

  const out = await pipe(text, labels, {
    hypothesis_template: hypothesisTemplate,
    multi_label: !!multiLabel,
  });
  // pipeline returns { sequence, labels (sorted), scores (aligned) }.
  const scored = out.labels.map((label, i) => ({ label, score: out.scores[i] }));

  // See-inside: the raw NLI 3-way distribution for each label, so you can watch entailment win/lose.
  const id2label = pipe.model.config.id2label ||
    { 0: "entailment", 1: "neutral", 2: "contradiction" };
  const classNames = Object.keys(id2label).sort((a, b) => a - b).map((k) => id2label[k]);
  const nli = [];
  for (const { label } of scored) {
    const hypothesis = hypothesisTemplate.replace("{}", label);
    const logits = await nliLogits(text, hypothesis);
    nli.push({ label, hypothesis, logits, probs: softmax(logits) });
  }

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    template: hypothesisTemplate,
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
      await classify(e.data.id, e.data.text, e.data.labels, e.data.multiLabel, e.data.template);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
