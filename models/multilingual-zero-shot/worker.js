// Multilingual zero-shot text classification worker — inference off the main thread.
// Model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli (task: zero-shot-classification), WASM, q8.
// ONNX build shipped in the canonical repo (onnx/model_quantized.onnx).
//
// What makes THIS demo distinct from the English DeBERTa zero-shot page: the backbone is
// mDeBERTa-v3-base (a 250k SentencePiece vocab spanning 100 languages) fine-tuned on MNLI + XNLI, so it
// scores labels for text in MANY languages against categories you invent at run time — no per-language
// model, no training. Zero-shot via NLI: each label becomes a hypothesis ("This example is {label}.")
// and the model's ENTAILMENT probability is the label's score. It outputs three logits per pass —
// entailment / neutral / contradiction (note mDeBERTa's config order puts entailment at index 0) — and
// we expose the raw 3-way distribution for the "See inside" surface. Because XNLI aligns hypotheses
// across languages, you can even write the hypothesis TEMPLATE in the text's own language for sharper,
// better-calibrated scores — a control this multilingual page adds over the English one.

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

// Raw 3-way NLI logits for (text, hypothesis) using the shared model instance — the "see inside" surface.
async function nliLogits(text, hypothesis) {
  const tok = pipe.tokenizer;
  const model = pipe.model;
  const enc = tok(text, { text_pair: hypothesis, padding: true, truncation: true });
  const out = await model(enc);
  return Array.from(out.logits.data, Number); // ordered per config.id2label
}

async function classify(id, text, labels, multiLabel, template) {
  await ensureLoaded();
  const t0 = performance.now();
  const tpl = template && template.includes("{}") ? template : DEFAULT_TEMPLATE;

  const out = await pipe(text, labels, {
    hypothesis_template: tpl,
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
    const hypothesis = tpl.replace("{}", label);
    const logits = await nliLogits(text, hypothesis);
    nli.push({ label, logits, probs: softmax(logits) });
  }

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    template: tpl,
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
