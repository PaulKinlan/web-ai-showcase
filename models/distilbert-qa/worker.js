// DistilBERT SQuAD question-answering worker — inference off the main thread so the UI stays smooth.
// Model: Xenova/distilbert-base-cased-distilled-squad (task: question-answering), WASM, q8.
//
// Extractive QA: the model reads a QUESTION and a CONTEXT passage and predicts two distributions over
// the context tokens — where the answer STARTS and where it ENDS. The answer is the span that maximises
// start_logit(i) + end_logit(j) for i <= j. We surface two things:
//   run     → the pipeline's clean answer span (char offsets, so we can highlight it) + top-k candidates.
//   inspect → the RAW start/end logits over every token, run through the same single model instance
//             (pipe.model + pipe.tokenizer) so "See inside" shows the model's real internal reasoning.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/distilbert-base-cased-distilled-squad";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "question-answering",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

function softmaxOver(logits, from, to) {
  let max = -Infinity;
  for (let i = from; i <= to; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const probs = new Array(logits.length).fill(0);
  for (let i = from; i <= to; i++) {
    const e = Math.exp(logits[i] - max);
    probs[i] = e;
    sum += e;
  }
  for (let i = from; i <= to; i++) probs[i] /= sum || 1;
  return probs;
}

// One pass: the pipeline answer(s) for a clean, highlightable span, PLUS the raw start/end token
// distributions from the same model instance for the "See inside" surface.
async function answer(id, question, context, topk) {
  await ensureLoaded();
  const t0 = performance.now();

  // Transformers.js expects `top_k` (not `topk`). The QA pipeline returns { answer, score } only — no
  // char offsets — so we locate each extracted span back in the context ourselves for highlighting.
  const out = await pipe(question, context, { top_k: topk || 5 });
  const answers = (Array.isArray(out) ? out : [out])
    .filter((a) => a && a.answer != null)
    .map((a) => {
      const start = context.indexOf(a.answer);
      return {
        answer: a.answer,
        score: a.score,
        start,
        end: start >= 0 ? start + a.answer.length : -1,
      };
    });

  // Raw internals from the shared model instance — no second copy of the weights.
  const tok = pipe.tokenizer;
  const model = pipe.model;
  const enc = tok(question, { text_pair: context, padding: true, truncation: true });
  const logits = await model({ input_ids: enc.input_ids, attention_mask: enc.attention_mask });
  const startLogits = Array.from(logits.start_logits.data, Number);
  const endLogits = Array.from(logits.end_logits.data, Number);
  const ids = Array.from(enc.input_ids.data, Number);

  // Sequence is [CLS] question [SEP] context [SEP]. The context region sits between the two [SEP]s.
  const sep = tok.sep_token_id;
  const sepIdx = [];
  for (let i = 0; i < ids.length; i++) if (ids[i] === sep) sepIdx.push(i);
  const ctxStart = (sepIdx[0] ?? 0) + 1;
  const ctxEnd = (sepIdx[1] ?? ids.length - 1) - 1;

  const tokens = ids.map((tid) => tok.decode([tid], { skip_special_tokens: false }));
  const startProbs = softmaxOver(startLogits, ctxStart, ctxEnd);
  const endProbs = softmaxOver(endLogits, ctxStart, ctxEnd);

  // Argmax start/end within the context region (for the highlighted token markers).
  let argStart = ctxStart, argEnd = ctxStart;
  for (let i = ctxStart; i <= ctxEnd; i++) {
    if (startProbs[i] > startProbs[argStart]) argStart = i;
    if (endProbs[i] > endProbs[argEnd]) argEnd = i;
  }

  // "No-answer" signal: SQuAD-v1 models can't say "unanswerable", but a diffuse start distribution
  // (low peak probability) is a real, model-grounded proxy for low confidence / likely-unanswerable.
  const startPeak = startProbs[argStart];
  const endPeak = endProbs[argEnd];

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    question,
    context,
    answers,
    tokens: tokens.slice(ctxStart, ctxEnd + 1),
    startProbs: startProbs.slice(ctxStart, ctxEnd + 1),
    endProbs: endProbs.slice(ctxStart, ctxEnd + 1),
    argStart: argStart - ctxStart,
    argEnd: argEnd - ctxStart,
    startPeak,
    endPeak,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await answer(e.data.id, e.data.question, e.data.context, e.data.topk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
