// Multilingual extractive question-answering worker — inference off the main thread so the UI stays smooth.
// Model: onnx-community/xlm-roberta-base-finetuned-squad2-ONNX (task: question-answering), WASM, q8.
//
// XLM-RoBERTa is pretrained on 100 languages, so this SQuAD2-finetuned head extracts an answer SPAN out
// of a passage in many languages — the question and the context can be non-English, and (thanks to the
// shared multilingual representation) an English question can even hit a non-English context.
//
// Why we don't use the stock question-answering pipeline's answer string: for SentencePiece models
// (XLM-R) the pipeline detokenises the span WITHOUT restoring inter-word spaces ("330metres"), so it
// can't be located back in the passage to highlight. Instead we run the shared model instance directly,
// read the raw start/end logits, pick the best span, decode it, and re-locate it in the ORIGINAL passage
// with a whitespace-insensitive scan — giving a correctly-spaced answer AND clean char offsets to
// highlight, plus the real start/end token distributions for the "See inside" surface. One pass, one
// copy of the weights.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/xlm-roberta-base-finetuned-squad2-ONNX";
const MAX_SPAN = 26; // answers longer than this many tokens are almost never right for extractive QA

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

// Locate an answer string back in the ORIGINAL context, ignoring whitespace differences (SentencePiece
// decode drops inter-subword spaces for Latin scripts). Returns char offsets into the original context
// so the highlight and the displayed answer keep the passage's real spacing/casing.
function locate(context, answer) {
  const noWs = answer.replace(/\s+/g, "");
  if (!noWs) return null;
  const idx = [];
  let stripped = "";
  for (let k = 0; k < context.length; k++) {
    if (!/\s/.test(context[k])) {
      idx.push(k);
      stripped += context[k];
    }
  }
  const pos = stripped.toLowerCase().indexOf(noWs.toLowerCase());
  if (pos < 0) return null;
  return { start: idx[pos], end: idx[pos + noWs.length - 1] + 1 };
}

async function answer(id, question, context, topk) {
  await ensureLoaded();
  const t0 = performance.now();

  const tok = pipe.tokenizer;
  const model = pipe.model;
  const enc = tok(question, { text_pair: context, padding: true, truncation: true });
  const logits = await model({ input_ids: enc.input_ids, attention_mask: enc.attention_mask });
  const startLogits = Array.from(logits.start_logits.data, Number);
  const endLogits = Array.from(logits.end_logits.data, Number);
  const ids = Array.from(enc.input_ids.data, Number);

  // XLM-R sequence: <s> question </s></s> context </s>. The context region sits between the second
  // </s> of the separator pair and the final </s>.
  const sep = tok.sep_token_id;
  const sepIdx = [];
  for (let i = 0; i < ids.length; i++) if (ids[i] === sep) sepIdx.push(i);
  const ctxStart = (sepIdx.length >= 2 ? sepIdx[1] : (sepIdx[0] ?? 0)) + 1;
  const ctxEnd = (sepIdx.length ? sepIdx[sepIdx.length - 1] : ids.length - 1) - 1;

  const startProbs = softmaxOver(startLogits, ctxStart, ctxEnd);
  const endProbs = softmaxOver(endLogits, ctxStart, ctxEnd);

  // Argmax start/end (for the token-strip markers) within the context region.
  let argStart = ctxStart, argEnd = ctxStart;
  for (let i = ctxStart; i <= ctxEnd; i++) {
    if (startProbs[i] > startProbs[argStart]) argStart = i;
    if (endProbs[i] > endProbs[argEnd]) argEnd = i;
  }

  // Score every candidate span i<=j (bounded length) by P(start)·P(end); keep the best few distinct
  // answers, decoding + re-locating each in the original passage.
  const scored = [];
  for (let i = ctxStart; i <= ctxEnd; i++) {
    if (startProbs[i] < 1e-4) continue;
    for (let j = i; j <= Math.min(ctxEnd, i + MAX_SPAN); j++) {
      scored.push({ i, j, score: startProbs[i] * endProbs[j] });
    }
  }
  scored.sort((a, b) => b.score - a.score);

  const answers = [];
  const seen = new Set();
  for (const s of scored) {
    if (answers.length >= (topk || 5)) break;
    const decoded = tok.decode(ids.slice(s.i, s.j + 1), { skip_special_tokens: true }).trim();
    if (!decoded) continue;
    const loc = locate(context, decoded);
    const clean = loc ? context.slice(loc.start, loc.end) : decoded;
    const key = clean.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    answers.push({
      answer: clean,
      score: s.score,
      start: loc ? loc.start : -1,
      end: loc ? loc.end : -1,
      located: !!loc,
    });
  }

  const startPeak = startProbs[argStart];
  const endPeak = endProbs[argEnd];

  // Per-token strip for "See inside" — strip the SentencePiece ▁ marker for display.
  const tokens = ids.slice(ctxStart, ctxEnd + 1).map((tid) =>
    tok.decode([tid], { skip_special_tokens: false })
  );

  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    question,
    context,
    answers,
    tokens,
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
