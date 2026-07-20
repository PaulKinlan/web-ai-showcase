// MGP-STR scene-text RECOGNITION worker — all inference off the main thread so the control UI stays
// responsive.
//
// MGP-STR (Multi-Granularity Prediction for Scene Text Recognition) is NOT an autoregressive line
// reader like TrOCR. It's a single-pass ViT that reads ONE cropped word and predicts it THREE ways in
// parallel — character-level, BPE subword-level, and WordPiece-level — then FUSES the three into one
// answer. Divergence between the three heads is the interesting signal: when a word is clean all three
// agree; when it's stylised/occluded they disagree and the fusion arbitrates. We surface all three
// plus the per-character char-head trace as the "see inside" surface.
//
// Model: onnx-community/mgp-str-base (task: image-to-text), WASM backend, q8. ~155 MB.
// API: MgpstrForSceneTextRecognition + MgpstrProcessor. model(inputs) → { logits: [char, bpe, wp] };
// processor.batch_decode(logits) → { generated_text, scores, char_preds, bpe_preds, wp_preds }.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/mgp-str-base";

// MGP-STR character charset (38): [GO] start, [s] end, then '0'–'9' and 'a'–'z'. Verified against the
// model's own char tokenizer (decode of indices 0..37). We read the char head directly for the
// per-character confidence trace, so we map indices with this table.
const CHARSET = ["[GO]", "[s]"];
for (let d = 0; d <= 9; d++) CHARSET.push(String(d));
for (let c = 0; c < 26; c++) CHARSET.push(String.fromCharCode(97 + c));
const GO_IDX = 0, EOS_IDX = 1;

let model = null;
let processor = null;
let RawImage = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const T = await import(TRANSFORMERS_URL);
  T.env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the SW.
  RawImage = T.RawImage;
  model = await T.MgpstrForSceneTextRecognition.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await T.MgpstrProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

function softmaxRow(row) {
  let max = -Infinity;
  for (let i = 0; i < row.length; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  const e = new Float64Array(row.length);
  for (let i = 0; i < row.length; i++) {
    e[i] = Math.exp(row[i] - max);
    sum += e[i];
  }
  for (let i = 0; i < row.length; i++) e[i] /= sum;
  return e;
}

// Character-head trace: for each position after the [GO] slot, the argmax character + its softmax
// probability, stopping at the [s] end token. This is the readable, per-glyph view of what the char
// head actually produced — the heart of "see inside".
function charTrace(charTensor) {
  const [, pos, vocab] = charTensor.dims;
  const data = charTensor.data;
  const steps = [];
  for (let p = 1; p < pos; p++) {
    const probs = softmaxRow(Array.from(data.subarray(p * vocab, p * vocab + vocab)));
    let mi = 0, mx = -Infinity;
    for (let i = 0; i < vocab; i++) {
      if (probs[i] > mx) {
        mx = probs[i];
        mi = i;
      }
    }
    if (mi === EOS_IDX) break;
    if (mi === GO_IDX) continue;
    steps.push({ ch: CHARSET[mi] ?? "?", p: +mx.toFixed(4) });
  }
  return steps;
}

// Per-head sequence confidence = geometric mean of the top-1 softmax probability over the first
// `nTokens` real positions (after the [GO] slot). A comparable 0..1 number per granularity so the page
// can show WHY the fusion trusted one head over another.
function seqConfidence(tensor, nTokens) {
  const [, pos, vocab] = tensor.dims;
  const data = tensor.data;
  const n = Math.max(1, Math.min(nTokens, pos - 1));
  let logSum = 0;
  for (let p = 1; p <= n; p++) {
    const probs = softmaxRow(Array.from(data.subarray(p * vocab, p * vocab + vocab)));
    let mx = 0;
    for (let i = 0; i < vocab; i++) if (probs[i] > mx) mx = probs[i];
    logSum += Math.log(Math.max(mx, 1e-9));
  }
  return +Math.exp(logSum / n).toFixed(4);
}

async function recognise(id, imageURL) {
  await ensureLoaded();
  const t0 = performance.now();
  const img = await RawImage.read(imageURL);
  const inputs = await processor(img);
  const outputs = await model(inputs);
  const logits = outputs.logits; // [char_logits, bpe_logits, wp_logits]
  const decoded = processor.batch_decode(logits);
  const text = (decoded.generated_text?.[0] ?? "").trim();
  const charPred = (decoded.char_preds?.[0] ?? "").trim();
  const bpePred = (decoded.bpe_preds?.[0] ?? "").trim();
  const wpPred = (decoded.wp_preds?.[0] ?? "").trim();
  const steps = charTrace(logits[0]);
  const result = {
    type: "result",
    id,
    text,
    fusedScore: +(decoded.scores?.[0] ?? 0).toFixed(4),
    granularities: [
      { name: "char", pred: charPred, conf: seqConfidence(logits[0], charPred.length) },
      { name: "bpe", pred: bpePred, conf: seqConfidence(logits[1], bpePred.length) },
      { name: "wp", pred: wpPred, conf: seqConfidence(logits[2], wpPred.length) },
    ],
    steps,
    ms: Math.round(performance.now() - t0),
    device,
  };
  post(result);
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await recognise(e.data.id, e.data.image);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
