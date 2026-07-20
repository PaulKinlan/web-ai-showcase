// Chinese BERT fill-mask worker — inference off the main thread so the control UI stays responsive.
// Model: Xenova/bert-base-chinese (task: fill-mask / masked language modelling), WASM, q8.
//
// This is Google's BERT pre-trained on Chinese Wikipedia. Its tokenizer is CHARACTER-LEVEL: unlike the
// SentencePiece models, it needs no special word-segmentation step — every Chinese character (Hànzì) is
// its own token in a 21,128-token vocabulary, so filling a [MASK] predicts a single character. Its mask
// token is "[MASK]" (BERT WordPiece style), same as English BERT.
//
// Like the other fill-mask pages, we load the tokenizer + model MANUALLY (AutoModelForMaskedLM) rather
// than the fill-mask pipeline, because the "see inside" surface needs the EXACT masked-token logits —
// the raw pre-softmax scores over the whole 21,128-character vocabulary — not just the top-k a pipeline
// hands back. From the logits row at each [MASK] we softmax ourselves.
//
// Operations:
//   fill            → for each [MASK] in the text, the top-k predictions (character, logit, probability).
//   fillMany        → the same over a batch of texts (chengyu probes, input-assist variants).
//   scoreCandidates → for a single-mask text, the logit + probability of a fixed candidate set
//                     (compare P(char_A) vs P(char_B) at the mask across Chinese templates).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/bert-base-chinese";
let tokenizer = null;
let model = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForMaskedLM, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the SW.
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForMaskedLM.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

// WordPiece decodes continuation pieces with a leading "##". A Chinese char is normally a whole token,
// but strip any "##" so the UI shows the clean character.
function cleanToken(idx) {
  return tokenizer.decode([idx]).replace(/^##/, "").replace(/\s+/g, "");
}

// Softmax over one logits row, returning the top-k indices with logit + probability. Partial top-k
// without sorting the whole 21k vocab: maintain a small ordered buffer (same approach as the BERT page).
function topkFromRow(row, vocab, k) {
  let max = -Infinity;
  for (let i = 0; i < vocab; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  for (let i = 0; i < vocab; i++) sum += Math.exp(row[i] - max);
  const best = [];
  for (let i = 0; i < vocab; i++) {
    const v = row[i];
    if (best.length < k) {
      best.push(i);
      if (best.length === k) best.sort((a, b) => row[a] - row[b]);
    } else if (v > row[best[0]]) {
      best[0] = i;
      let j = 0;
      while (j < k - 1 && row[best[j]] > row[best[j + 1]]) {
        const t = best[j];
        best[j] = best[j + 1];
        best[j + 1] = t;
        j++;
      }
    }
  }
  best.sort((a, b) => row[b] - row[a]);
  return best.map((idx) => ({
    tokenId: idx,
    token: cleanToken(idx) || tokenizer.decode([idx]),
    logit: row[idx],
    prob: Math.exp(row[idx] - max) / sum,
  }));
}

function maskPositions(ids) {
  const maskId = tokenizer.mask_token_id;
  const out = [];
  for (let i = 0; i < ids.length; i++) if (ids[i] === maskId) out.push(i);
  return out;
}

// Char-level view of the whole tokenisation: every input token id → its decoded character, so the
// "see inside" surface can show that Chinese BERT tokenises one character at a time.
function tokenStrip(ids) {
  const maskId = tokenizer.mask_token_id;
  return ids.map((id) => ({
    id,
    ch: id === maskId ? "[MASK]" : cleanToken(id),
    isMask: id === maskId,
    isSpecial: id === tokenizer.cls_token_id || id === tokenizer.sep_token_id,
  }));
}

async function runOne(text, topk) {
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const positions = maskPositions(ids);
  const { logits } = await model(inputs);
  const dims = logits.dims; // [1, seq, vocab]
  const vocab = dims[2];
  const data = logits.data;
  const masks = positions.map((pos) => {
    const row = data.subarray(pos * vocab, pos * vocab + vocab);
    return { pos, predictions: topkFromRow(row, vocab, topk) };
  });
  return { text, masks, maskCount: positions.length, tokens: tokenStrip(ids) };
}

async function fill(id, text, topk) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await runOne(text, topk || 8);
  post({ type: "fill", id, ...r, ms: Math.round(performance.now() - t0), device });
}

async function fillMany(id, texts, topk) {
  await ensureLoaded();
  const t0 = performance.now();
  const results = [];
  for (const text of texts) results.push(await runOne(text, topk || 8));
  post({ type: "fillMany", id, results, ms: Math.round(performance.now() - t0), device });
}

// Token id for a single-character candidate (Chinese BERT is char-level, so most candidates are one
// character). Multi-token candidates are compared by their leading token — a disclosed simplification.
function firstTokenId(word) {
  const enc = tokenizer.encode(word, { add_special_tokens: false });
  return enc && enc.length ? enc[0] : null;
}

async function scoreCandidates(id, text, candidates) {
  await ensureLoaded();
  const t0 = performance.now();
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const positions = maskPositions(ids);
  if (positions.length === 0) {
    post({ type: "scores", id, text, scores: [], ms: 0, device });
    return;
  }
  const { logits } = await model(inputs);
  const vocab = logits.dims[2];
  const pos = positions[0];
  const row = logits.data.subarray(pos * vocab, pos * vocab + vocab);
  let max = -Infinity;
  for (let i = 0; i < vocab; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  for (let i = 0; i < vocab; i++) sum += Math.exp(row[i] - max);
  const scores = candidates.map((word) => {
    const tid = firstTokenId(word);
    if (tid == null) return { word, logit: null, prob: 0 };
    return { word, tokenId: tid, logit: row[tid], prob: Math.exp(row[tid] - max) / sum };
  });
  post({ type: "scores", id, text, scores, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "fill") await fill(e.data.id, e.data.text, e.data.topk);
    else if (type === "fillMany") await fillMany(e.data.id, e.data.texts, e.data.topk);
    else if (type === "scoreCandidates") {
      await scoreCandidates(e.data.id, e.data.text, e.data.candidates);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
