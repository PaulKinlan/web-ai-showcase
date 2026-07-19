// ModernBERT fill-mask worker — inference off the main thread so the control UI stays responsive.
// Model: answerdotai/ModernBERT-base (task: fill-mask / masked language modelling), WASM, q8.
//
// ModernBERT (Answer.AI + LightOn, Dec 2024) is a genuinely NEW encoder architecture, not a BERT
// fine-tune: rotary position embeddings (RoPE), GeGLU feed-forwards, alternating global/local
// attention (every 3rd layer is global; the rest use a 128-token sliding window), and an 8192-token
// context — 16× BERT's 512. It is loaded here through the SAME masked-LM head BERT uses, so the
// interaction is familiar; only the engine underneath is modern.
//
// As with the BERT demo we load the tokenizer + model MANUALLY (AutoModelForMaskedLM) instead of the
// fill-mask pipeline, because the "see inside" surface needs the EXACT masked-token logits — the raw
// pre-softmax scores the model produced — not just the top-k a pipeline hands back. From the logits
// row at each [MASK] position we softmax ourselves, so the page shows both the real logit and the
// probability for every candidate.
//
// One tokenizer difference from BERT: ModernBERT uses a byte-level BPE tokenizer (like OLMo/GPT), so a
// word-initial token carries a leading space (Ġ). We trim it for display and prepend a space when
// scoring a bare candidate word, so "nurse" is scored as the word-initial token, not a sub-word.
//
// Operations:
//   fill            → for each [MASK] in the text, the top-k predictions (token, logit, probability).
//   fillMany        → the same over a batch of texts (bias probes, multi-model candidate generation).
//   scoreCandidates → for a single-mask text, the logit + probability of a fixed candidate set.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "answerdotai/ModernBERT-base";
let tokenizer = null;
let model = null;
const device = "wasm";

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

// Byte-level BPE tokens carry a leading space on word starts; show a clean word.
function cleanToken(idx) {
  return tokenizer.decode([idx]).replace(/^\s+/, "");
}

// Softmax over one logits row, returning the top-k indices with logit + probability. Partial top-k
// without sorting the whole 50k vocab (a small ordered buffer), plus the row max/sum for the caller.
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
    token: cleanToken(idx),
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
  return { text, masks, maskCount: positions.length, tokenCount: ids.length };
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

// Word-initial BPE token id for a candidate word (prepend a space so it isn't scored as a sub-word).
// Multi-token candidates are compared by their leading token — a disclosed simplification.
function firstTokenId(word) {
  const enc = tokenizer.encode(" " + String(word).trim(), null, { add_special_tokens: false });
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
