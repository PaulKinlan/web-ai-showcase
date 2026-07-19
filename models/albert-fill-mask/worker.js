// ALBERT fill-mask worker — masked-language-model inference off the main thread.
// Model: Xenova/albert-base-v2 (task: fill-mask), WASM, q8. ALBERT is a "lite" BERT: it reuses ONE
// transformer block across all 12 layers (cross-layer parameter sharing) and factorises its embedding
// (vocab → 128 → 768) — ~11.8M params vs BERT-base's ~110M for the same depth.
//
// We load the tokenizer + model MANUALLY (AutoModelForMaskedLM), not the fill-mask pipeline, because
// the "see inside" surface needs the EXACT masked-token logits — the raw pre-softmax scores the model
// produced — not just the rounded top-k a pipeline returns. From the logits row at each [MASK] we
// softmax ourselves, so the page can show both the real logit and the probability for every candidate.
//
// Operations:
//   fill            → for each [MASK] in the text, the top-k predictions (token, logit, probability).
//   fillMany        → the same over a batch of texts (plausibility ladders, bias probes, chaining).
//   scoreCandidates → for a single-mask text, the logit + probability of a fixed candidate set.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/albert-base-v2";
let tokenizer = null;
let model = null;
const device = "wasm"; // ALBERT-base is tiny; WASM everywhere, no WebGPU needed.

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

// Softmax over one logits row; return the top-k indices with logit + probability. Partial top-k
// (no full 30k sort) via a small ordered buffer — the same discipline as the BERT worker.
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
    token: cleanToken(tokenizer.decode([idx])),
    logit: row[idx],
    prob: Math.exp(row[idx] - max) / sum,
  }));
}

// SentencePiece decode can surface the ▁ word-boundary marker / leading space — tidy for display.
function cleanToken(s) {
  return String(s).replace(/▁/g, " ").trim();
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
  return { text, masks, maskCount: positions.length };
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

// Leading-subword id for a candidate word (SentencePiece). Multi-token candidates are compared by
// their first token — a deliberate, disclosed simplification for the scoring surfaces.
function firstTokenId(word) {
  // Transformers.js v3 signature is encode(text, { add_special_tokens, … }) — the options object is
  // the SECOND positional arg (passing null there crashes reading text_pair).
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
