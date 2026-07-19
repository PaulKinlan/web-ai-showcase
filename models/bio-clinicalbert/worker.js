// Bio_ClinicalBERT fill-mask worker — inference off the main thread so the control UI stays responsive.
// Primary model: onnx-community/Bio_ClinicalBERT-ONNX (task: fill-mask / masked language modelling),
// WASM, q8. Bio_ClinicalBERT (emilyalsentzer) is BERT continued-pretrained on ~2M MIMIC-III clinical
// notes — a DOMAIN-SPECIALISED masked LM. It shares bert-base-cased's tokenizer/vocab (28,996
// WordPiece), which is exactly why we can run it head-to-head against GENERAL bert-base-cased on the
// SAME masked sentence: any difference is the domain, not the tokenizer.
//
// We load the tokenizer + model MANUALLY (AutoModelForMaskedLM) rather than the fill-mask pipeline,
// because the "see inside" surface needs the EXACT masked-token logits — the raw pre-softmax scores
// the model actually produced — not just the top-k probabilities a pipeline hands back.
//
// Two model handles live here so the domain-gap comparison runs both in one place:
//   • clinical  → onnx-community/Bio_ClinicalBERT-ONNX   (auto-loaded, primary)
//   • general   → onnx-community/bert-base-cased-ONNX     (loaded on first Compare — same vocab, cased)
//
// Operations:
//   fill(which)     → for each [MASK] in the text, the top-k predictions (token, logit, probability).
//   compare         → run BOTH models on the same single-mask text; return each one's top-k so the page
//                     can show the domain gap side by side (this is the story).
//   scoreCandidates → for a single-mask text, the logit + probability of a fixed candidate set on the
//                     clinical model (e.g. compare P("respiratory") vs P("renal") at the mask).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODELS = {
  clinical: "onnx-community/Bio_ClinicalBERT-ONNX",
  general: "onnx-community/bert-base-cased-ONNX",
};

const loaded = {}; // which → { tokenizer, model }
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded(which = "clinical") {
  if (loaded[which]) return loaded[which];
  const { AutoTokenizer, AutoModelForMaskedLM, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the SW.
  const id = MODELS[which];
  const tokenizer = await AutoTokenizer.from_pretrained(id, {
    progress_callback: (p) => post({ type: "progress", which, p }),
  });
  const model = await AutoModelForMaskedLM.from_pretrained(id, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", which, p }),
  });
  loaded[which] = { tokenizer, model };
  post({ type: "ready", which, device });
  return loaded[which];
}

// Softmax over one logits row, returning the top-k indices with their logit + probability.
function topkFromRow(row, vocab, k, tokenizer) {
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
    token: tokenizer.decode([idx]),
    logit: row[idx],
    prob: Math.exp(row[idx] - max) / sum,
  }));
}

function maskPositions(ids, tokenizer) {
  const maskId = tokenizer.mask_token_id;
  const out = [];
  for (let i = 0; i < ids.length; i++) if (ids[i] === maskId) out.push(i);
  return out;
}

async function runOne(text, topk, which) {
  const { tokenizer, model } = await ensureLoaded(which);
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const positions = maskPositions(ids, tokenizer);
  const { logits } = await model(inputs);
  const dims = logits.dims; // [1, seq, vocab]
  const vocab = dims[2];
  const data = logits.data;
  const masks = positions.map((pos) => {
    const row = data.subarray(pos * vocab, pos * vocab + vocab);
    return { pos, predictions: topkFromRow(row, vocab, topk, tokenizer) };
  });
  return { text, masks, maskCount: positions.length, vocab };
}

async function fill(id, text, topk, which) {
  const t0 = performance.now();
  const r = await runOne(text, topk || 8, which || "clinical");
  post({
    type: "fill",
    id,
    which: which || "clinical",
    ...r,
    ms: Math.round(performance.now() - t0),
    device,
  });
}

async function compare(id, text, topk) {
  const t0 = performance.now();
  const clinical = await runOne(text, topk || 8, "clinical");
  const general = await runOne(text, topk || 8, "general");
  post({ type: "compare", id, clinical, general, ms: Math.round(performance.now() - t0), device });
}

function firstTokenId(word, tokenizer) {
  const enc = tokenizer.encode(word, null, { add_special_tokens: false });
  return enc && enc.length ? enc[0] : null;
}

async function scoreCandidates(id, text, candidates) {
  const { tokenizer, model } = await ensureLoaded("clinical");
  const t0 = performance.now();
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const positions = maskPositions(ids, tokenizer);
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
    const tid = firstTokenId(word, tokenizer);
    if (tid == null) return { word, logit: null, prob: 0 };
    return { word, tokenId: tid, logit: row[tid], prob: Math.exp(row[tid] - max) / sum };
  });
  post({ type: "scores", id, text, scores, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.which || "clinical");
    else if (type === "fill") await fill(e.data.id, e.data.text, e.data.topk, e.data.which);
    else if (type === "compare") await compare(e.data.id, e.data.text, e.data.topk);
    else if (type === "scoreCandidates") {
      await scoreCandidates(e.data.id, e.data.text, e.data.candidates);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
