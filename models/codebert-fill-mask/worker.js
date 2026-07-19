// CodeBERTa code fill-mask worker — inference off the main thread so the control UI stays responsive.
// Model: onnx-community/CodeBERTa-small-v1-ONNX (a RoBERTa MLM from Hugging Face's CodeBERT family,
// pre-trained on source CODE — CodeSearchNet: go, java, javascript, php, python, ruby). Task: fill-mask.
// WASM, q8.
//
// We load the tokenizer + model MANUALLY (AutoModelForMaskedLM) rather than the fill-mask pipeline, so
// the "see inside" surface can read the EXACT masked-token logits — the raw pre-softmax scores the model
// produced — over the whole 52,000-token byte-level-BPE vocabulary, and softmax them ourselves. That is
// how we show both the real logit and the probability for every predicted code token.
//
// The mask marker in the RoBERTa tokenizer is <mask> (unlike BERT's [MASK]); the tokenizer uses
// byte-level BPE, so a leading space is encoded as the character Ġ — we surface that in the "see inside"
// tokenization view because it is exactly why code tokenizes differently from prose.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/CodeBERTa-small-v1-ONNX";
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
  console.log(`[codebert worker] loading ${MODEL_ID} on wasm (q8) — code masked-LM`);
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForMaskedLM.from_pretrained(MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[codebert worker] ready on wasm");
  post({ type: "ready", device });
}

// Softmax over one logits row, returning the top-k indices with logit + probability. Partial top-k
// without sorting the whole 52k vocab (a small ordered buffer), like the BERT demo.
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
    // Byte-level BPE: a leading Ġ means "preceded by a space". Show it as a visible ␣ so code tokens
    // like " len" read correctly. decode() with a single id keeps the raw sub-word form.
    token: tokenizer.decode([idx]).replace(/^Ġ/, " ").replace(/Ġ/g, " "),
    rawToken: tokenizer.decode([idx]),
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
  return { text, masks, maskCount: positions.length, vocab };
}

/** Tokenize a snippet for the "see inside" view: expose byte-level-BPE tokens with the Ġ space marker. */
async function tokenizeView(text) {
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const tokens = ids.map((id) => {
    const raw = tokenizer.decode([id]);
    const special = /^<.*>$/.test(raw);
    return { id, raw, special, space: /^Ġ/.test(raw), text: raw.replace(/Ġ/g, "␣") };
  });
  return { tokens, count: ids.length };
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

async function tokenize(id, text) {
  await ensureLoaded();
  const r = await tokenizeView(text);
  post({ type: "tokens", id, ...r, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "fill") await fill(e.data.id, e.data.text, e.data.topk);
    else if (type === "fillMany") await fillMany(e.data.id, e.data.texts, e.data.topk);
    else if (type === "tokenize") await tokenize(e.data.id, e.data.text);
  } catch (err) {
    console.error("[codebert worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
