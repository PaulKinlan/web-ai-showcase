// Multilingual mBERT fill-mask worker — the "generalist" side of the Portuguese-native vs multilingual
// contrast. Model: Xenova/bert-base-multilingual-cased (task: fill-mask), WASM, q8. Loaded ON DEMAND.
//
// mBERT spread a 119,547-token WordPiece vocabulary across ~104 languages; BERTimbau spent its whole
// 29,794-token budget on Portuguese. Running the SAME Portuguese [MASK] sentence through both shows what
// a language specialist buys you. Same "[MASK]" WordPiece masking style → an apples-to-apples
// comparison (unlike XLM-R, which would need "<mask>").
//
// We load the tokenizer + model manually (AutoModelForMaskedLM) to read the exact masked-token logits.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/bert-base-multilingual-cased";
let tokenizer = null;
let model = null;
const device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  const { AutoTokenizer, AutoModelForMaskedLM, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
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

function cleanToken(idx) {
  return tokenizer.decode([idx]).replace(/^##/, "").replace(/\s+/g, " ").trim();
}

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

async function runOne(text, topk) {
  const inputs = await tokenizer(text);
  const ids = Array.from(inputs.input_ids.data, Number);
  const positions = maskPositions(ids);
  const { logits } = await model(inputs);
  const vocab = logits.dims[2];
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

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "fill") await fill(e.data.id, e.data.text, e.data.topk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
