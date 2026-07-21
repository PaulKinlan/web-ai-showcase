// Protein language model worker — runs ESM-2 (a masked language model for PROTEINS) off the main thread.
// Instead of words, its "language" is amino acids: given a protein sequence, it predicts which residue
// belongs at a position, having learned the grammar of real proteins from millions of sequences. This is
// the family behind protein structure/variant-effect models (ESMFold, ESM-1v).
//
// Model: Xenova/esm2_t6_8M_UR50D — the tiny (8M-param) ESM-2 from Meta, task fill-mask, WASM, q8 (~8 MB).
// The weights are MIT (base facebook/esm2_t6_8M_UR50D). DISTINCT from the built natural-language fill-mask
// demos (BERT etc.): this is a different MODALITY — biology, not text.
//
// WHY THE LOW-LEVEL PATH (not the fill-mask pipeline): the ESM tokenizer here doesn't expose mask_token
// where transformers.js's FillMaskPipeline looks, so the pipeline throws "Mask token (null) not found". We
// run AutoModelForMaskedLM directly, place the mask ourselves, and softmax the logits at that position.
//
// Correctness proven FIRST in headless Chrome (transformers.js 3.7.5, WASM, q8): on ubiquitin the model
// recovers conserved residues under masking (the 100%-conserved C-terminal di-glycine -> G; a conserved
// lysine -> K) and, elsewhere, predicts chemically similar amino acids (e.g. I for V, both aliphatic).
// Nothing leaves the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/esm2_t6_8M_UR50D";
const AMINO = "ACDEFGHIKLMNPQRSTVWY"; // the 20 standard amino acids

let tokenizer = null;
let model = null;
let maskId = null;

function post(msg) {
  self.postMessage(msg);
}
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

async function ensureLoaded() {
  if (model) return;
  const mod = await import(TRANSFORMERS_URL);
  const { AutoTokenizer, AutoModelForMaskedLM, env } = mod;
  env.allowLocalModels = false;
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForMaskedLM.from_pretrained(MODEL_ID, {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  maskId = tokenizer.mask_token_id ?? tokenizer.model?.tokens_to_ids?.get("<mask>") ?? 32;
  post({ type: "ready", device: "wasm" });
}

// Run the model with residue `pos` masked → { probs: Float over vocab, ids } at the mask position.
async function maskedLogits(seq, pos) {
  const chars = seq.split("");
  chars[pos] = "<mask>";
  const enc = await tokenizer(chars.join(""));
  const ids = Array.from(enc.input_ids.data, Number);
  const mpos = ids.indexOf(maskId);
  const { logits } = await model(enc);
  const V = logits.dims[2];
  const row = Array.from(logits.data.slice(mpos * V, (mpos + 1) * V));
  return { probs: softmax(row), V };
}

// Predict the amino acid at a masked position → top-k [{aa, prob}] over the 20 amino acids.
async function predict(id, seq, pos, topK) {
  await ensureLoaded();
  const t0 = performance.now();
  const { probs } = await maskedLogits(seq, pos);
  const scored = AMINO.split("").map((aa) => ({
    aa,
    prob: probs[tokenizer.model.tokens_to_ids.get(aa)] || 0,
  }));
  scored.sort((a, b) => b.prob - a.prob);
  post({
    type: "predict",
    id,
    pos,
    truth: seq[pos],
    top: scored.slice(0, topK || 6),
    ms: Math.round(performance.now() - t0),
  });
}

// Conservation scan: mask each position in turn → probability the model assigns to the TRUE residue.
// High = the model is confident this residue fits (conserved/constrained); low = a flexible position.
async function scan(id, seq) {
  await ensureLoaded();
  const t0 = performance.now();
  const conf = new Array(seq.length);
  for (let i = 0; i < seq.length; i++) {
    const { probs } = await maskedLogits(seq, i);
    conf[i] = probs[tokenizer.model.tokens_to_ids.get(seq[i])] || 0;
    if (i % 4 === 0 || i === seq.length - 1) {
      post({ type: "progress", p: { status: "progress", progress: ((i + 1) / seq.length) * 100 } });
    }
  }
  post({ type: "scan", id, conf, ms: Math.round(performance.now() - t0) });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "predict") await predict(e.data.id, e.data.seq, e.data.pos, e.data.topK);
    else if (type === "scan") await scan(e.data.id, e.data.seq);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
