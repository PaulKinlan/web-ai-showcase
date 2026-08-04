// Protein mutation oracle worker — a PROTEIN language model off the main thread.
// Model: Xenova/esm2_t12_35M_UR50D (ESM-2, 35M) — Meta's protein masked language model, WASM, q8 (~35 MB).
// ESM-2 is trained on ~60M protein sequences (UniRef) to predict masked amino-acid RESIDUES; the model
// learns the "grammar" of proteins. This is a DISTINCT capability with NO built counterpart: every built
// fill-mask demo is a NATURAL-LANGUAGE model (English/multilingual words) — this reads BIOLOGICAL SEQUENCES
// (20 amino acids) and does zero-shot VARIANT-EFFECT prediction (Meier et al. 2021): mask a position, and
// the log-likelihood of each substitution vs the wild-type residue estimates whether a mutation is
// tolerated or damaging. The fill-mask PIPELINE is unusable here (the conversion omits mask_token), so we
// run AutoModelForMaskedLM + AutoTokenizer directly and read the logits. ESM-2 upstream is MIT
// (facebook/esm2_t12_35M_UR50D). Nothing leaves the tab.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const REPO = "Xenova/esm2_t12_35M_UR50D";
const AAS = "ACDEFGHIKLMNPQRSTVWY".split(""); // the 20 standard amino acids

let T = null;
let tokenizer = null;
let model = null;
let aaIds = null;
let maskId = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  T = await import(TRANSFORMERS_URL);
  tokenizer = await T.AutoTokenizer.from_pretrained(REPO, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await T.AutoModelForMaskedLM.from_pretrained(REPO, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  // token id for each amino acid + the mask, via single-token encoding: encode("X") → [cls, X, eos]
  const idOf = (s) => Number(tokenizer.encode(s)[1]);
  aaIds = Object.fromEntries(AAS.map((a) => [a, idOf(a)]));
  maskId = idOf("<mask>");
  post({ type: "ready", device: "wasm" });
}

// Mask ONE position and return the model's probability for every amino acid there (softmax over the 20
// standard AAs), plus the wild-type residue at that position. From this the page derives, for each
// substitution, the log-likelihood ratio ln(P_mut) − ln(P_wt): < 0 ⇒ the model finds the mutation less
// plausible than wild-type (a zero-shot "likely deleterious" signal); > 0 ⇒ at least as plausible.
async function scan(id, seq, pos) {
  await ensureLoaded();
  const t0 = performance.now();
  const wtAA = seq[pos];
  const enc = await tokenizer(seq);
  const ids = Array.from(enc.input_ids.data).map(Number);
  const tokenPos = pos + 1; // +1 for the leading <cls>
  ids[tokenPos] = maskId;
  const input_ids = new T.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  const attention_mask = new T.Tensor("int64", BigInt64Array.from(ids.map(() => 1n)), [1, ids.length]);
  const out = await model({ input_ids, attention_mask });
  const logits = out.logits; // [1, L, V]
  const V = logits.dims[2];
  const base = tokenPos * V;
  const aaLogits = AAS.map((a) => Number(logits.data[base + aaIds[a]]));
  const mx = Math.max(...aaLogits);
  const exps = aaLogits.map((x) => Math.exp(x - mx));
  const Z = exps.reduce((s, x) => s + x, 0);
  const probs = {};
  AAS.forEach((a, i) => (probs[a] = exps[i] / Z));
  post({
    type: "result",
    id,
    pos,
    wtAA,
    probs, // { A: p, C: p, … } over the 20 AAs at this position
    ms: Math.round(performance.now() - t0),
    device: "wasm",
  });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "scan") await scan(d.id, d.seq, d.pos);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
