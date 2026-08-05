// MobileBERT-NLI worker — natural-language inference off the main thread, WASM int8.
// Model: Xenova/mobilebert-uncased-mnli (task: text-classification, MNLI labels).
//
// MobileBERT is Google's mobile-optimized BERT: a bottleneck architecture with grouped attention
// that keeps ~95% of BERT-base quality at a fraction of the size (~25 MB int8 here). This is the
// MNLI fine-tune: given a premise and a hypothesis, it scores entailment / neutral / contradiction.
// We use the text-classification pipeline and return the real softmax probabilities — nothing canned.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/mobilebert-uncased-mnli";
const DTYPE = "int8";
let classifier = null;
let mod = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (classifier) return;
  mod = await import(TRANSFORMERS_URL);
  const { pipeline } = mod;
  classifier = await pipeline("text-classification", "Xenova/mobilebert-uncased-mnli", {
    device,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

async function classify(id, premise, hypothesis) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await classifier(`${premise} ${hypothesis}`, { top_k: 3 });
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    scores: out.map((r) => ({ label: r.label, score: r.score })),
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "classify") await classify(id, e.data.premise, e.data.hypothesis);
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
