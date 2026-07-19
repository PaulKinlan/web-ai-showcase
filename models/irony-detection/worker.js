// Irony / sarcasm worker — binary irony classification off the main thread so the "as you type" UI
// stays smooth. Model: Xenova/twitter-roberta-base-irony (task: text-classification), WASM, q8.
// Weights are cardiffnlp/twitter-roberta-base-irony (TweetEval irony task) exported to ONNX by Xenova.
//
// The head is single-label softmax over 2 classes: "irony" vs "non_irony". A high irony score means
// the model thinks the literal meaning and the intended meaning diverge — the hallmark of sarcasm.
// This is genuinely hard: irony lives in CONTEXT and tone, not the words, so the model reads cues
// (over-the-top praise, incongruous framing, "oh great", "just what I needed") learned from tweets.
//
// Operations:
//   run   → score one text, return irony + non_irony probabilities (a real 2-class softmax).
//   batch → score many texts in one padded forward pass (for the sarcasm pre-check queue).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "text-classification",
    model: "Xenova/twitter-roberta-base-irony",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Recover the irony probability from a top_k:2 result (order can vary; softmax so the pair sums to 1).
function ironyProb(rows) {
  const arr = Array.isArray(rows) ? rows : [rows];
  const ir = arr.find((r) => r.label === "irony");
  if (ir) return ir.score;
  const non = arr.find((r) => r.label === "non_irony");
  return non ? 1 - non.score : 0.5;
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 2 });
  const irony = ironyProb(out);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    irony,
    nonIrony: 1 - irony,
    label: irony >= 0.5 ? "irony" : "non_irony",
    scores: out,
    ms,
    device,
  });
}

async function classifyBatch(id, texts) {
  await ensureLoaded();
  if (!texts.length) {
    post({ type: "batch", id, texts: [], results: [], ms: 0, device });
    return;
  }
  const t0 = performance.now();
  const out = await pipe(texts, { top_k: 2 });
  const results = texts.map((_, i) => {
    const irony = ironyProb(out[i]);
    return { irony, nonIrony: 1 - irony, label: irony >= 0.5 ? "irony" : "non_irony" };
  });
  const ms = Math.round(performance.now() - t0);
  post({ type: "batch", id, texts, results, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "batch") await classifyBatch(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
