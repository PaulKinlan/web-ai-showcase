// Clickbait / manipulative-headline worker — binary headline classification off the main thread so the
// "as you type" UI stays smooth. Model: CoolpantsMcBadass/headline-classifier (task: text-classification),
// a DistilBERT sequence classifier exported to ONNX (q8), WASM.
//
// The head is single-label softmax over 2 classes: "manipulative" vs "legitimate". A high manipulative
// score means the headline leans on curiosity-gap / withheld-payoff / emotional-bait patterns
// ("you won't believe…", "this one weird trick…", "N shocking secrets…") rather than stating the news.
// This is a SIGNAL, not a verdict — a legit outlet can write a punchy headline, and a scam can write a
// dry one. Treat the score as a triage cue for media-literacy, not ground truth.
//
// Operations:
//   run   → score one headline, return manipulative + legitimate probabilities (a real 2-class softmax).
//   batch → score many headlines in one padded forward pass (for the feed-hygiene queue).

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
    model: "CoolpantsMcBadass/headline-classifier",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Recover the manipulative ("clickbait") probability from a top_k:2 result (order can vary; softmax so
// the pair sums to 1).
function clickbaitProb(rows) {
  const arr = Array.isArray(rows) ? rows : [rows];
  const cb = arr.find((r) => r.label === "manipulative");
  if (cb) return cb.score;
  const legit = arr.find((r) => r.label === "legitimate");
  return legit ? 1 - legit.score : 0.5;
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { top_k: 2 });
  const clickbait = clickbaitProb(out);
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text,
    clickbait,
    legit: 1 - clickbait,
    label: clickbait >= 0.5 ? "manipulative" : "legitimate",
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
    const clickbait = clickbaitProb(out[i]);
    return {
      clickbait,
      legit: 1 - clickbait,
      label: clickbait >= 0.5 ? "manipulative" : "legitimate",
    };
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
