// News topic classification worker — inference off the main thread so the tagging board stays smooth.
// Model: tarekziade/topic_classification (task: text-classification), WASM, q8. A dedicated
// 10-class topic head (xtremedistil-L6-H256 fine-tuned). Its config is
// problem_type: "multi_label_classification", so ONE forward pass returns an INDEPENDENT sigmoid score
// per topic (0..1 each; they do NOT sum to 1) — a headline can legitimately belong to several desks.
// This is deliberately DISTINCT from a zero-shot classifier, which instead runs one NLI pass per
// candidate label you supply at query time.
//
// Operations:
//   run  → classify one text, return the winning topic + the FULL 10-class probability distribution.
//   batch → classify many texts in a single pass (for the newsroom auto-tagging feed).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "tarekziade/topic_classification";
const NUM_LABELS = 10;

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "text-classification",
    model: MODEL_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Return the full sorted list of independent per-topic sigmoid scores. top_k >= NUM_LABELS makes the
// pipeline emit every topic, sorted by score, so the "see inside" surface shows the real multi-label
// output — not just the top-1.
//
// Because the scores are independent (they don't form one probability distribution), we quantify
// "how torn is the model" in multi-label terms that stay honest:
//   • ambiguity  = runner-up score ÷ top score (0 = one clear desk, →1 = a close second desk).
//   • secondary  = how many desks clear the 0.5 mark (a story genuinely spanning several topics).
async function classifyOne(text) {
  const out = await pipe(text, { top_k: NUM_LABELS });
  const dist = out.map((r) => ({ label: r.label, score: r.score }));
  const top = dist[0];
  const runnerUp = dist[1] || null;
  return {
    label: top.label,
    score: top.score,
    runnerUp,
    dist,
    ambiguity: runnerUp && top.score > 0 ? Math.min(1, runnerUp.score / top.score) : 0,
    secondary: dist.filter((d) => d.score >= 0.5).length,
    margin: top.score - (runnerUp?.score ?? 0),
  };
}

async function classify(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const r = await classifyOne(text);
  post({ type: "result", id, text, ...r, ms: Math.round(performance.now() - t0), device });
}

async function batch(id, texts) {
  await ensureLoaded();
  const t0 = performance.now();
  // One pipeline call over the whole array — a single batched pass through the model.
  const outs = await pipe(texts, { top_k: NUM_LABELS });
  // With an array input the pipeline returns an array of per-text distributions.
  const rows = texts.map((text, i) => {
    const dist = (outs[i] || []).map((r) => ({ label: r.label, score: r.score }));
    return { text, label: dist[0]?.label ?? "?", score: dist[0]?.score ?? 0, dist };
  });
  post({ type: "batch", id, rows, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await classify(e.data.id, e.data.text);
    else if (type === "batch") await batch(e.data.id, e.data.texts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
