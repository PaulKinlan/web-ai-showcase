// Multi-model worker for the news-topic "route → summarize" composition. It loads TWO real models and
// chains them off the main thread:
//   1. tarekziade/topic_classification (text-classification, q8) → route the article to its desk.
//   2. Xenova/distilbart-cnn-6-6 (summarization, q8)             → write a short desk-ready TL;DR.
// Both run locally via Transformers.js; nothing leaves the device.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const TOPIC_ID = "tarekziade/topic_classification";
const SUM_ID = "Xenova/distilbart-cnn-6-6";
const NUM_LABELS = 10;

let topicPipe = null;
let sumPipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (topicPipe && sumPipe) return;
  // Load the small router first (fast feedback), then the heavier summariser.
  if (!topicPipe) {
    const t = await loadPipeline({
      task: "text-classification",
      model: TOPIC_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", stage: "topic", p }),
    });
    topicPipe = t.pipe;
    device = t.device;
  }
  if (!sumPipe) {
    const s = await loadPipeline({
      task: "summarization",
      model: SUM_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", stage: "summary", p }),
    });
    sumPipe = s.pipe;
  }
  post({ type: "ready", device });
}

async function route(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  // Stage 1 — classify (full distribution so the UI can show the routing confidence).
  const cls = await topicPipe(text, { top_k: NUM_LABELS });
  const dist = cls.map((r) => ({ label: r.label, score: r.score }));
  const topicMs = Math.round(performance.now() - t0);
  post({ type: "routed", id, label: dist[0].label, score: dist[0].score, dist, topicMs, device });
  // Stage 2 — summarise.
  const t1 = performance.now();
  const out = await sumPipe(text, {
    max_new_tokens: 80,
    min_new_tokens: 15,
    no_repeat_ngram_size: 3,
  });
  const summary = (Array.isArray(out) ? out[0] : out)?.summary_text?.trim() ?? "";
  post({ type: "summary", id, summary, sumMs: Math.round(performance.now() - t1), device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "route") await route(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
