// Granite multi-model worker — a real two-model RAG composition, off the main thread:
//   1. all-MiniLM-L6-v2 (Transformers.js feature-extraction) embeds the question + every note and picks
//      the most similar note by cosine similarity (retrieval).
//   2. Granite-4.0-350M answers grounded ONLY in that retrieved note (generation).
// Embedder: Xenova/all-MiniLM-L6-v2 (~25 MB, q8). LLM: onnx-community/granite-4.0-350m-ONNX-web (~576 MB, q4).
import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";
const EMB_ID = "Xenova/all-MiniLM-L6-v2";
const LLM_ID = "onnx-community/granite-4.0-350m-ONNX-web";
let mod = null, embedder = null, generator = null, device = "wasm";
function post(msg) {
  self.postMessage(msg);
}
async function ensureLoaded() {
  if (embedder && generator) return;
  mod = await import(TRANSFORMERS_URL);
  mod.env.allowLocalModels = false;
  embedder = await mod.pipeline("feature-extraction", EMB_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  generator = await mod.pipeline("text-generation", LLM_ID, {
    dtype: "q4",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}
function cosine(a, b) {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
async function embed(t) {
  const o = await embedder(t, { pooling: "mean", normalize: true });
  return Array.from(o.data);
}
async function run(id, query, notes) {
  await ensureLoaded();
  const t0 = performance.now();
  const qv = await embed(query);
  const scored = [];
  for (const note of notes) scored.push({ note, score: cosine(qv, await embed(note)) });
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const retrMs = Math.round(performance.now() - t0);
  const { TextStreamer } = mod;
  const messages = [{
    role: "system",
    content:
      "Answer the question using ONLY the provided context. If the context does not contain the answer, say you don't know. Be concise.",
  }, { role: "user", content: `Context:\n${best.note}\n\nQuestion: ${query}` }];
  post({ type: "retrieved", id, best, ranked: scored });
  let count = 0;
  const tg = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      count++;
      post({ type: "token", id, token });
    },
  });
  const out = await generator(messages, {
    max_new_tokens: 200,
    do_sample: false,
    streamer,
    return_full_text: false,
  });
  const genMs = Math.round(performance.now() - tg);
  const full = out?.[0]?.generated_text;
  const text = Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "");
  post({
    type: "done",
    id,
    text,
    tokens: count,
    retrMs,
    genMs,
    ms: Math.round(performance.now() - t0),
    device,
  });
}
self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.query, e.data.notes);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
