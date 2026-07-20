// BLOOMZ multi-model worker — a real two-model RAG composition, off the main thread:
//   1. all-MiniLM-L6-v2 (Transformers.js feature-extraction) embeds the question + every knowledge
//      note and picks the most similar note by cosine similarity (the "retrieval" step).
//   2. BLOOMZ-560m answers the question grounded ONLY in that retrieved note (the "generation" step).
// Embedder: Xenova/all-MiniLM-L6-v2 (~25 MB, q8). LLM: Xenova/bloomz-560m (~350 MB, q8, legacy layout).
//
// This is classic Retrieval-Augmented Generation, entirely on-device: no vector DB service, no LLM API.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const EMB_ID = "Xenova/all-MiniLM-L6-v2";
const LLM_ID = "Xenova/bloomz-560m";
let mod = null, embedder = null, generator = null;
let device = "wasm";

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
    dtype: "q8",
    model_file_name: "decoder_model_merged",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "wasm";
  post({ type: "ready", device });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
async function embed(text) {
  const out = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(out.data);
}

async function run(id, query, notes) {
  await ensureLoaded();
  const t0 = performance.now();
  // 1) retrieval
  const qv = await embed(query);
  const scored = [];
  for (const note of notes) {
    const nv = await embed(note);
    scored.push({ note, score: cosine(qv, nv) });
  }
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  const retrMs = Math.round(performance.now() - t0);

  // 2) grounded generation
  const { TextStreamer } = mod;
  const prompt = `Context: ${best.note}\nQuestion: ${query}\nAnswer:`;
  post({ type: "retrieved", id, best, ranked: scored, prompt });
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
  const out = await generator(prompt, {
    max_new_tokens: 48,
    do_sample: false,
    repetition_penalty: 1.3,
    streamer,
    return_full_text: false,
  });
  const genMs = Math.round(performance.now() - tg);
  const full = out?.[0]?.generated_text;
  const text = typeof full === "string" ? full.slice(prompt.length) : String(full ?? "");
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
