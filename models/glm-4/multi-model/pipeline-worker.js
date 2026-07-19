// Cross-lingual RAG worker — two models, both off the main thread.
//   Retriever: Xenova/multilingual-e5-small (feature-extraction) — multilingual bi-encoder embeddings.
//   Generator: onnx-community/glm-edge-1.5b-chat-ONNX (text-generation) — Zhipu's bilingual GLM.
// The retriever embeds an EN+ZH corpus and the query, ranks chunks by cosine (works ACROSS languages
// because E5 is multilingual), and hands the top-k to GLM, which answers grounded in them — even when
// the question and the best chunk are in different languages. Retrieval on WASM (small), generation on
// WebGPU. Nothing is faked: if there's no adapter, generation reports an error the page surfaces.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const EMBED_ID = "Xenova/multilingual-e5-small";
const GEN_ID = "onnx-community/glm-edge-1.5b-chat-ONNX";

let embedPipe = null;
let generator = null;
let mod = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (embedPipe && generator) return;
  if (!embedPipe) {
    post({ type: "stage", stage: "Loading multilingual E5 retriever…" });
    const e = await loadPipeline({
      task: "feature-extraction",
      model: EMBED_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    embedPipe = e.pipe;
  }
  if (!generator) {
    post({ type: "stage", stage: "Loading GLM-Edge generator (WebGPU)…" });
    mod = await import(TRANSFORMERS_URL);
    generator = await mod.pipeline("text-generation", GEN_ID, {
      device: "webgpu",
      dtype: "q4f16",
      progress_callback: (p) => post({ type: "progress", p }),
    });
    device = "webgpu";
  }
  post({ type: "ready", device });
}

function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function embed(texts) {
  const out = await embedPipe(texts, { pooling: "mean", normalize: true });
  const dim = out.dims[out.dims.length - 1];
  const rows = [];
  for (let i = 0; i < texts.length; i++) {
    rows.push(Array.from(out.data.slice(i * dim, (i + 1) * dim), Number));
  }
  return rows;
}

async function run(id, query, passages, topK) {
  await ensureLoaded();
  const t0 = performance.now();

  // --- Stage 1: multilingual retrieval (E5 wants "query:" / "passage:" prefixes) ---
  const rows = await embed([`query: ${query}`, ...passages.map((p) => `passage: ${p}`)]);
  const qVec = rows[0];
  const scored = passages.map((passage, i) => ({
    idx: i,
    passage,
    cosine: cosine(qVec, rows[i + 1]),
  }));
  const retrieved = [...scored].sort((a, b) => b.cosine - a.cosine).slice(
    0,
    Math.min(topK, passages.length),
  );
  const t1 = performance.now();
  post({ type: "retrieved", id, retrieved });

  // --- Stage 2: GLM answers grounded ONLY in the retrieved chunks ---
  const context = retrieved.map((r, i) => `[${i + 1}] ${r.passage}`).join("\n");
  const sys =
    "You are a helpful assistant. Answer the user's question using ONLY the numbered context passages provided. The context may be in English or Chinese; answer in the language of the question. Cite the passage numbers you used like [1]. If the context doesn't contain the answer, say so.";
  const user = `Context:\n${context}\n\nQuestion: ${query}`;
  const messages = [{ role: "system", content: sys }, { role: "user", content: user }];

  const streamer = new mod.TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => post({ type: "token", id, token }),
  });
  const out = await generator(messages, {
    max_new_tokens: 320,
    do_sample: true,
    temperature: 0.4,
    top_p: 0.9,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
  });
  const full = out?.[0]?.generated_text;
  const answer = Array.isArray(full) ? (full.at(-1)?.content ?? "") : String(full ?? "");
  const t2 = performance.now();

  post({
    type: "result",
    id,
    answer,
    retrieved,
    retrieveMs: Math.round(t1 - t0),
    generateMs: Math.round(t2 - t1),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.query, e.data.passages, e.data.topK ?? 3);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
