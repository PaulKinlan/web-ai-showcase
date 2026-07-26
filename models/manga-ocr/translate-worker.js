// M2M100 (418M, q8, WASM) translation worker for the manga-ocr → English composition demo.
// Mirrors the pattern of models/m2m100-translation/worker.js: pipeline inference off the main
// thread, token streaming, real per-run timing. Model: Xenova/m2m100_418M (MIT; weights ODC-BY).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let TextStreamer = null;
let tokenizer = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const { pipeline, TextStreamer: TS, env } = await import(TRANSFORMERS_URL);
  TextStreamer = TS;
  env.allowLocalModels = false;
  pipe = await pipeline("translation", "Xenova/m2m100_418M", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  tokenizer = pipe.tokenizer;
  post({ type: "ready" });
}

async function translate(id, text) {
  await ensureLoaded();
  let partial = "";
  const t0 = performance.now();
  const out = await pipe(text, {
    src_lang: "ja",
    tgt_lang: "en",
    num_beams: 1,
    max_new_tokens: 256,
    streamer: new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (t) => {
        partial += t;
        post({ type: "stream", id, text: partial });
      },
    }),
  });
  const ms = Math.round(performance.now() - t0);
  const textOut = (Array.isArray(out) ? out[0] : out)?.translation_text ?? partial;
  post({ type: "result", id, text: textOut, ms });
}

self.onmessage = (e) => {
  const m = e.data || {};
  if (m.type === "load") {
    ensureLoaded().catch((err) => post({ type: "error", id: m.id ?? 0, message: String(err?.message || err) }));
  } else if (m.type === "run") {
    translate(m.id, m.text).catch((err) => post({ type: "error", id: m.id, message: String(err?.message || err) }));
  }
};
