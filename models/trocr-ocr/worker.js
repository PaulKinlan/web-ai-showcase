// TrOCR printed-text OCR worker — all inference off the main thread so the control UI stays
// responsive. TrOCR is a VisionEncoderDecoder: a ViT-style image encoder reads a cropped LINE of
// text into patch embeddings, and a RoBERTa-style text decoder transcribes it character/token by
// token. We stream each decoded token back to the page (via a TextStreamer) with a timestamp, so the
// page can show the transcription forming in real time and the per-token timing in "See inside".
//
// Model: Xenova/trocr-small-printed (task: image-to-text), WASM backend, q8. ~185 MB.

import { loadPipeline, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";
let mod = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  mod = await import(TRANSFORMERS_URL);
  const loaded = await loadPipeline({
    task: "image-to-text",
    model: "Xenova/trocr-small-printed",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function run(id, imageURL, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const t0 = performance.now();
  let i = 0;

  // Stream each newly decoded token to the page with a timestamp — the heart of "See inside".
  const streamer = new TextStreamer(pipe.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      if (!token) return;
      post({ type: "token", id, token, t: Math.round(performance.now() - t0), i: i++ });
    },
  });

  const out = await pipe(imageURL, {
    max_new_tokens: Math.max(1, Math.min(80, maxTokens || 48)),
    streamer,
  });

  const text = (Array.isArray(out) ? out[0]?.generated_text : out?.generated_text) ?? "";
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text: text.trim(), tokens: i, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.maxTokens);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
