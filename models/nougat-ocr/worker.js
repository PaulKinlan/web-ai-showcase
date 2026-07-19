// Nougat academic-document OCR worker — all inference off the main thread so the control UI stays
// responsive. Nougat is a VisionEncoderDecoder: a Donut-Swin image encoder looks at a whole scanned
// research PAGE and an mBART-style decoder GENERATES the page back as Markdown — with LaTeX math
// preserved (\( … \), \[ … \]) rather than flattened to garbled characters the way plain OCR does.
// We stream each decoded token back to the page (via a TextStreamer) with a timestamp, so the page can
// show the transcription forming and the per-token timing.
//
// Model: Xenova/nougat-small (task: image-to-text), WASM, q8. Donut-Swin encoder + mBART decoder.

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
    model: "Xenova/nougat-small",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function transcribe(id, imageURL, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const t0 = performance.now();
  let i = 0;
  let firstT = null;

  // Best-effort token streaming for the "See inside" trace. If the pipeline ignores the streamer we
  // still resolve with the real final transcription below — never a faked result.
  let streamer;
  try {
    streamer = new TextStreamer(pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token) => {
        if (!token) return;
        if (firstT === null) firstT = Math.round(performance.now() - t0);
        post({ type: "token", id, token, t: Math.round(performance.now() - t0), i: i++ });
      },
    });
  } catch { /* tokenizer/streamer unavailable — fall back to final text only */ }

  const out = await pipe(imageURL, {
    max_new_tokens: Math.max(16, Math.min(600, maxTokens || 300)),
    ...(streamer ? { streamer } : {}),
  });

  const first = Array.isArray(out) ? out[0] : out;
  const markdown = (first?.generated_text ?? first?.text ?? "").trim();
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, markdown, tokens: i, firstT: firstT ?? ms, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await transcribe(e.data.id, e.data.image, e.data.maxTokens);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
