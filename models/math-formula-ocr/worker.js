// Math-formula OCR worker — all inference off the main thread so the UI stays responsive.
// Reads an image of a mathematical equation and generates the LaTeX that produces it. Like Donut, texify is
// OCR-FREE end to end: a Donut-Swin encoder looks at the whole equation image and an mBART decoder
// generates the LaTeX token by token — no separate character segmentation.
//
// Model: Xenova/texify (task: image-to-text), WASM, q8. ~320 MB (Donut-Swin encoder + mBART decoder,
// merged decoder for cached generation). ONNX build of vikp/texify (cc-by-sa-4.0). DISTINCT from the built
// document OCR (nougat = whole academic pages → markdown; trocr = printed text lines; mgp-str = scene text):
// a DEDICATED math-formula recogniser that emits editable LaTeX. We stream each decoded token back to the
// page so the LaTeX forms live. Nothing leaves the tab.

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
    model: "Xenova/texify",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function recognise(id, imageURL, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const t0 = performance.now();
  let i = 0;

  // Best-effort token streaming for a live "LaTeX forming" effect. If the pipeline ignores the streamer we
  // still resolve with the real final LaTeX below — never a faked result.
  let streamer;
  try {
    streamer = new TextStreamer(pipe.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: (token) => {
        if (!token) return;
        post({ type: "token", id, token, t: Math.round(performance.now() - t0), i: i++ });
      },
    });
  } catch { /* tokenizer/streamer unavailable — fall back to final output only */ }

  const out = await pipe(imageURL, {
    max_new_tokens: Math.max(1, Math.min(384, maxTokens || 256)),
    ...(streamer ? { streamer } : {}),
  });

  const first = Array.isArray(out) ? out[0] : out;
  const latex = (first?.generated_text ?? "").trim();
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, latex, tokens: i, ms, device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "recognise") await recognise(d.id, d.imageURL, d.maxTokens);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
