// German→English translation worker for the German wav2vec2 multi-model demo. Runs OPUS-MT (MarianMT) off
// the main thread so the ASR→translate chain never blocks the UI.
// Model: Xenova/opus-mt-de-en (task: translation) — the DEDICATED Helsinki-NLP German→English MarianMT
// (a real transformers.js ONNX build, higher quality for German than the multilingual mul-en model).
// Same MarianMT class as the built opus-mt-de-en / marianmt-translation demos.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/opus-mt-de-en";
const DEVICE = "wasm";
let translator = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (translator) return;
  const { pipeline } = await import(TRANSFORMERS_URL);
  translator = await pipeline("translation", MODEL_ID, {
    device: DEVICE,
    dtype: "q8", // onnx/*_quantized — dedicated de→en MarianMT, verified to translate German→English on WASM
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: DEVICE });
}

async function run(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await translator(text, { max_new_tokens: 200 });
  const ms = Math.round(performance.now() - t0);
  const english = Array.isArray(out)
    ? (out[0]?.translation_text ?? "")
    : String(out?.translation_text ?? "");
  post({ type: "result", id, text: english, ms, device: DEVICE });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.text);
  } catch (err) {
    console.error("[opus-mt-de-en worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
