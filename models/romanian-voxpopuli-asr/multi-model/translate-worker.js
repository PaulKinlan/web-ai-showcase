// Romanian→English translation worker for the Romanian wav2vec2 multi-model demo. Runs OPUS-MT (MarianMT)
// off the main thread so the ASR→translate chain never blocks the UI.
//
// Model: Xenova/opus-mt-mul-en (task: translation). There is no browser ONNX for Helsinki-NLP/opus-mt-ro-en
// (that repo is PyTorch-only, and Xenova/opus-mt-ro-en ships no .onnx), so we honestly use the multilingual
// OPUS-MT many-to-English model, which is trained on OPUS with Romanian among its source languages and DOES
// ship a real transformers.js-loadable ONNX. Same MarianMT class as the built marianmt-translation demo.
// Verified in headless Chrome to translate the Romanian ASR output into real English.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/opus-mt-mul-en";
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
    dtype: "q8", // onnx/*_quantized — verified to translate Romanian→en on WASM
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
    console.error("[opus-mt-mul worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
