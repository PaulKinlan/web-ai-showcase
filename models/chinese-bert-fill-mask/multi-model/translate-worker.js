// Chinese→English translation worker for the Chinese BERT multi-model demo. Off the main thread.
// Model: Xenova/opus-mt-zh-en (task: translation, MarianMT), WASM q8. The second stage of the chain:
// after Chinese BERT fills the [MASK], this reads the COMPLETED Chinese sentence and translates it to
// English so a non-Chinese reader can see what the fill produced.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/opus-mt-zh-en";
let translator = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (translator) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  translator = await pipeline("translation", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "wasm" });
}

async function translate(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await translator(text);
  const english = (Array.isArray(out) ? out[0] : out)?.translation_text ?? "";
  post({
    type: "translation",
    id,
    english,
    ms: Math.round(performance.now() - t0),
    device: "wasm",
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "translate") await translate(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
