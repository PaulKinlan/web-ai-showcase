// Translation worker for the arctic-embed-v2 multi-model demo — inference off the main thread.
// Model: Xenova/m2m100_418M (task: translation), WASM, q8. One model translates directly between any
// pair of 100 languages via src_lang / tgt_lang. Here it's the SECOND stage: after the multilingual
// embedder retrieves a passage cross-lingually, we translate the winning (non-English) passage into
// English so the reader can see what it says. The language codes come from the page's guessLang().

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "translation",
    model: "Xenova/m2m100_418M",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function translate(id, text, srcLang, tgtLang) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(text, { src_lang: srcLang, tgt_lang: tgtLang, max_new_tokens: 200 });
  const ms = Math.round(performance.now() - t0);
  const translation = (out[0]?.translation_text ?? "").trim();
  post({ type: "result", id, translation, srcLang, tgtLang, ms, device });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") {
      await translate(e.data.id, e.data.text, e.data.srcLang, e.data.tgtLang);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
