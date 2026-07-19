// Translation worker for the LaBSE multi-model demo — inference off the main thread.
// Model: Xenova/nllb-200-distilled-600M (task: translation), WASM, q8. ONE model translates directly
// between any pair of 200 languages via FLORES-200 codes. Here it is the SECOND stage: after LaBSE
// retrieves a passage cross-lingually, NLLB translates the winning (non-English) passage into English
// so the reader can see what it says. The short language codes from the page's guessLang() are mapped
// to FLORES-200 codes below.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

let pipe = null;
let device = "wasm";

// guessLang() short code → FLORES-200 code NLLB expects.
const FLORES = {
  en: "eng_Latn",
  fr: "fra_Latn",
  es: "spa_Latn",
  de: "deu_Latn",
  pt: "por_Latn",
  it: "ita_Latn",
  ja: "jpn_Jpan",
  ko: "kor_Hang",
  zh: "zho_Hans",
  ru: "rus_Cyrl",
  ar: "arb_Arab",
  hi: "hin_Deva",
  el: "ell_Grek",
};

export function toFlores(code) {
  return FLORES[code] || "eng_Latn";
}

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: "translation",
    model: "Xenova/nllb-200-distilled-600M",
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

async function translate(id, text, srcCode, tgtCode) {
  await ensureLoaded();
  const srcLang = FLORES[srcCode] || "eng_Latn";
  const tgtLang = FLORES[tgtCode] || "eng_Latn";
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
