// Two-model worker: DETECT the language, then ROUTE to the matching translator — both off the main thread.
//   Stage 1 (detector): onnx-community/language_detection-ONNX (text-classification).
//   Stage 2 (translator): Xenova/opus-mt-<lang>-en — a small bilingual MarianMT model, loaded on demand
//     for whichever language stage 1 detected. Only the translator you actually need is downloaded.
// This is the point of language ID as glue: you can't pick the right per-language model until you know
// the language, and detecting it locally lets you branch instantly with nothing sent to a server.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const DETECT_ID = "onnx-community/language_detection-ONNX";

// Marian bilingual models that translate INTO English, keyed by the detector's FLORES-200 label. Only
// these routes have a bundled translator; every other detected language reports "no translator bundled".
const TRANSLATORS = {
  fra_Latn: "Xenova/opus-mt-fr-en",
  deu_Latn: "Xenova/opus-mt-de-en",
  spa_Latn: "Xenova/opus-mt-es-en",
  ita_Latn: "Xenova/opus-mt-it-en",
  rus_Cyrl: "Xenova/opus-mt-ru-en",
  nld_Latn: "Xenova/opus-mt-nl-en",
  zho_Hans: "Xenova/opus-mt-zh-en",
  zho_Hant: "Xenova/opus-mt-zh-en",
};

let detectPipe = null;
const translators = new Map(); // lang → pipeline
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureDetector() {
  if (detectPipe) return;
  post({ type: "stage", stage: "Loading language detector…" });
  const d = await loadPipeline({
    task: "text-classification",
    model: DETECT_ID,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  detectPipe = d.pipe;
  device = d.device;
  post({ type: "ready", device });
}

async function ensureTranslator(lang) {
  if (translators.has(lang)) return translators.get(lang);
  const model = TRANSLATORS[lang];
  if (!model) return null;
  post({ type: "stage", stage: `Loading ${lang}→en translator…` });
  const t = await loadPipeline({
    task: "translation",
    model,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  translators.set(lang, t.pipe);
  return t.pipe;
}

async function detectAndTranslate(id, text) {
  await ensureDetector();
  const t0 = performance.now();
  const scores = await detectPipe(text, { top_k: 201 });
  const top = scores[0];
  const t1 = performance.now();

  let translation = null, translated = false, route = null, translateMs = 0, modelId = null;
  if (top.label === "eng_Latn") {
    route = "already-english";
  } else if (TRANSLATORS[top.label]) {
    route = "translate";
    modelId = TRANSLATORS[top.label];
    const tr = await ensureTranslator(top.label);
    const tStart = performance.now();
    const out = await tr(text);
    translation = Array.isArray(out) ? out[0].translation_text : out.translation_text;
    translated = true;
    translateMs = Math.round(performance.now() - tStart);
  } else {
    route = "no-translator";
  }

  post({
    type: "result",
    id,
    text,
    scores,
    detected: top.label,
    confidence: top.score,
    route,
    modelId,
    translation,
    translated,
    detectMs: Math.round(t1 - t0),
    translateMs,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureDetector();
    else if (type === "run") await detectAndTranslate(e.data.id, e.data.text);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
