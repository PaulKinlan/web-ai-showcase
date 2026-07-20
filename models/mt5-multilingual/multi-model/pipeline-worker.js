// Two-model "detect then fill" worker — both models run off the main thread.
//   Stage 1 (language ID): onnx-community/xlm-roberta-base-language-detection-ONNX (text-classification)
//                          — reads the sentence and names its language.
//   Stage 2 (span fill):   Xenova/mt5-small (text2text-generation) — reconstructs the masked span.
// The point: you never tell mT5 which language to use. A separate detector labels the input language, and
// then mT5 — from its single 101-language vocabulary — fills the blank in that same language. Showing both
// stages makes the multilingual behaviour concrete and verifiable end to end.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const LANGID_ID = "onnx-community/xlm-roberta-base-language-detection-ONNX";
const MT5_ID = "Xenova/mt5-small";

let langPipe = null;
let mt5Pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (langPipe && mt5Pipe) return;
  if (!langPipe) {
    post({ type: "stage", stage: "Loading XLM-R language detector…" });
    const l = await loadPipeline({
      task: "text-classification",
      model: LANGID_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    langPipe = l.pipe;
    device = l.device;
  }
  if (!mt5Pipe) {
    post({ type: "stage", stage: "Loading mT5-small…" });
    const m = await loadPipeline({
      task: "text2text-generation",
      model: MT5_ID,
      backend: "wasm",
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    mt5Pipe = m.pipe;
  }
  post({ type: "ready", device });
}

const LANG_NAMES = {
  ar: "Arabic",
  bg: "Bulgarian",
  de: "German",
  el: "Greek",
  en: "English",
  es: "Spanish",
  fr: "French",
  hi: "Hindi",
  it: "Italian",
  ja: "Japanese",
  nl: "Dutch",
  pl: "Polish",
  pt: "Portuguese",
  ru: "Russian",
  sw: "Swahili",
  th: "Thai",
  tr: "Turkish",
  ur: "Urdu",
  vi: "Vietnamese",
  zh: "Chinese",
};

function reconstruct(input, output) {
  const fills = output.split(/<extra_id_\d+>/).map((s) => s.trim()).slice(1);
  let used = false;
  const recon = input.replace(/<extra_id_(\d+)>/g, (m, n) => {
    const f = fills[Number(n)];
    if (f != null && f !== "") {
      used = true;
      return `⟦${f}⟧`;
    }
    return m;
  });
  return used ? recon : null;
}

async function detectThenFill(id, input) {
  await ensureLoaded();

  // Stage 1: detect the language of the input (strip sentinels so the detector reads only real words).
  const forDetect = input.replace(/<extra_id_\d+>/g, " ").replace(/\s+/g, " ").trim();
  const t0 = performance.now();
  const det = await langPipe(forDetect, { top_k: 3 });
  const t1 = performance.now();
  const top = det[0];
  const langCode = top?.label ?? "?";
  const langName = LANG_NAMES[langCode] || langCode;

  // Stage 2: mT5 fills the blank — no language hint given anywhere.
  const out = await mt5Pipe(input, { max_new_tokens: 40, no_repeat_ngram_size: 3 });
  const output = (out[0]?.generated_text ?? "").trim();
  const t2 = performance.now();

  post({
    type: "result",
    id,
    langCode,
    langName,
    langScore: top?.score ?? 0,
    langAlts: det.slice(0, 3).map((d) => ({
      code: d.label,
      name: LANG_NAMES[d.label] || d.label,
      score: d.score,
    })),
    output,
    reconstruction: reconstruct(input, output),
    detectMs: Math.round(t1 - t0),
    fillMs: Math.round(t2 - t1),
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await detectThenFill(e.data.id, e.data.input);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
