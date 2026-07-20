// wav2vec2-XLSR-53 (French) CTC worker — ALL inference off the main thread.
//
// DISTINCT model from both the English wav2vec2 demo (models/wav2vec2-asr) and the Russian XLSR demo
// (models/xlsr-multilingual-asr). All three are CTC, but this one is wav2vec2-large-XLSR-53 fine-tuned
// on FRENCH (jonatasgrosman/wav2vec2-large-xlsr-53-french): a cross-lingual model pre-trained on 53
// languages (56k hours) then specialised for French. Its output alphabet is the 59-symbol Latin-French
// vocab — a-z plus the full set of French diacritics (à â ç è é ê ë î ï ô ù û ü œ …). Feeding it English
// or Russian produces French-phonetic gibberish; it is language-specialised for French. That French
// specialisation — and the XLSR-53 transfer that made a good French recogniser trainable — is the point.
//
// Model: Poulpidot/wav2vec2-large-xlsr-53-french-onnx (ONNX export of jonatasgrosman's French XLSR-53).
// This repo ships ONE fp32 ONNX at the REPO ROOT (onnx/ subfolder absent), so we load it with
// subfolder:"" + dtype:"fp32" (the q8/int8 exports do not exist here — fp32 is the honest runnable one).
// It also ships NO tokenizer.json, so AutoProcessor fails; we load the FEATURE EXTRACTOR directly with
// AutoFeatureExtractor (reads preprocessor_config.json) and do the CTC decode ourselves from the model's
// vocab.json — which also lets us surface the raw per-frame argmax strip + the CTC collapse + real
// per-word forced-alignment timings. Transformers.js via the SHARED 3.7.5 CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Poulpidot/wav2vec2-large-xlsr-53-french-onnx";

// id → character, inverted from the model's vocab.json (deterministic for this pinned model).
// 0 = <pad> (the CTC BLANK), 4 = "|" (word boundary → space). 1/2/3 are <s>/</s>/<unk> (rarely argmax).
const ID2CHAR = {
  4: " ",
  5: "'",
  6: "-",
  7: "a",
  8: "b",
  9: "c",
  10: "d",
  11: "e",
  12: "f",
  13: "g",
  14: "h",
  15: "i",
  16: "j",
  17: "k",
  18: "l",
  19: "m",
  20: "n",
  21: "o",
  22: "p",
  23: "q",
  24: "r",
  25: "s",
  26: "t",
  27: "u",
  28: "v",
  29: "w",
  30: "x",
  31: "y",
  32: "z",
  33: "à",
  34: "á",
  35: "â",
  36: "ä",
  37: "ç",
  38: "è",
  39: "é",
  40: "ê",
  41: "ë",
  42: "í",
  43: "î",
  44: "ï",
  45: "ñ",
  46: "ó",
  47: "ô",
  48: "ö",
  49: "ù",
  50: "ú",
  51: "û",
  52: "ü",
  53: "ć",
  54: "č",
  55: "ō",
  56: "œ",
  57: "š",
  58: "ș",
};
const BLANK = 0;
const BOUNDARY = 4;

let model = null;
let featureExtractor = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function loadOn(dev) {
  const { AutoFeatureExtractor, AutoModelForCTC } = await import(TRANSFORMERS_URL);
  // No tokenizer.json in this repo → AutoProcessor would 404. The feature extractor is all we need to
  // turn 16 kHz mono audio into input_values; we decode the CTC logits ourselves from ID2CHAR.
  featureExtractor = await AutoFeatureExtractor.from_pretrained(MODEL, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForCTC.from_pretrained(MODEL, {
    device: dev,
    dtype: "fp32", // the ONNX is a single fp32 export…
    subfolder: "", // …that lives at the repo ROOT (model.onnx), not the usual onnx/ subfolder.
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = dev;
}

async function ensureLoaded(preferred) {
  if (model) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  try {
    await loadOn(want);
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      model = null;
      featureExtractor = null;
      await loadOn("wasm");
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// Greedy CTC decode: argmax per frame, then collapse repeats and drop blanks. Also derive real per-word
// forced-alignment timings from the frame index at which each character was emitted.
function decodeCTC(ids, frameSec) {
  const strip = [];
  const collapsed = [];
  const words = [];
  let cur = null;
  let prev = -1;
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    const blank = id === BLANK;
    const boundary = id === BOUNDARY;
    strip.push({ c: ID2CHAR[id] ?? "", blank, boundary });

    if (id !== prev) { // CTC: only act on a change of symbol (this merges duplicates)
      if (blank) {
        // blank emits nothing, but it DOES separate two identical letters
      } else if (boundary) {
        collapsed.push(" ");
        if (cur) {
          words.push(cur);
          cur = null;
        }
      } else {
        const ch = ID2CHAR[id] ?? "";
        if (ch) {
          collapsed.push(ch);
          if (!cur) cur = { text: "", startFrame: t, endFrame: t };
          cur.text += ch;
          cur.endFrame = t;
        }
      }
    }
    prev = id;
  }
  if (cur) words.push(cur);

  const text = words.map((w) => w.text).join(" ").replace(/\s+/g, " ").trim();
  const timedWords = words.map((w) => ({
    text: w.text,
    start: w.startFrame * frameSec,
    end: (w.endFrame + 1) * frameSec,
  }));
  return { strip, collapsed, text, words: timedWords };
}

async function run(id, audio, audioDur) {
  await ensureLoaded();
  const t0 = performance.now();
  const inputs = await featureExtractor(audio); // 16 kHz mono Float32Array → { input_values, ... }
  const { logits } = await model(inputs); // Tensor [1, T, V=59]
  const ms = Math.round(performance.now() - t0);

  const [, T, V] = logits.dims;
  const data = logits.data;
  const ids = new Array(T);
  for (let t = 0; t < T; t++) {
    let best = -Infinity;
    let bi = 0;
    const base = t * V;
    for (let v = 0; v < V; v++) {
      const val = data[base + v];
      if (val > best) {
        best = val;
        bi = v;
      }
    }
    ids[t] = bi;
  }

  const audioSec = audioDur || (audio.length / 16000);
  const frameSec = T ? audioSec / T : 0;
  const decoded = decodeCTC(ids, frameSec);

  post({
    type: "result",
    id,
    text: decoded.text,
    strip: decoded.strip,
    collapsed: decoded.collapsed,
    words: decoded.words,
    frames: T,
    frameMs: frameSec * 1000,
    emitted: decoded.collapsed.filter((c) => c !== " ").length,
    audioSec,
    rtf: audioSec ? (ms / 1000) / audioSec : null,
    speedup: audioSec && ms ? audioSec / (ms / 1000) : null,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded(e.data.device);
    } else if (type === "run") {
      await run(e.data.id, e.data.audio, e.data.audioDur);
    }
  } catch (err) {
    console.error("[xlsr-fr worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
