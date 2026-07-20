// wav2vec2-XLSR-53 (Portuguese) CTC worker — ALL inference off the main thread.
//
// DISTINCT model from the built English wav2vec2 (models/wav2vec2-asr), the Russian XLSR
// (models/xlsr-multilingual-asr), the French XLSR (models/french-xlsr-asr), the Spanish XLSR
// (models/spanish-xlsr-asr) AND the Italian XLSR (models/italian-xlsr-asr). All are CTC, but this one is
// wav2vec2-large-XLSR-53 fine-tuned on PORTUGUESE (jonatasgrosman/wav2vec2-large-xlsr-53-portuguese): the
// cross-lingual model pre-trained on 53 languages (56k hours) then specialised for Portuguese. Its output
// alphabet is the 46-symbol Latin-Portuguese vocab — a-z plus the Portuguese diacritics (à á â ã ç é ê í
// ó ô õ ú ü). Feeding it English or Italian produces Portuguese-phonetic gibberish; it is
// language-specialised for Portuguese. That Portuguese specialisation — and the XLSR-53 transfer that made
// a good Portuguese recogniser trainable — is the point.
//
// Model: FinDIT-Studio/wav2vec2-large-xlsr-53-portuguese-onnx (ONNX export of jonatasgrosman's Portuguese
// XLSR-53). This repo ships ONE fp32 ONNX at the REPO ROOT (onnx/ subfolder absent) plus a tokenizer.json,
// but NO config.json and NO preprocessor_config.json. So:
//   • we load it with subfolder:"" + dtype:"fp32" (the q8/int8 exports do not exist here — fp32 is the
//     honest runnable one; the single self-contained model.onnx is ~1.2 GB);
//   • the missing config.json means AutoModel can't infer the architecture, so we load the architecture
//     config from the ORIGINAL repo it was exported from (jonatasgrosman/wav2vec2-large-xlsr-53-portuguese
//     — model_type "wav2vec2", Wav2Vec2ForCTC, vocab_size 46) and pass it in as `config`;
//   • the missing preprocessor_config.json means AutoFeatureExtractor would 404, so we do the wav2vec2
//     feature extraction ourselves — zero-mean/unit-variance normalisation of the raw 16 kHz waveform
//     (do_normalize / feat_extract_norm="layer" for XLSR-large);
//   • we do the greedy CTC decode ourselves from the model's 46-symbol vocab, which also lets us surface
//     the raw per-frame argmax strip + the CTC collapse + real per-word forced-alignment timings.
// Transformers.js via the SHARED 3.7.5 CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "FinDIT-Studio/wav2vec2-large-xlsr-53-portuguese-onnx";
const BASE_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"; // architecture config only (no weights)

// id → character, inverted from the model's tokenizer.json (deterministic for this pinned model).
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
  36: "ã",
  37: "ç",
  38: "é",
  39: "ê",
  40: "í",
  41: "ó",
  42: "ô",
  43: "õ",
  44: "ú",
  45: "ü",
};
const BLANK = 0;
const BOUNDARY = 4;

let model = null;
let Tensor = null;
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

// wav2vec2 feature extraction: normalise the raw waveform to zero mean / unit variance over the whole
// clip (do_normalize / feat_extract_norm="layer"). Returns a fresh Float32Array (input_values).
function normalize(audio) {
  const n = audio.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += audio[i];
  mean /= n || 1;
  let variance = 0;
  for (let i = 0; i < n; i++) {
    const d = audio[i] - mean;
    variance += d * d;
  }
  const std = Math.sqrt(variance / (n || 1)) + 1e-7;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (audio[i] - mean) / std;
  return out;
}

async function loadOn(dev) {
  const { AutoConfig, AutoModelForCTC, Tensor: T } = await import(TRANSFORMERS_URL);
  Tensor = T;
  // The FinDIT ONNX-export repo ships no config.json → load the architecture config from the original
  // jonatasgrosman repo it was exported from (tiny file, no weights), then load the ONNX with it.
  const config = await AutoConfig.from_pretrained(BASE_MODEL, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForCTC.from_pretrained(MODEL, {
    device: dev,
    dtype: "fp32", // the ONNX is a single fp32 export…
    subfolder: "", // …that lives at the repo ROOT (model.onnx), not the usual onnx/ subfolder.
    config, // …with the architecture config supplied from the base repo (repo has no config.json).
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
  const input_values = normalize(audio); // 16 kHz mono → zero-mean/unit-variance
  const { logits } = await model({
    input_values: new Tensor("float32", input_values, [1, input_values.length]),
  }); // Tensor [1, T, V=46]
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
    console.error("[xlsr-pt worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
