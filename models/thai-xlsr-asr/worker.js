// wav2vec2-XLSR-53 (Thai) CTC worker — ALL inference off the main thread.
//
// DISTINCT model from the built English wav2vec2 (models/wav2vec2-asr), the Russian XLSR
// (models/xlsr-multilingual-asr), the French/Spanish/Italian/Portuguese/Chinese/Japanese/Korean XLSR
// demos. All are CTC, but this one is wav2vec2-large-XLSR-53 fine-tuned on THAI
// (airesearch/wav2vec2-large-xlsr-53-th): the cross-lingual model pre-trained on 53 languages
// (56k hours) then specialised for Thai. Its output alphabet is a 70-symbol Thai vocab — Thai
// consonants, vowels and tone marks (ก ข … ะ ั า … ่ ้ ๊ ๋) plus a "|" word-boundary symbol. Thai
// is written WITHOUT spaces between words, but this model was fine-tuned on word-segmented text
// (PyThaiNLP tokenisation), so it emits the "|" boundary symbol between words — we render those as
// spaces so the word segmentation is visible. Feeding it English or Thai-less audio produces
// Thai-phonetic gibberish; it is language-specialised for Thai.
//
// Model: BlackHand013/Wav2Vec2-large-xlsr-53-th-onnx — an ONNX export of airesearch's Thai XLSR-53.
// This repo ships ONE fp32 ONNX at the REPO ROOT (onnx/ subfolder absent), named
// wav2vec2-large-xlsr-53-th.onnx, with NO config.json and NO vocab. So:
//   • we load it with subfolder:"" + model_file_name:"wav2vec2-large-xlsr-53-th" + dtype:"fp32"
//     (no q8/int8 export exists here — fp32 is the honest runnable one; the file is ~1.26 GB);
//   • the ONNX graph's single input is named `input` (not the usual `input_values`);
//   • the missing config.json means AutoModel can't infer the architecture, so we load the config
//     from the ORIGINAL airesearch repo (model_type "wav2vec2", Wav2Vec2ForCTC, vocab_size 70) and
//     pass it in as `config`;
//   • we do the wav2vec2 feature extraction ourselves — zero-mean/unit-variance normalisation of the
//     raw 16 kHz waveform (do_normalize / feat_extract_norm="layer" for XLSR-large);
//   • we do the greedy CTC decode ourselves from the model's 70-symbol vocab.
// Transformers.js via the SHARED 3.7.5 CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "BlackHand013/Wav2Vec2-large-xlsr-53-th-onnx";
const MODEL_FILE = "wav2vec2-large-xlsr-53-th"; // ONNX at repo root, non-standard filename
const BASE_MODEL = "airesearch/wav2vec2-large-xlsr-53-th"; // architecture config only (no weights)
const INPUT_NAME = "input"; // this ONNX graph's input node is `input`, not `input_values`

// id → character, from airesearch's vocab.json (deterministic for this pinned model).
// 69 = [PAD] (the CTC BLANK), 42 = "|" (word boundary → space), 68 = [UNK].
const ID2CHAR = {
  0: "ฑ",
  1: "ๅ",
  2: "ก",
  3: "ง",
  4: "ฒ",
  5: "ะ",
  6: "๊",
  7: "้",
  8: "ฌ",
  9: "ซ",
  10: "ด",
  11: "ฯ",
  12: "ใ",
  13: "ึ",
  14: "ญ",
  15: "่",
  16: "า",
  17: "ฤ",
  18: "๋",
  19: "อ",
  20: "ฬ",
  21: "ท",
  22: "โ",
  23: "ภ",
  24: "ย",
  25: "็",
  26: "ล",
  27: "ุ",
  28: "เ",
  29: "ฮ",
  30: "ฝ",
  31: "ป",
  32: "ี",
  33: "บ",
  34: "ฐ",
  35: "ต",
  36: "ถ",
  37: "ศ",
  38: "ฟ",
  39: "ณ",
  40: "ห",
  41: "ร",
  43: "พ",
  44: "ฆ",
  45: "ั",
  46: "ค",
  47: "ว",
  48: "ฏ",
  49: "จ",
  50: "แ",
  51: "ม",
  52: "ฎ",
  53: "ฉ",
  54: "์",
  55: "ษ",
  56: "ำ",
  57: "ผ",
  58: "ข",
  59: "ไ",
  60: "ู",
  61: "ื",
  62: "น",
  63: "ช",
  64: "ิ",
  65: "ธ",
  66: "ฃ",
  67: "ส",
};
const BLANK = 69;
const BOUNDARY = 42;

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
// clip (do_normalize / feat_extract_norm="layer"). Returns a fresh Float32Array.
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
  // The ONNX-export repo ships no config.json → load the architecture config from the original
  // airesearch repo it was exported from (tiny file, no weights), then load the ONNX with it.
  const config = await AutoConfig.from_pretrained(BASE_MODEL, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForCTC.from_pretrained(MODEL, {
    device: dev,
    dtype: "fp32", // the ONNX is a single fp32 export…
    subfolder: "", // …that lives at the repo ROOT, not the usual onnx/ subfolder…
    model_file_name: MODEL_FILE, // …under a non-standard filename…
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

// Greedy CTC decode: argmax per frame, then collapse repeats and drop blanks. The "|" boundary
// symbol becomes a space so the model's word segmentation is visible (Thai has no native spaces).
// Also derive real per-word forced-alignment timings from the frame index at which each word ran.
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
        // blank emits nothing, but it DOES separate two identical symbols
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
  const feeds = {};
  feeds[INPUT_NAME] = new Tensor("float32", input_values, [1, input_values.length]);
  const { logits } = await model(feeds); // Tensor [1, T, V=70]
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
    console.error("[xlsr-th worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
