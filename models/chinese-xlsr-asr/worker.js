// wav2vec2-XLSR-53 (Mandarin Chinese) CTC worker — ALL inference off the main thread.
//
// DISTINCT model from the built English wav2vec2 (models/wav2vec2-asr), the Russian XLSR
// (models/xlsr-multilingual-asr) and the French/Spanish/Italian/Portuguese XLSR demos. All are CTC, but
// this one is wav2vec2-large-XLSR-53 fine-tuned on MANDARIN CHINESE
// (jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn): the cross-lingual model pre-trained on 53
// languages (56k hours) then specialised for Mandarin. Its output alphabet is NOT a Latin/Cyrillic
// letter set — it is a ~3,500-symbol CHARACTER vocab of Chinese hanzi (plus digits and Latin letters).
// A character-level CTC head with no language model, so it substitutes homophones the way a phonetic
// recogniser does (加/迦, 大/打, 木/目) — honest, real behaviour. Feeding it English produces
// Mandarin-phonetic character gibberish; it is language-specialised for Chinese.
//
// Model: onnx-community/wav2vec2-large-xlsr-53-chinese-zh-cn-ONNX (ONNX export of jonatasgrosman's
// Mandarin XLSR-53). This repo ships the STANDARD transformers.js layout (config.json,
// preprocessor_config.json, onnx/ subfolder with fp32 + quantized exports, tokenizer.json, vocab.json),
// so it loads directly as AutoModelForCTC. We load the 4-bit quantized export (dtype "q4", ~244 MB) —
// verified to produce coherent Mandarin (vs the 1.28 GB fp32) so the download is mobile-friendly. We load
// as AutoModelForCTC (NOT the ASR pipeline) so we can surface the raw per-frame argmax strip + the CTC
// collapse + real per-character forced-alignment timings. Because the vocab is ~3,500 characters (not a
// hand-list), we build the id→char map from the bundled vocab.json. Transformers.js via the SHARED 3.7.5
// CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/wav2vec2-large-xlsr-53-chinese-zh-cn-ONNX";
const VOCAB_URL = "/web-ai-showcase/models/chinese-xlsr-asr/vocab.json";

// 0 = <pad> (CTC BLANK), 4 = "|" (word boundary → space). 1/2/3 are <s>/</s>/<unk> (rarely argmax).
const BLANK = 0;
const BOUNDARY = 4;

let model = null;
let Tensor = null;
let id2char = null; // built from vocab.json (~3,500 entries)
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

async function loadVocab() {
  if (id2char) return;
  const vocab = await (await fetch(VOCAB_URL)).json(); // { "字": id, ... }
  id2char = {};
  for (const [c, i] of Object.entries(vocab)) id2char[i] = c;
}

async function loadOn(dev) {
  const { AutoModelForCTC, Tensor: T } = await import(TRANSFORMERS_URL);
  Tensor = T;
  await loadVocab();
  model = await AutoModelForCTC.from_pretrained(MODEL, {
    device: dev,
    dtype: "q4", // 4-bit quantized export (~244 MB) — verified coherent Mandarin; fp32 is 1.28 GB.
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

// Greedy CTC decode: argmax per frame, then collapse repeats and drop blanks. For a character-level
// (CJK) model we also emit per-CHARACTER forced-alignment timings (Chinese has no spaces, so per-word
// timing degenerates — per-character is the natural unit) plus the boundary-grouped "words" for parity.
function decodeCTC(ids, frameSec) {
  const strip = [];
  const collapsed = [];
  const chars = []; // { c, start, end } per emitted character
  const words = []; // groups split at "|" boundaries
  let curWord = null;
  let prev = -1;
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    const blank = id === BLANK;
    const boundary = id === BOUNDARY;
    strip.push({
      c: (id2char[id] && id2char[id].length === 1) ? id2char[id] : "",
      blank,
      boundary,
    });

    if (id !== prev) { // CTC: only act on a change of symbol (this merges duplicates)
      if (blank) {
        // blank emits nothing, but DOES separate two identical characters
      } else if (boundary) {
        collapsed.push(" ");
        if (curWord) {
          words.push(curWord);
          curWord = null;
        }
      } else {
        const ch = id2char[id];
        if (ch && ch.length === 1) { // real character (skips multi-char specials <s>/</s>/<unk>)
          collapsed.push(ch);
          chars.push({ c: ch, start: t * frameSec, end: (t + 1) * frameSec });
          if (!curWord) curWord = { text: "", startFrame: t, endFrame: t };
          curWord.text += ch;
          curWord.endFrame = t;
        }
      }
    }
    prev = id;
  }
  if (curWord) words.push(curWord);

  const text = collapsed.join("").replace(/\s+/g, " ").trim();
  const timedWords = words.map((w) => ({
    text: w.text,
    start: w.startFrame * frameSec,
    end: (w.endFrame + 1) * frameSec,
  }));
  return { strip, collapsed, text, chars, words: timedWords };
}

async function run(id, audio, audioDur) {
  await ensureLoaded();
  const t0 = performance.now();
  const input_values = normalize(audio); // 16 kHz mono → zero-mean/unit-variance
  const { logits } = await model({
    input_values: new Tensor("float32", input_values, [1, input_values.length]),
  }); // Tensor [1, T, V≈3503]
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
    chars: decoded.chars,
    words: decoded.words,
    frames: T,
    frameMs: frameSec * 1000,
    emitted: decoded.chars.length,
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
    console.error("[xlsr-zh worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
