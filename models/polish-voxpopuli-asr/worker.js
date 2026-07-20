// wav2vec2 (Polish) CTC worker — ALL inference off the main thread.
//
// DISTINCT model + language from every other ASR demo in this showcase. This is a POLISH speech
// recogniser: facebook/wav2vec2-base-10k-voxpopuli fine-tuned on Polish, mirrored to ONNX by
// onnx-community. All the built ASR demos (English wav2vec2, Russian/French/Spanish/Italian/
// Portuguese/Chinese/Japanese/Korean/Thai/Finnish XLSR) are CTC too, but none of them is Polish.
//
// HONEST NOTE ON THE BACKBONE: the canonical jonatasgrosman/wav2vec2-large-xlsr-53-polish checkpoint
// is PyTorch/Flax only — no wav2vec2-large-XLSR-53 Polish ONNX export exists anywhere on the Hub (a
// full Hub enumeration confirmed it, same as the blocked Dutch/Arabic XLSR seats). So this demo uses
// the base that DOES have a genuine, browser-loadable Polish CTC ONNX: wav2vec2-BASE-10k-voxpopuli
// (a ~95M-param wav2vec2-base pre-trained on 10k h of unlabelled VoxPopuli EU-Parliament audio, then
// fine-tuned on Polish). It is NOT the 315M XLSR-53 large model — the page says so plainly. Same
// Wav2Vec2ForCTC architecture, real Polish CTC head, so the CTC mechanics (per-frame argmax → collapse)
// are identical.
//
// Model: onnx-community/wav2vec2-base-10k-voxpopuli-ft-pl-ONNX. STANDARD transformers.js layout
// (config.json, preprocessor_config.json, onnx/ subfolder with fp32 + q4/q8/fp16 exports, tokenizer.json,
// vocab.json), so it loads directly as AutoModelForCTC — no borrowed config, no subfolder override. We
// load the 4-bit quantized export (dtype "q4", ~90 MB): verified in real headless Chrome to produce
// cleaner, more accurate Polish than the q8/int8 export (which is a touch degraded) and far smaller than
// the 378 MB fp32 — so the download is mobile-friendly. Loaded as AutoModelForCTC (NOT the ASR pipeline)
// so we can surface the raw per-frame argmax strip + the CTC collapse + real per-word forced-alignment
// timings. The id→char map is built from the bundled vocab.json (41 Latin-Polish symbols incl. the Polish
// diacritics ą ć ę ł ń ó ś ź ż). Transformers.js via the SHARED 3.7.5 CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/wav2vec2-base-10k-voxpopuli-ft-pl-ONNX";
const VOCAB_URL = "/web-ai-showcase/models/polish-voxpopuli-asr/vocab.json";

// 0 = <pad> (CTC BLANK), 4 = "|" (word boundary → space). 1/2/3 are 1/<s>/</s>/<unk> (rarely argmax).
const BLANK = 0;
const BOUNDARY = 4;

let model = null;
let Tensor = null;
let id2char = null; // built from vocab.json (41 entries)
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
// clip (the model's preprocessor_config sets do_normalize=true). Returns a fresh Float32Array.
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
  const vocab = await (await fetch(VOCAB_URL)).json(); // { "a": id, ... }
  id2char = {};
  for (const [c, i] of Object.entries(vocab)) id2char[i] = c;
  id2char[BOUNDARY] = " "; // "|" → space
}

async function loadOn(dev) {
  const { AutoModelForCTC, Tensor: T } = await import(TRANSFORMERS_URL);
  Tensor = T;
  await loadVocab();
  model = await AutoModelForCTC.from_pretrained(MODEL, {
    device: dev,
    dtype: "q4", // 4-bit export (~90 MB) — verified cleaner Polish than q8, far smaller than 378 MB fp32.
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
    const ch = id2char[id];
    strip.push({ c: (ch && ch !== " " && ch.length === 1) ? ch : "", blank, boundary });

    if (id !== prev) { // CTC: only act on a change of symbol (this merges duplicates)
      if (blank) {
        // blank emits nothing, but it DOES separate two identical letters
      } else if (boundary) {
        collapsed.push(" ");
        if (cur) {
          words.push(cur);
          cur = null;
        }
      } else if (ch && ch.length === 1) { // real character (skips multi-char specials <s>/</s>/<unk>)
        collapsed.push(ch);
        if (!cur) cur = { text: "", startFrame: t, endFrame: t };
        cur.text += ch;
        cur.endFrame = t;
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
  }); // Tensor [1, T, V=41]
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
    console.error("[wav2vec2-pl worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
