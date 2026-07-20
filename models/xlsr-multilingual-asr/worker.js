// wav2vec2-XLSR-53 (Russian) CTC worker — ALL inference off the main thread.
//
// This is a DISTINCT model from the English wav2vec2 demo (models/wav2vec2-asr). Both are CTC, but this
// one is wav2vec2-large-XLSR-53: a CROSS-LINGUAL model pre-trained on 53 languages (56k hours of
// multilingual speech) and then fine-tuned on Russian Common Voice. Its output alphabet is CYRILLIC —
// a 39-symbol vocab (blank + word-boundary "|" + hyphen + 33 Cyrillic letters incl. ё). Feeding it
// English would produce Cyrillic gibberish; it is specialised for Russian. That language specialisation,
// and the XLSR-53 cross-lingual transfer that made a good Russian recogniser trainable from limited
// data, is the whole point of this page.
//
// Model: onnx-community/wav2vec2-large-xlsr-53-russian-ONNX (base: jonatasgrosman/wav2vec2-large-xlsr-53-
// russian). q4 quantized ONNX (dtype "q4"), WASM (WebGPU when a real adapter exists). Loaded directly as
// AutoModelForCTC (NOT the ASR pipeline) so we can surface the raw per-frame argmax strip + the CTC
// collapse + real per-word forced-alignment timings. Transformers.js via the SHARED CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/wav2vec2-large-xlsr-53-russian-ONNX";

// id → character, inverted from the model's vocab.json (deterministic for this pinned model).
// 0 = <pad> (the CTC BLANK), 4 = "|" (word boundary → space). 1/2/3 are <s>/</s>/<unk> (rarely argmax).
const ID2CHAR = {
  4: " ",
  5: "-",
  6: "ё",
  7: "а",
  8: "б",
  9: "в",
  10: "г",
  11: "д",
  12: "е",
  13: "ж",
  14: "з",
  15: "и",
  16: "й",
  17: "к",
  18: "л",
  19: "м",
  20: "н",
  21: "о",
  22: "п",
  23: "р",
  24: "с",
  25: "т",
  26: "у",
  27: "ф",
  28: "х",
  29: "ц",
  30: "ч",
  31: "ш",
  32: "щ",
  33: "ъ",
  34: "ы",
  35: "ь",
  36: "э",
  37: "ю",
  38: "я",
};
const BLANK = 0;
const BOUNDARY = 4;

let model = null;
let processor = null;
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
  const { AutoProcessor, AutoModelForCTC } = await import(TRANSFORMERS_URL);
  processor = await AutoProcessor.from_pretrained(MODEL, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForCTC.from_pretrained(MODEL, {
    device: dev,
    dtype: "q4", // maps to onnx/model_q4.onnx — the q8/int8 exports are degenerate for this model
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
      processor = null;
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
  const inputs = await processor(audio); // 16 kHz mono Float32Array
  const { logits } = await model(inputs); // Tensor [1, T, V=39]
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
    console.error("[xlsr-ru worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
