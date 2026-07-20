// wav2vec2-large (VoxPopuli-Uralic) — Finnish CTC worker — ALL inference off the main thread.
//
// HONEST NAMING: this is NOT the XLSR-53 checkpoint. Neither jonatasgrosman/wav2vec2-large-xlsr-53-finnish
// nor aapot's XLSR-53 Finnish model ships a browser-loadable ONNX (both are safetensors/PyTorch only),
// so a true "Finnish XLSR-53" ONNX does not exist on the Hub today. The runnable Finnish wav2vec2 CTC
// ASR we CAN load is Finnish-NLP/wav2vec2-large-uralic-voxpopuli-v2-finnish: a wav2vec2-large model
// pre-trained on the VoxPopuli Uralic-languages subset (a DIFFERENT cross-lingual pretraining than
// XLSR-53's 53-language / 56k-hour corpus) and then fine-tuned on Finnish. It IS a Wav2Vec2ForCTC with a
// 34-symbol Latin-Finnish vocab (a-z + ä å ö + "|" word boundary + [PAD] blank + [UNK] + <s>/</s>). The
// page and metadata name it precisely for what it is — a VoxPopuli-Uralic Finnish recogniser — and do
// NOT claim it is XLSR-53.
//
// Model: KalleLaht/wav2vec2-large-uralic-voxpopuli-v2-finnish-ONNX — an ONNX export of the Finnish-NLP
// model above. This repo ships ONE fp32 ONNX at the REPO ROOT named wav2vec2_model.onnx (onnx/ subfolder
// absent) plus a processor folder, but NO config.json. So:
//   • we load it with subfolder:"" + model_file_name:"wav2vec2_model" + dtype:"fp32"
//     (no q8/int8 export exists here — fp32 is the honest runnable one; the file is ~1.26 GB);
//   • the graph has TWO inputs — input_values (float32) AND attention_mask — and the mask must be int32
//     (an all-ones mask, no padding, for a single clip);
//   • the missing config.json means AutoModel can't infer the architecture, so we load the config from
//     the ORIGINAL Finnish-NLP repo (model_type "wav2vec2", Wav2Vec2ForCTC, vocab_size 34) and pass it in;
//   • we do the wav2vec2 feature extraction ourselves — zero-mean/unit-variance normalisation of the raw
//     16 kHz waveform (do_normalize / feat_extract_norm="layer" for wav2vec2-large);
//   • we do the greedy CTC decode ourselves from the model's 34-symbol vocab. Finnish IS written with
//     spaces, and the model emits a "|" word-boundary symbol we render as a space.
// Transformers.js via the SHARED 3.7.5 CDN url.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "KalleLaht/wav2vec2-large-uralic-voxpopuli-v2-finnish-ONNX";
const MODEL_FILE = "wav2vec2_model"; // ONNX at repo root, non-standard filename
const BASE_MODEL = "Finnish-NLP/wav2vec2-large-uralic-voxpopuli-v2-finnish"; // architecture config only

// id → character, from the base repo's vocab.json (deterministic for this pinned model).
// 31 = [PAD] (the CTC BLANK), 0 = "|" (word boundary → space), 30 = [UNK], 32/33 = <s>/</s> (rare).
const ID2CHAR = {
  1: "a",
  2: "b",
  3: "c",
  4: "d",
  5: "e",
  6: "f",
  7: "g",
  8: "h",
  9: "i",
  10: "j",
  11: "k",
  12: "l",
  13: "m",
  14: "n",
  15: "o",
  16: "p",
  17: "q",
  18: "r",
  19: "s",
  20: "t",
  21: "u",
  22: "v",
  23: "w",
  24: "x",
  25: "y",
  26: "z",
  27: "ä",
  28: "å",
  29: "ö",
};
const BLANK = 31;
const BOUNDARY = 0;

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
  // Finnish-NLP repo it was exported from (tiny file, no weights), then load the ONNX with it.
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

// Greedy CTC decode: argmax per frame, then collapse repeats and drop blanks. The "|" boundary symbol
// becomes a space. Also derive real per-word forced-alignment timings from the frame each word ran at.
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
  const n = input_values.length;
  const { logits } = await model({
    input_values: new Tensor("float32", input_values, [1, n]),
    // this graph requires an attention_mask; int32 all-ones (no padding) for a single clip.
    attention_mask: new Tensor("int32", new Int32Array(n).fill(1), [1, n]),
  }); // Tensor [1, T, V=34]
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
    console.error("[voxpopuli-fi worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
