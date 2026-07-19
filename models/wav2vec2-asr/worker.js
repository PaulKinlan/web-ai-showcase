// wav2vec2 (CTC) worker — runs ALL inference off the main thread so the UI stays responsive.
//
// This is a DIFFERENT architecture from Whisper/Moonshine/Distil-Whisper. Those are seq2seq: a decoder
// autoregressively generates text tokens. wav2vec2 is a CTC model — the network emits, for every ~20 ms
// audio frame, a probability distribution over a 32-symbol vocabulary (blank + "|" word boundary + 26
// letters + apostrophe). The transcript is the per-frame argmax with (a) consecutive duplicates merged
// and (b) blanks removed. No decoder, no autoregression, so it can't hallucinate words that weren't
// there. We deliberately DON'T use the ASR pipeline (it hides the frames) — we call the model directly
// so we can return the raw per-frame argmax strip AND real per-word forced-alignment timings for the viz.
//
// Model: Xenova/wav2vec2-base-960h (facebook/wav2vec2-base-960h, English), q8 quantized ONNX, WASM
// (WebGPU when a real adapter exists). We import Transformers.js via the SHARED CDN url from lib/webai.js.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/wav2vec2-base-960h";

// id → character, from the model's vocab.json (deterministic for this pinned model).
// 0 = <pad> (the CTC BLANK), 4 = "|" (word boundary), 1/2/3 = special tokens (rarely the argmax).
const ID2CHAR = {
  1: "",
  2: "",
  3: "?",
  4: " ",
  5: "E",
  6: "T",
  7: "A",
  8: "O",
  9: "N",
  10: "I",
  11: "H",
  12: "S",
  13: "R",
  14: "D",
  15: "L",
  16: "U",
  17: "M",
  18: "W",
  19: "C",
  20: "F",
  21: "G",
  22: "Y",
  23: "P",
  24: "B",
  25: "V",
  26: "K",
  27: "'",
  28: "X",
  29: "J",
  30: "Q",
  31: "Z",
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
    dtype: "q8", // maps to onnx/model_quantized.onnx
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
        // blank emits nothing, but it DOES separate two identical letters (prev reset below)
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

  const text = words.map((w) => w.text).join(" ").trim();
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
  const inputs = await processor(audio); // audio is already 16 kHz mono Float32Array
  const { logits } = await model(inputs); // Tensor [1, T, V]
  const ms = Math.round(performance.now() - t0);

  const [, T, V] = logits.dims;
  const data = logits.data; // Float32Array length T*V
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
    charsPerSec: ms ? decoded.collapsed.filter((c) => c !== " ").length / (ms / 1000) : null,
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
    console.error("[wav2vec2 worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
