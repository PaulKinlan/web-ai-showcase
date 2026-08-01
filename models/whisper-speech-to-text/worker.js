// Whisper ASR worker — runs ALL inference off the main thread so the UI stays responsive.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker runs the pipeline and returns the transcript plus real word/segment timestamps and a token
// count for a tok/s readout.
//
// Model: onnx-community/whisper-base (task: automatic-speech-recognition), WebGPU q4, WASM q8 fallback.
// We import the SHARED loader from lib/webai.js — no invented API. If WebGPU load fails we honestly
// retry on WASM and report which backend actually ran.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

// whisper-base, exported with cross-attentions so Transformers.js can extract real WORD-level
// timestamps (the plain onnx-community/whisper-base build only supports segment timestamps).
const TASK = "automatic-speech-recognition";

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// `navigator.gpu` existing is not enough — headless / locked-down browsers expose the object but can't
// return an adapter. Actually ask for one so we degrade to WASM honestly instead of failing to load.
async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

// Load, preferring WebGPU when a real adapter is available. If it still throws, fall back to WASM and
// say so — never present one backend as another.
//
// dtype differs BY BACKEND on purpose: int8 (q8) quantized Whisper on WebGPU hits a known ONNX
// Runtime bug (transformers.js #1317/#1512) that makes the decoder emit deterministic multilingual
// garbage loops with broken UTF-8 — the exact failure this guard exists for. WASM + q8 is verified
// clean; WebGPU uses q4 (accurate for whisper-base; q4f16 is the unstable small-model variant and is
// avoided here).
async function ensureLoaded(preferred) {
  if (pipe) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  const dtype = want === "webgpu" ? "q4" : "q8";
  try {
    const loaded = await loadPipeline({
      task: TASK,
      model: "onnx-community/whisper-base_timestamped",
      backend: want,
      dtype,
      onProgress: (p) => post({ type: "progress", p }),
    });
    pipe = loaded.pipe;
    device = loaded.device;
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      const loaded = await loadPipeline({
        task: TASK,
        model: "onnx-community/whisper-base_timestamped",
        backend: "wasm",
        dtype: "q8",
        onProgress: (p) => post({ type: "progress", p }),
      });
      pipe = loaded.pipe;
      device = loaded.device;
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// Garbage-output detector for the WebGPU corruption mode: broken UTF-8 replacement chars, or the
// same long window of text repeating over and over (the decoder repetition loop). Either means the
// transcript is not to be trusted.
function looksGarbled(text) {
  if (!text) return false;
  if (text.includes("\uFFFD")) return true;
  const compact = text.replace(/\s+/g, " ");
  const WINDOW = 60;
  if (compact.length >= WINDOW * 3) {
    for (let i = 0; i + WINDOW <= compact.length; i += 20) {
      const slice = compact.slice(i, i + WINDOW);
      let count = 0, from = 0;
      while ((from = compact.indexOf(slice, from)) !== -1) { count++; from += slice.length; }
      if (count >= 3) return true;
    }
  }
  return false;
}

// Group flat word chunks into readable segments (~sentence boundaries or ~8-word runs).
function toSegments(words) {
  const segs = [];
  let cur = null;
  for (const w of words) {
    if (!cur) cur = { text: "", start: w.timestamp?.[0] ?? null, end: null, words: [] };
    cur.text += w.text;
    cur.end = w.timestamp?.[1] ?? cur.end;
    cur.words.push(w);
    const endsSentence = /[.!?]["')\]]?\s*$/.test(w.text);
    if (endsSentence || cur.words.length >= 12) {
      cur.text = cur.text.trim();
      segs.push(cur);
      cur = null;
    }
  }
  if (cur) {
    cur.text = cur.text.trim();
    segs.push(cur);
  }
  return segs;
}

async function run(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  post({
    type: "stage",
    id,
    message: "Preparing audio features and decoding speech tokens locally…",
  });
  // Word-level timestamps give us both the karaoke word stream and (grouped) segments. The runtime
  // does not expose a trustworthy percentage for this inference, so the page shows elapsed time rather
  // than fabricating one. no_repeat_ngram_size guards against residual decoder repetition loops.
  let output = await pipe(audio, {
    return_timestamps: "word",
    chunk_length_s: 30,
    stride_length_s: 5,
    no_repeat_ngram_size: 3,
  });
  let note = null;

  // If WebGPU still produces garbled output (upstream numerical bug), reload on WASM once and
  // re-run — and say plainly that it happened. Never show garbage as if it were a transcript.
  if (device === "webgpu" && looksGarbled((output.text || "").trim())) {
    post({ type: "stage", id, message: "WebGPU output was unstable — re-running on WASM…" });
    try { pipe.dispose?.(); } catch { /* best effort */ }
    pipe = null;
    await ensureLoaded("wasm");
    output = await pipe(audio, {
      return_timestamps: "word",
      chunk_length_s: 30,
      stride_length_s: 5,
      no_repeat_ngram_size: 3,
    });
    note = "WebGPU produced garbled output on this clip, so it was re-transcribed on WASM.";
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "stage", id, message: "Formatting the transcript and timestamps…" });

  const text = (output.text || "").trim();
  const words = (output.chunks || []).map((c) => ({
    text: c.text,
    timestamp: c.timestamp,
  }));
  const segments = toSegments(words);

  // Real token count of the decoded text — a genuine number, not a claim.
  let tokens = null;
  try {
    const enc = pipe.tokenizer(text);
    tokens = enc?.input_ids?.dims
      ? enc.input_ids.dims[enc.input_ids.dims.length - 1]
      : (enc?.input_ids?.size ?? null);
  } catch {
    tokens = null;
  }
  const tokPerSec = tokens && ms ? tokens / (ms / 1000) : null;

  post({
    type: "result",
    id,
    text,
    words,
    segments,
    tokens,
    tokPerSec,
    ms,
    device,
    note,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded(e.data.device);
    } else if (type === "run") {
      await run(e.data.id, e.data.audio, e.data.opts);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
