// Distil-Whisper ASR worker — runs ALL inference off the main thread so the UI stays responsive.
//
// Distil-Whisper is Whisper knowledge-distilled: same encoder-decoder (seq2seq) architecture and the
// same pipeline call as Whisper, but the decoder is shrunk (distil-small.en keeps 12 encoder layers and
// drops to 4 decoder layers, vs whisper-small's 12). Fewer decoder passes per token → faster wall-clock
// transcription at close accuracy, English-only. That speed is the whole point, so we measure and report
// a REAL real-time factor and tok/s — never a claim. It still supports timestamps, so we return
// segment-level timings too.
//
// Model: distil-whisper/distil-small.en (task: automatic-speech-recognition), q8 (quantized ONNX),
// WASM (WebGPU when a real adapter exists). We use the SHARED loader from lib/webai.js — no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "distil-whisper/distil-small.en";
const TASK = "automatic-speech-recognition";

let pipe = null;
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

// Load, preferring WebGPU when a real adapter is available; fall back to WASM honestly and report which
// backend actually ran.
async function ensureLoaded(preferred) {
  if (pipe) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  try {
    const loaded = await loadPipeline({
      task: TASK,
      model: MODEL,
      backend: want,
      dtype: "q8", // maps to the *_quantized.onnx files (encoder + merged decoder)
      onProgress: (p) => post({ type: "progress", p }),
    });
    pipe = loaded.pipe;
    device = loaded.device;
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      const loaded = await loadPipeline({
        task: TASK,
        model: MODEL,
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

async function run(id, audio, audioDur) {
  await ensureLoaded();
  const t0 = performance.now();
  // Segment-level timestamps: distil-whisper keeps Whisper's timestamp tokens. Chunk long audio.
  const output = await pipe(audio, {
    return_timestamps: true,
    chunk_length_s: 20,
    stride_length_s: 3,
  });
  const ms = Math.round(performance.now() - t0);

  const text = (output.text || "").trim();
  const segments = (output.chunks || []).map((c) => ({
    start: c.timestamp?.[0] ?? null,
    end: c.timestamp?.[1] ?? null,
    text: (c.text || "").trim(),
  }));

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

  const audioSec = audioDur || (audio.length / 16000);
  post({
    type: "result",
    id,
    text,
    segments,
    tokens,
    tokPerSec: tokens && ms ? tokens / (ms / 1000) : null,
    ms,
    audioSec,
    // Real-time factor: processing time ÷ audio length. <1 means faster than real time.
    rtf: audioSec ? (ms / 1000) / audioSec : null,
    speedup: audioSec && ms ? audioSec / (ms / 1000) : null,
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
    console.error("[distil-whisper worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
