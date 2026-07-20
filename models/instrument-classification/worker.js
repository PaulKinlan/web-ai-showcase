// Musical Instrument Classification worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker runs the audio-classification pipeline and returns the per-instrument class probabilities +
// latency + backend. This is timbre recognition: WHICH instrument is playing, from the raw waveform.
//
// Model: onnx-community/Musical-Instrument-Classification-ONNX (task: audio-classification),
// a wav2vec2-base sequence-classifier (Wav2Vec2ForSequenceClassification). 9 classes:
// Acoustic_Guitar, Bass_Guitar, Drum_set, Electro_Guitar, flute, Hi_Hats, Keyboard, Trumpet, Violin.
// We import the SHARED loader from lib/webai.js — no invented API.
//
// dtype NOTE: this demo loads the q8 build (onnx/model_quantized.onnx). It's a wav2vec2 CNN+transformer
// classifier that runs on the universal WebAssembly path (and on WebGPU when a real adapter exists).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/Musical-Instrument-Classification-ONNX";
const TASK = "audio-classification";
const SR = 16000;
const NUM_CLASSES = 9;

let pipe = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

async function ensureLoaded(preferred) {
  if (pipe) return;
  const want = preferred || ((await webgpuUsable()) ? "webgpu" : "wasm");
  try {
    const loaded = await loadPipeline({
      task: TASK,
      model: MODEL,
      backend: want,
      dtype: "q8",
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

// Classify one clip → all class probabilities (top_k = NUM_CLASSES, the full label set).
async function run(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  const output = await pipe(audio, { top_k: NUM_CLASSES });
  const ms = Math.round(performance.now() - t0);
  const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms, device, durationS: audio.length / SR });
}

// Classify a series of windows (for the instrument-over-time timeline). Each window is a 16 kHz slice.
async function runWindows(id, windows, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  const out = [];
  for (let i = 0; i < windows.length; i++) {
    const w = windows[i];
    const output = await pipe(w, { top_k: NUM_CLASSES });
    const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
      label: o.label,
      score: o.score,
    }));
    out.push(labels);
    post({ type: "window", id, index: i, total: windows.length, labels });
  }
  const ms = Math.round(performance.now() - t0);
  post({ type: "windows-done", id, windows: out, ms, device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded(d.device);
    else if (d.type === "run") await run(d.id, d.audio, d.opts);
    else if (d.type === "run-windows") await runWindows(d.id, d.windows, d.opts);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
