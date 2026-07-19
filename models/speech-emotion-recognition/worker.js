// Speech Emotion Recognition worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker runs the audio-classification pipeline and returns the emotion class probabilities + latency +
// backend. This is affective audio: emotion from the VOICE, not the words.
//
// Model: onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX (task: audio-classification),
// q8 ONNX, WASM (WebGPU when a real adapter exists). 6 classes: ANGRY, DISGUST, FEAR, HAPPY, NEUTRAL,
// SAD. We import the SHARED loader from lib/webai.js — no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/wav2vec2-base-Speech_Emotion_Recognition-ONNX";
const TASK = "audio-classification";
const SR = 16000;

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

// Classify one clip → all class probabilities (top_k = 6, the full label set).
async function run(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  const output = await pipe(audio, { top_k: 6 });
  const ms = Math.round(performance.now() - t0);
  const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms, device, durationS: audio.length / SR });
}

// Classify a series of windows (for the emotion-over-time timeline). Each window is a 16 kHz slice.
async function runWindows(id, windows, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  const out = [];
  for (let i = 0; i < windows.length; i++) {
    const w = windows[i];
    const output = await pipe(w, { top_k: 6 });
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
