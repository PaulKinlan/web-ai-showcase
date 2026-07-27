// ASR worker for the Demucs × Whisper multi-model demo. Runs Whisper OFF the main thread via the
// SHARED lib/webai.js loader (transformers.js) — no invented API. Demucs (onnxruntime-web) runs in its
// own worker (../worker.js); this one only transcribes the separated VOCALS stem, so the page can show
// what a second model hears in the first model's output.
//
// Model: onnx-community/whisper-tiny.en (automatic-speech-recognition) — a small English Whisper.
// WebGPU when a real adapter exists, WASM (q8) fallback otherwise.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

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

async function ensureLoaded() {
  if (pipe) return;
  const want = (await webgpuUsable()) ? "webgpu" : "wasm";
  try {
    const loaded = await loadPipeline({
      task: TASK,
      model: "onnx-community/whisper-tiny.en",
      backend: want,
      dtype: want === "webgpu" ? "fp32" : "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    pipe = loaded.pipe;
    device = loaded.device;
  } catch (err) {
    if (want !== "wasm") {
      const loaded = await loadPipeline({
        task: TASK,
        model: "onnx-community/whisper-tiny.en",
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

async function transcribe(id, audio) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(audio, { chunk_length_s: 30, return_timestamps: true });
  const ms = Math.round(performance.now() - t0);
  post({
    type: "result",
    id,
    text: (out?.text ?? "").trim(),
    chunks: out?.chunks ?? null,
    ms,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "transcribe") await transcribe(d.id, d.audio);
    else if (d.type === "dispose") {
      // Genuine release: drop the pipeline reference and close the worker from the inside (the main
      // thread also terminate()s it — belt and braces).
      pipe = null;
      post({ type: "disposed" });
      self.close();
    }
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
