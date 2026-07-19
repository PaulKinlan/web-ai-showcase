// Thin ASR worker for the "what was said vs how it sounded" multi-model demo. Runs Whisper OFF the main
// thread via the SHARED lib/webai.js loader (transformers.js). The emotion model runs in its own worker;
// this one only produces the transcript so the page can show words + voice-emotion side by side.
//
// Model: onnx-community/whisper-base (automatic-speech-recognition) — a built, verified demo in this
// repo. WebGPU when a real adapter exists, WASM fallback otherwise.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/whisper-base";
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
      model: MODEL,
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

async function transcribe(id, audio) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await pipe(audio);
  const ms = Math.round(performance.now() - t0);
  post({ type: "result", id, text: (out?.text ?? "").trim(), ms, device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "transcribe") await transcribe(d.id, d.audio);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
