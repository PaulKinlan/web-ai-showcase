// Second-model worker for the multi-model page: the AudioSet generalist.
// Runs the AST (Audio Spectrogram Transformer) audio-classification pipeline — 527 AudioSet classes —
// off the main thread, so the multi-model page can compare the 9-class instrument SPECIALIST against a
// broad GENERALIST tagger on the same clip. This is the same model that backs the built
// `ast-audio-classification` demo. All inference off the main thread; no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/ast-finetuned-audioset-10-10-0.4593";
const TASK = "audio-classification";

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

async function run(id, audio) {
  await ensureLoaded();
  const t0 = performance.now();
  const output = await pipe(audio, { top_k: 5 });
  const ms = Math.round(performance.now() - t0);
  const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms, device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded(d.device);
    else if (d.type === "run") await run(d.id, d.audio);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
