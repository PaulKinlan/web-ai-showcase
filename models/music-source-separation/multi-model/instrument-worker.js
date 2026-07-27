// Second-model worker for the multi-model page: the 9-class instrument specialist.
// Runs the Musical Instrument Classification pipeline (onnx-community/Musical-Instrument-Classification-ONNX)
// — a forced-choice classifier over 9 instrument families (drums/hi-hats, bass guitar, guitars, keyboard,
// flute, trumpet, violin) — off the main thread, so the multi-model page can cross-examine Demucs's
// separated stems with an independent specialist. This is the same model that backs the built
// `instrument-classification` demo. The model id is a string literal AT the call site (repo convention,
// required by the portfolio-acceptance gate). All inference off the main thread.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

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
      task: "audio-classification",
      model: "onnx-community/Musical-Instrument-Classification-ONNX",
      backend: want,
      dtype: "q8",
      onProgress: (p) => post({ type: "progress", p }),
    });
    pipe = loaded.pipe;
    device = loaded.device;
  } catch (err) {
    if (want !== "wasm") {
      const loaded = await loadPipeline({
        task: "audio-classification",
        model: "onnx-community/Musical-Instrument-Classification-ONNX",
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

async function classify(id, pcm) {
  await ensureLoaded();
  const t0 = performance.now();
  const output = await pipe(pcm, { top_k: 4 });
  const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms: Math.round(performance.now() - t0), device });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded(d.backend);
    else if (d.type === "classify") await classify(d.id, d.pcm);
    else if (d.type === "dispose") {
      // Genuine teardown: release the pipeline session and close this worker.
      try {
        await pipe?.release?.();
      } catch { /* best effort */ }
      pipe = null;
      post({ type: "disposed" });
      self.close();
    }
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
