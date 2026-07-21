// Music genre classification worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the worker
// runs the audio-classification pipeline and returns the genre probability distribution + latency + backend.
//
// Model: onnx-community/Musical-genres-Classification-Hubert-V1-ONNX (task: audio-classification), q8 ONNX,
// WASM. A DistilHuBERT fine-tuned on GTZAN — 10 genres: blues, classical, country, disco, hiphop, jazz,
// metal, pop, reggae, rock. Weights Apache-2.0 (base SeyedAli/Musical-genres-Classification-Hubert-V1 →
// ntu-spml/distilhubert, both Apache-2.0; the onnx-community conversion's blank field inherits it). We
// import the SHARED loader from lib/webai.js — no invented API. DISTINCT from the built AudioSet tagger
// (ast-audio-classification, 527 general sound events) and zero-shot CLAP: a dedicated MUSIC-GENRE verdict.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/Musical-genres-Classification-Hubert-V1-ONNX";
const TASK = "audio-classification";
const SR = 16000;

let pipe = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (pipe) return;
  const loaded = await loadPipeline({
    task: TASK,
    model: MODEL,
    backend: "wasm",
    dtype: "q8",
    onProgress: (p) => post({ type: "progress", p }),
  });
  pipe = loaded.pipe;
  device = loaded.device;
  post({ type: "ready", device });
}

// Classify one clip → the full 10-genre distribution (top_k = 10).
async function run(id, audio) {
  await ensureLoaded();
  const t0 = performance.now();
  const output = await pipe(audio, { top_k: 10 });
  const ms = Math.round(performance.now() - t0);
  const labels = (Array.isArray(output) ? output : [output]).map((o) => ({
    label: o.label,
    score: o.score,
  }));
  post({ type: "result", id, labels, ms, device, durationS: audio.length / SR });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded();
    else if (d.type === "run") await run(d.id, d.audio);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
