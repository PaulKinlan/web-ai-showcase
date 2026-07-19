// Spoken-language identification worker — ALL inference off the main thread so the control UI stays
// responsive. The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it
// here; the worker runs the audio-classification pipeline and returns the language scores over all 126
// languages (so the "see inside" surface can show the full distribution, entropy and margin), plus the
// latency + backend actually used.
//
// Model: Xenova/mms-lid-126 (task: audio-classification), q8 ONNX, WASM. This is Meta's MMS-LID head on
// a Wav2Vec2 (XLS-R 1B) backbone — it listens to the RAW waveform (no mel spectrogram) and predicts a
// spoken language. It is a big one-time download (~974 MB) because the backbone is a 1B-param model.
// We import the SHARED loader from lib/webai.js — no invented API.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/mms-lid-126";
const TASK = "audio-classification";
const SR = 16000;

let pipe = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
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

// Shannon entropy of the score distribution, normalised to [0,1] against log(N). High = the model is
// torn between languages (short/ambiguous/code-switched clip); low = one language dominates.
function normEntropy(scores) {
  let h = 0;
  for (const p of scores) if (p > 0) h -= p * Math.log(p);
  return h / Math.log(scores.length);
}

async function run(id, audio) {
  await ensureLoaded();
  const t0 = performance.now();
  // top_k = 126 → the FULL per-language distribution, so we can show every score + real calibration.
  const output = await pipe(new Float32Array(audio), { top_k: 126 });
  const ms = Math.round(performance.now() - t0);
  const all = (Array.isArray(output) ? output : [output]).map((o) => ({
    code: o.label,
    score: o.score,
  }));
  const scores = all.map((a) => a.score);
  const margin = (scores[0] ?? 0) - (scores[1] ?? 0);
  const entropy = normEntropy(scores);
  post({
    type: "result",
    id,
    all,
    entropy,
    margin,
    numClasses: all.length,
    ms,
    device,
    durationS: audio.length / SR,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.audio);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
