// wav2vec2 XLS-R voice-label worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker runs the checkpoint and returns BOTH acoustic voice labels with probabilities, the RAW
// logits behind them, the decision margin, and the latency + backend actually used.
//
// Model: Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech (task: audio-classification),
// q8 ONNX (~319 MB), WASM (WebGPU tried first when a real adapter exists). This is the ONNX mirror of
// the canonical alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech checkpoint (Apache-2.0):
// a wav2vec2-large (XLS-R) encoder + a 2-way sequence-classification head whose config id2label is
// {0: "female", 1: "male"} — BINARY ACOUSTIC VOICE LABELS learned from LibriSpeech audiobook speech,
// NOT a reading of anyone's gender identity.
//
// We load the pipeline via the SHARED loader from lib/webai.js (progress + honest device pick), then
// call pipe.processor + pipe.model directly so the "see inside" surface gets the exact pre-softmax
// logits [1,2] — not just the probabilities the pipeline's post-processing hands back.

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/wav2vec2-large-xlsr-53-gender-recognition-librispeech";
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

async function run(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  // One real forward pass: raw 16 kHz waveform → CNN feature encoder → transformer → mean-pool →
  // 2-way head. We read the logits BEFORE softmax so the page can show the model's exact numbers.
  const feats = await pipe.processor(audio);
  const out = await pipe.model(feats);
  const ms = Math.round(performance.now() - t0);
  const logits = Array.from(out.logits.data); // length 2, in config id order
  const id2label = pipe.model.config.id2label || { 0: "female", 1: "male" };
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  const labels = logits
    .map((logit, i) => ({ label: id2label[i] ?? `LABEL_${i}`, logit, score: exps[i] / sum }))
    .sort((a, b) => b.score - a.score);
  post({
    type: "result",
    id,
    labels, // both classes, most confident first
    logits, // raw, in config id order [0, 1]
    margin: Math.abs(labels[0].score - labels[1].score),
    ms,
    device,
    durationS: audio.length / SR,
    nSamples: audio.length,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.device);
    else if (type === "run") await run(e.data.id, e.data.audio, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
