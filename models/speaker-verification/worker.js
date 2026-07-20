// Speaker verification / embedding worker — ALL inference off the main thread.
// The main thread decodes/records audio into a 16 kHz mono Float32Array and transfers it here; the
// worker turns each clip into a 512-D speaker EMBEDDING (an "x-vector") and returns it. Two clips are
// likely the SAME speaker when the cosine of their embeddings is high. This is on-device voice
// SIMILARITY — the audio and the embeddings never leave the tab. It is not identity, lookup, or
// surveillance.
//
// Model: Xenova/wavlm-base-plus-sv (WavLMForXVector, task: x-vector speaker embedding), q8 ONNX,
// ~102 MB. transformers.js 3.7.5 registers WavLMForXVector + AutoModelForXVector, but there is NO
// audio-xvector pipeline task — so we load the model + processor directly (no invented API). WebGPU is
// used when a real adapter exists; WASM otherwise.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/wavlm-base-plus-sv";

let processor = null;
let model = null;
let mod = null;
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

async function loadWith(dev) {
  const { AutoModelForXVector, AutoProcessor, env } = mod;
  env.allowLocalModels = false;
  processor = processor ||
    await AutoProcessor.from_pretrained(MODEL, {
      progress_callback: (p) => post({ type: "progress", p }),
    });
  model = await AutoModelForXVector.from_pretrained(MODEL, {
    dtype: "q8",
    device: dev,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = dev;
}

async function ensureLoaded(preferred) {
  if (model) return;
  mod = await import(TRANSFORMERS_URL);
  const want = preferred || ((await webgpuUsable()) ? "webgpu" : "wasm");
  try {
    await loadWith(want);
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      model = null;
      await loadWith("wasm");
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// Embed a 16 kHz mono Float32Array → an L2-normalized 512-D speaker embedding, so cosine == dot
// product and the same/different threshold is stable.
async function embed(id, audio, opts) {
  await ensureLoaded(opts?.device);
  const t0 = performance.now();
  const inputs = await processor(audio);
  const out = await model(inputs);
  const raw = out.embeddings.data; // Float32Array(512), NOT normalized
  let norm = 0;
  for (const v of raw) norm += v * v;
  norm = Math.sqrt(norm) || 1;
  const emb = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) emb[i] = raw[i] / norm;
  const ms = Math.round(performance.now() - t0);
  post(
    {
      type: "result",
      id,
      embedding: emb,
      dims: raw.length,
      ms,
      device,
      durationS: audio.length / 16000,
    },
    [emb.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "load") await ensureLoaded(d.device);
    else if (d.type === "run") await embed(d.id, d.audio, d.opts);
  } catch (err) {
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
