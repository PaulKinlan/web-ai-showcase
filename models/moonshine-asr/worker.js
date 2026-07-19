// Moonshine ASR worker — runs ALL inference off the main thread so the UI stays responsive.
// Moonshine is a fast, low-latency speech-to-text model built for the browser/edge: unlike Whisper it
// doesn't pad every clip to 30 s, so short utterances transcribe in a fraction of the time. That speed
// is the whole point, so we measure and report real latency and a real-time factor — never a claim.
//
// Model: onnx-community/moonshine-base-ONNX (task: automatic-speech-recognition), q8 (quantized ONNX),
// WASM (WebGPU when a real adapter exists). We use the SHARED loader from lib/webai.js — no invented API.
// Moonshine has no timestamp tokens, so this returns plain text (that's expected, not a limitation bug).

import { loadPipeline } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/moonshine-base-ONNX";
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

// Load, preferring WebGPU when a real adapter is available; fall back to WASM honestly and report which
// backend actually ran.
async function ensureLoaded(preferred) {
  if (pipe) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  try {
    const loaded = await loadPipeline({
      task: TASK,
      model: MODEL,
      backend: want,
      dtype: "q8", // maps to the *_quantized.onnx files Moonshine ships
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

async function run(id, audio, audioDur) {
  await ensureLoaded();
  const t0 = performance.now();
  // Moonshine transcribes the whole (short) clip in one pass — no 30 s padding, no timestamp tokens.
  const output = await pipe(audio);
  const ms = Math.round(performance.now() - t0);
  const text = (Array.isArray(output) ? output.map((o) => o.text).join(" ") : output.text || "")
    .trim();

  // Real token count of the decoded text — a genuine number, not a claim.
  let tokens = null;
  try {
    const enc = pipe.tokenizer(text);
    tokens = enc?.input_ids?.dims
      ? enc.input_ids.dims[enc.input_ids.dims.length - 1]
      : (enc?.input_ids?.size ?? null);
  } catch {
    tokens = null;
  }

  const audioSec = audioDur || (audio.length / 16000);
  post({
    type: "result",
    id,
    text,
    ms,
    tokens,
    tokPerSec: tokens && ms ? tokens / (ms / 1000) : null,
    audioSec,
    // Real-time factor: processing time ÷ audio length. <1 means faster than real time.
    rtf: audioSec ? (ms / 1000) / audioSec : null,
    speedup: audioSec && ms ? audioSec / (ms / 1000) : null,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded(e.data.device);
    } else if (type === "run") {
      await run(e.data.id, e.data.audio, e.data.audioDur);
    }
  } catch (err) {
    console.error("[moonshine worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
