// Kokoro TTS worker — runs ALL synthesis off the main thread so the control UI stays responsive.
// Kokoro is a StyleTTS2-style model, not a generic Transformers.js pipeline, so we use its real API:
// the `kokoro-js` library (which itself wraps @huggingface/transformers under the hood). No invented
// surface — this is exactly how the model card says to call it.
//
// Model: onnx-community/Kokoro-82M-v1.0-ONNX (task: text-to-speech), WebGPU q8, honest WASM fallback.

import { KokoroTTS } from "https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm";

const MODEL = "onnx-community/Kokoro-82M-v1.0-ONNX";

// `navigator.gpu` existing is not enough — headless / locked-down browsers expose the object but can't
// return an adapter. Actually ask for one so we degrade to WASM honestly instead of failing to load.
async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

let tts = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

// Encode a mono Float32 waveform to a 16-bit PCM WAV ArrayBuffer (so playback needs no extra deps).
function encodeWav(samples, rate) {
  const n = samples.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const dv = new DataView(buf);
  const wr = (o, s) => {
    for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i));
  };
  wr(0, "RIFF");
  dv.setUint32(4, 36 + n * 2, true);
  wr(8, "WAVE");
  wr(12, "fmt ");
  dv.setUint32(16, 16, true);
  dv.setUint16(20, 1, true); // PCM
  dv.setUint16(22, 1, true); // mono
  dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * 2, true);
  dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true);
  wr(36, "data");
  dv.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    dv.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return buf;
}

async function loadWith(dev) {
  return KokoroTTS.from_pretrained(MODEL, {
    dtype: "q8",
    device: dev,
    progress_callback: (p) => post({ type: "progress", p }),
  });
}

// Load, preferring WebGPU. If that throws, fall back to WASM and say which one actually ran.
async function ensureLoaded(preferred) {
  if (tts) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  try {
    tts = await loadWith(want);
    device = want;
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      tts = await loadWith("wasm");
      device = "wasm";
    } else {
      throw err;
    }
  }
  // Ship the real voice catalogue to the page so the picker isn't hard-coded.
  const voices = Object.entries(tts.voices || {}).map(([id, v]) => ({
    id,
    name: v.name || id,
    gender: v.gender || "",
    language: v.language || "",
    grade: v.overallGrade || "",
  }));
  post({ type: "ready", device, voices });
}

async function run(id, text, voice, speed) {
  await ensureLoaded();
  const t0 = performance.now();
  const audio = await tts.generate(text, { voice, speed });
  const ms = Math.round(performance.now() - t0);
  const pcm = audio.audio; // Float32Array
  const rate = audio.sampling_rate || 24000;
  const wav = encodeWav(pcm, rate);
  const durSec = pcm.length / rate;
  post(
    {
      type: "result",
      id,
      wav,
      pcm,
      rate,
      durSec,
      ms,
      rtf: durSec ? ms / 1000 / durSec : null, // real-time factor: <1 is faster than realtime
      device,
      voice,
    },
    [wav, pcm.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded(e.data.device);
    } else if (type === "run") {
      await run(e.data.id, e.data.text, e.data.voice, e.data.speed);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
