// OuteTTS worker — runs ALL synthesis off the main thread so the control UI stays responsive.
// OuteTTS is NOT a generic Transformers.js pipeline and NOT a StyleTTS-style vocoder like Kokoro. It is
// an LLM-based TTS: a Qwen2 500M language model autoregressively GENERATES discrete audio tokens from
// the text (plus an optional speaker profile), and a separate neural codec — WavTokenizer — DECODES
// those tokens back into a 24 kHz waveform. We use the model's real API: the `outetts` library (which
// wraps @huggingface/transformers under the hood), exactly as the model card prescribes. No invented
// surface.
//
// Models: onnx-community/OuteTTS-0.2-500M (the LLM, Qwen2, WebGPU or WASM) +
//         onnx-community/WavTokenizer-large-speech-75token_decode (the codec decoder, WASM/CPU only).

import { HFModelConfig_v1, InterfaceHF } from "https://cdn.jsdelivr.net/npm/outetts@0.2.0/+esm";

// navigator.gpu existing is not enough — headless / locked-down browsers expose the object but cannot
// return an adapter. Actually ask for one so we degrade to WASM honestly instead of stalling.
async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

let tts = null;
let device = "wasm";
let speakers = [];

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

// The English default speaker profiles shipped with OuteTTS v0.2.
const EN_SPEAKERS = [
  { id: "male_1", name: "Male 1" },
  { id: "male_2", name: "Male 2" },
  { id: "male_3", name: "Male 3" },
  { id: "male_4", name: "Male 4" },
  { id: "female_1", name: "Female 1" },
  { id: "female_2", name: "Female 2" },
];

async function ensureLoaded(preferred) {
  if (tts) return;
  const want = preferred || (await webgpuUsable() ? "webgpu" : "wasm");
  const cfg = new HFModelConfig_v1({
    model_path: "onnx-community/OuteTTS-0.2-500M",
    language: "en",
    dtype: "q8", // 8-bit LLM (~508 MB) + int8 WavTokenizer (~72 MB)
    device: want, // WebGPU accelerates the LLM; the WavTokenizer decoder always runs on WASM/CPU
  });
  try {
    tts = await InterfaceHF({ model_version: "0.2", cfg });
    device = want;
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      const cfg2 = new HFModelConfig_v1({
        model_path: "onnx-community/OuteTTS-0.2-500M",
        language: "en",
        dtype: "q8",
        device: "wasm",
      });
      tts = await InterfaceHF({ model_version: "0.2", cfg: cfg2 });
      device = "wasm";
    } else {
      throw err;
    }
  }
  speakers = EN_SPEAKERS;
  post({ type: "ready", device, speakers });
}

async function run(id, { text, voice, temperature, repetitionPenalty, maxLength }) {
  await ensureLoaded();
  const speaker = tts.load_default_speaker(voice || "male_1");
  const t0 = performance.now();
  const output = await tts.generate({
    text,
    temperature: temperature ?? 0.1,
    repetition_penalty: repetitionPenalty ?? 1.1,
    max_length: maxLength ?? 2048,
    speaker,
  });
  const ms = Math.round(performance.now() - t0);
  // output.audio is a transformers.js Tensor; .data is the Float32 waveform. output.sr is the rate.
  const pcm = output.audio?.data ?? output.audio;
  const rate = output.sr || 24000;
  const samples = pcm instanceof Float32Array ? pcm : Float32Array.from(pcm);
  const wav = encodeWav(samples, rate);
  const durSec = samples.length / rate;
  // OuteTTS/WavTokenizer emit 75 audio tokens per second of audio — a real, reportable count.
  const audioTokens = Math.round(durSec * 75);
  post(
    {
      type: "result",
      id,
      wav,
      pcm: samples,
      rate,
      durSec,
      audioTokens,
      genTokPerSec: durSec ? +(audioTokens / (ms / 1000)).toFixed(1) : null,
      ms,
      rtf: durSec ? ms / 1000 / durSec : null, // real-time factor: <1 is faster than realtime
      device,
      voice: voice || "male_1",
    },
    [wav, samples.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.device);
    else if (type === "run") await run(e.data.id, e.data);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
