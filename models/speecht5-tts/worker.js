// SpeechT5 TTS worker — all synthesis runs off the main thread so the control UI stays responsive.
// Model: Xenova/speecht5_tts (task: text-to-speech), q8 ONNX, honest WASM fallback. The pipeline
// auto-loads the matching HiFi-GAN vocoder (Xenova/speecht5_hifigan) internally.
//
// SpeechT5 is a DISTINCT TTS architecture from Kokoro: it is conditioned on a 512-dim SPEAKER
// X-VECTOR (a voice fingerprint). Swap the x-vector and the same text is spoken in a different voice —
// voice-cloning-adjacent, using the CMU-Arctic speaker embeddings the model was trained against.
//
// No invented surface: this is exactly how the transformers.js model card says to call it — a
// `speaker_embeddings` Float32Array (or URL) passed to the pipeline, which returns a 16 kHz waveform.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "Xenova/speecht5_tts";

async function webgpuUsable() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return false;
  try {
    return !!(await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}

let synth = null;
let device = "wasm";
const embCache = new Map(); // speaker id -> Float32Array (fetched once)

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Encode a mono Float32 waveform to a 16-bit PCM WAV ArrayBuffer (no extra deps for playback/download).
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
  dv.setUint16(20, 1, true);
  dv.setUint16(22, 1, true);
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

// A compact log-magnitude STFT of the ACTUAL generated waveform — the real spectrogram of what the
// model produced, not a decorative animation. Returns { data:Float32Array(frames*bins), frames, bins }.
function spectrogram(pcm, rate, fftSize = 512, hop = 256, bins = 64) {
  const win = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (fftSize - 1)); // Hann
  }
  const frames = Math.max(1, Math.floor((pcm.length - fftSize) / hop));
  const out = new Float32Array(frames * bins);
  const half = fftSize / 2;
  // Naive DFT per frame over `bins` mel-ish (linear) frequency slots — small sizes keep this cheap.
  for (let f = 0; f < frames; f++) {
    const start = f * hop;
    for (let b = 0; b < bins; b++) {
      const k = Math.floor((b / bins) * half) + 1;
      let re = 0, im = 0;
      for (let n = 0; n < fftSize; n++) {
        const s = (pcm[start + n] || 0) * win[n];
        const ang = (-2 * Math.PI * k * n) / fftSize;
        re += s * Math.cos(ang);
        im += s * Math.sin(ang);
      }
      out[f * bins + b] = Math.log10(1 + Math.sqrt(re * re + im * im));
    }
  }
  return { data: out, frames, bins };
}

async function loadWith(dev) {
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  return pipeline("text-to-speech", MODEL, {
    dtype: "q8",
    device: dev,
    progress_callback: (p) => post({ type: "progress", p }),
  });
}

async function ensureLoaded(preferred) {
  if (synth) return;
  const want = preferred || ((await webgpuUsable()) ? "webgpu" : "wasm");
  try {
    synth = await loadWith(want);
    device = want;
  } catch (err) {
    if (want !== "wasm") {
      post({ type: "progress", p: { status: "initiate", file: "retrying on WASM…" } });
      synth = await loadWith("wasm");
      device = "wasm";
    } else {
      throw err;
    }
  }
  post({ type: "ready", device });
}

// Fetch a bundled speaker x-vector (.bin of 512 float32) once, cache it, and hand back the array so the
// page can visualise the exact fingerprint used for synthesis.
async function loadEmbedding(spk) {
  if (embCache.has(spk.id)) return embCache.get(spk.id);
  const res = await fetch(spk.url);
  if (!res.ok) {
    throw new Error(`Couldn't load the "${spk.name}" speaker embedding (${res.status}).`);
  }
  const emb = new Float32Array(await res.arrayBuffer());
  if (emb.length !== 512) throw new Error(`Unexpected embedding size ${emb.length} (want 512).`);
  embCache.set(spk.id, emb);
  return emb;
}

async function run(id, text, spk) {
  await ensureLoaded();
  const emb = await loadEmbedding(spk);
  const t0 = performance.now();
  const out = await synth(text, { speaker_embeddings: emb });
  const ms = Math.round(performance.now() - t0);
  const pcm = out.audio; // Float32Array
  const rate = out.sampling_rate || 16000;
  const wav = encodeWav(pcm, rate);
  const durSec = pcm.length / rate;
  const spec = spectrogram(pcm, rate);
  // Ship a downsampled copy of the x-vector for the "see inside" heatmap (all 512 values, compact).
  const embCopy = new Float32Array(emb);
  post(
    {
      type: "result",
      id,
      wav,
      pcm,
      rate,
      durSec,
      ms,
      rtf: durSec ? ms / 1000 / durSec : null,
      device,
      voice: spk.name,
      emb: embCopy,
      spec: spec.data,
      specFrames: spec.frames,
      specBins: spec.bins,
    },
    [wav, pcm.buffer, embCopy.buffer, spec.data.buffer],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.device);
    else if (type === "run") await run(e.data.id, e.data.text, e.data.spk);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
