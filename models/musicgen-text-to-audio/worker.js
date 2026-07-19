// MusicGen worker — generates MUSIC from a text prompt, entirely off the main thread so the control
// UI never janks during the (compute-heavy) decode. This is REAL generation: MusicGen autoregressively
// predicts EnCodec audio tokens, then EnCodec decodes them to a 32 kHz waveform. Nothing is faked.
//
// Model: Xenova/musicgen-small (task: text-to-audio). We use the canonical low-level classes
// (MusicgenForConditionalGeneration + AutoTokenizer) rather than the pipeline() helper, because that
// is the only way to attach a token-level streamer for HONEST progress during a long generation.
// dtype config is the known-good one from HF's own musicgen-web demo: q8 text-encoder + q8 decoder,
// but fp32 EnCodec decode (quantising the decoder produces garbled audio). Runs on WASM.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/musicgen-small";
// MusicGen's 32 kHz model emits EnCodec frames at 50 Hz → ~50 audio tokens per second of music.
const TOKENS_PER_SECOND = 50;

let model = null;
let tokenizer = null;
let mod = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

// Encode a mono Float32 waveform to a 16-bit PCM WAV ArrayBuffer (playback needs no extra deps).
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

async function ensureLoaded() {
  if (model) return;
  mod = await import(TRANSFORMERS_URL);
  const { MusicgenForConditionalGeneration, AutoTokenizer, env } = mod;
  env.allowLocalModels = false;
  // We're ALREADY in a dedicated worker — do NOT enable env.backends.onnx.wasm.proxy (that would
  // spawn a second proxy worker). MusicGen has no reliable WebGPU path with fp32 EnCodec, so we run
  // the known-good WASM config and report the backend honestly.
  console.log(`[musicgen worker] loading ${MODEL_ID} (~656 MB, WASM)…`);
  model = await MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
    dtype: {
      text_encoder: "q8",
      decoder_model_merged: "q8",
      encodec_decode: "fp32", // quantising EnCodec decode garbles the audio — keep it fp32.
    },
    device: "wasm",
  });
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  device = "wasm";
  console.log("[musicgen worker] ready (WASM)");
  post({ type: "ready", device });
}

// A minimal streamer: MusicGen calls put() once per decoding step, so counting calls gives a REAL
// token-level progress signal (not a spinner). end() marks completion.
function makeStreamer(id, maxLength) {
  const { BaseStreamer } = mod;
  let num = 0;
  return new (class extends BaseStreamer {
    put() {
      num++;
      post({ type: "gen", id, done: Math.min(num, maxLength), total: maxLength });
    }
    end() {
      post({ type: "gen", id, done: maxLength, total: maxLength });
    }
    get count() {
      return num;
    }
  })();
}

async function run(id, prompt, opts) {
  await ensureLoaded();
  const seconds = Math.max(1, Math.min(30, opts?.seconds ?? 5));
  const cap = model.generation_config?.max_length ?? 1500;
  const maxLength = Math.min(Math.floor(seconds * TOKENS_PER_SECOND) + 4, cap);

  const streamer = makeStreamer(id, maxLength);
  const inputs = tokenizer(prompt);

  const t0 = performance.now();
  const audioValues = await model.generate({
    ...inputs,
    max_length: maxLength,
    guidance_scale: opts?.guidanceScale ?? 3,
    temperature: opts?.temperature ?? 1,
    do_sample: true,
    streamer,
  });
  const ms = Math.round(performance.now() - t0);

  const rate = model.config.audio_encoder.sampling_rate || 32000;
  const pcm = audioValues.data instanceof Float32Array
    ? audioValues.data
    : Float32Array.from(audioValues.data);
  const durSec = pcm.length / rate;
  const wav = encodeWav(pcm, rate);

  post(
    {
      type: "result",
      id,
      pcm,
      wav,
      rate,
      durSec,
      ms,
      rtf: durSec ? ms / 1000 / durSec : null,
      tokens: streamer.count,
      maxLength,
      guidanceScale: opts?.guidanceScale ?? 3,
      temperature: opts?.temperature ?? 1,
      device,
    },
    [pcm.buffer, wav],
  );
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.prompt, e.data.opts);
    }
  } catch (err) {
    console.error("[musicgen worker]", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
