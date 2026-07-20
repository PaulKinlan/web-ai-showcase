// MMS-TTS Yoruba (VITS) worker — end-to-end Yoruba speech synthesis off the main thread so the control UI
// never janks during synthesis. REAL inference: Meta's MMS-TTS is a VITS model — a single-stage,
// end-to-end conditional VAE with a normalizing-flow decoder and a STOCHASTIC duration predictor. There
// is no separate vocoder: Yoruba text → token ids → (flows + duration) → 16 kHz waveform, all in one
// network. Nothing here is faked.
//
// This demo is the YORUBA checkpoint (Xenova/mms-tts-yor) — a materially DISTINCT model from the other
// built MMS-TTS demos: its own native Yoruba vocabulary (43 symbols, is_uroman:false). Yoruba orthography
// is Latin-based but carries the sub-dot letters ẹ/ọ/ṣ and the tone marks (à/á, è/é) that encode Yoruba's
// three tones — glyphs no other checkpoint's vocab includes. Yoruba is a major West African tonal
// language. This Xenova export ships the full tokenizer (tokenizer.json) + an onnx/ subfolder, so it
// loads with the plain pipeline("text-to-speech") — the real VitsTokenizer tokenises Yoruba directly.
//
// Because the duration predictor is stochastic, generating the same Yoruba text twice yields DIFFERENT
// timing/prosody — that variation is real and is what the "stochastic prosody" surface visualises.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const DEFAULT_MODEL = "Xenova/mms-tts-yor";
const pipes = new Map(); // modelId -> pipeline
let device = "wasm";
let transformers = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function getPipeline(modelId, notifyProgress) {
  if (pipes.has(modelId)) return pipes.get(modelId);
  if (!transformers) transformers = await import(TRANSFORMERS_URL);
  const { pipeline, env } = transformers;
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the service worker.
  const tts = await pipeline("text-to-speech", modelId, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => notifyProgress && post({ type: "progress", p }),
  });
  pipes.set(modelId, tts);
  return tts;
}

async function synth(id, modelId, text) {
  const tts = await getPipeline(modelId, false);
  const t0 = performance.now();
  const out = await tts(text);
  const ms = Math.round(performance.now() - t0);
  // out.audio is a Float32Array; transfer its buffer to avoid a copy.
  const audio = out.audio instanceof Float32Array ? out.audio : Float32Array.from(out.audio);
  const rate = out.sampling_rate || 16000;
  post({ type: "audio", id, audio, rate, ms, samples: audio.length, device }, [audio.buffer]);
}

self.addEventListener("message", async (e) => {
  const { type, id, modelId } = e.data;
  try {
    if (type === "load") {
      await getPipeline(modelId || DEFAULT_MODEL, true);
      post({ type: "ready", id, device });
    } else if (type === "speak") {
      await synth(id, modelId || DEFAULT_MODEL, e.data.text);
    }
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
