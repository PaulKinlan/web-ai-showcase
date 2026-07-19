// MMS-TTS (VITS) worker — end-to-end speech synthesis off the main thread so the control UI never
// janks during synthesis. This is REAL inference: Meta's MMS-TTS is a VITS model — a single-stage,
// end-to-end conditional VAE with a normalizing-flow decoder and a STOCHASTIC duration predictor.
// There is no separate vocoder: text → phoneme-ish tokens → (flows + duration) → 16 kHz waveform,
// all in one network. Nothing here is faked.
//
// One MMS checkpoint = one language (facebook/mms-tts covers 1000+). We keep a small per-language
// pipeline cache so the multilingual page can switch languages without reloading the ones it has.
//
// Because the duration predictor is stochastic, generating the same text twice yields DIFFERENT
// timing/prosody — that variation is real and is what the "stochastic prosody" surface visualises.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

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
      await getPipeline(modelId || "Xenova/mms-tts-eng", true);
      post({ type: "ready", id, device });
    } else if (type === "speak") {
      await synth(id, modelId || "Xenova/mms-tts-eng", e.data.text);
    }
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
