// MMS-TTS Russian (VITS) worker — end-to-end Russian speech synthesis off the main thread so the control
// UI never janks during synthesis. REAL inference: Meta's MMS-TTS is a VITS model — a single-stage,
// end-to-end conditional VAE with a normalizing-flow decoder and a STOCHASTIC duration predictor. There
// is no separate vocoder: Russian text → Cyrillic token ids → (flows + duration) → 16 kHz waveform, all
// in one network. Nothing here is faked.
//
// This demo is the RUSSIAN checkpoint (Xenova/mms-tts-rus) — a materially DISTINCT model from the built
// English/German/Spanish/French/Arabic/Vietnamese/Hindi and Indic MMS-TTS demos: its own native CYRILLIC
// character vocabulary (44 symbols, is_uroman:false — native Cyrillic, NOT romanised Latin, a distinct
// non-Indic, non-Latin script) and its own VITS weights, trained on Russian speech. This Xenova export
// ships the full tokenizer (tokenizer.json) + an onnx/ subfolder, so it loads with the plain
// pipeline("text-to-speech") — the real VitsTokenizer tokenises Cyrillic directly, no manual step.
//
// Because the duration predictor is stochastic, generating the same Russian text twice yields DIFFERENT
// timing/prosody — that variation is real and is what the "stochastic prosody" surface visualises.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const DEFAULT_MODEL = "Xenova/mms-tts-rus";
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
