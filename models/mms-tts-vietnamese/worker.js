// MMS-TTS Vietnamese (VITS) worker — end-to-end Vietnamese speech synthesis off the main thread so the
// control UI never janks during synthesis. REAL inference: Meta's MMS-TTS is a VITS model — a single-
// stage, end-to-end conditional VAE with a normalizing-flow decoder and a STOCHASTIC duration predictor.
// There is no separate vocoder: Vietnamese text → language-specific token ids → (flows + duration) →
// 16 kHz waveform, all in one network. Nothing here is faked.
//
// This demo is the VIETNAMESE checkpoint (Xenova/mms-tts-vie) — a materially DISTINCT model from the
// built English/German/Spanish/French/Arabic MMS-TTS demos: its own Vietnamese character vocabulary and
// its own VITS weights, trained on Vietnamese speech, so it renders Vietnamese orthography (the full set
// of tone-marked vowels, đ, etc.) that the other checkpoints cannot. The Vietnamese tokenizer vocab
// includes every tone-marked character, so tones are preserved through synthesis. The default modelId is
// the Vietnamese one; the worker still accepts any mms-tts-<iso> id to compare a second language.
//
// Because the duration predictor is stochastic, generating the same Vietnamese text twice yields
// DIFFERENT timing/prosody — that variation is real and is what the "stochastic prosody" surface shows.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const DEFAULT_MODEL = "Xenova/mms-tts-vie";
const DTYPE = "q8";
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
    dtype: DTYPE,
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
