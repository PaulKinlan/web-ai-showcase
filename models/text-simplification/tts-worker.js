// Text-to-speech worker for the text-simplification multi-model demo — inference off the main thread.
// Model: Xenova/mms-tts-eng (task: text-to-speech), WASM, q8. Meta's MMS-TTS is a VITS model: a single
// end-to-end network (no separate vocoder) that turns text → a 16 kHz waveform. Here it is the SECOND
// stage: after the simplifier rewrites dense text into plain language, MMS-TTS reads the plain version
// aloud — the accessibility win (simpler words AND spoken). Real inference; nothing is faked.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

let tts = null;
let device = "wasm";

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded() {
  if (tts) return;
  const { pipeline, env } = await import(TRANSFORMERS_URL);
  env.allowLocalModels = false;
  tts = await pipeline("text-to-speech", "Xenova/mms-tts-eng", {
    device: "wasm",
    dtype: "q8",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device });
}

async function speak(id, text) {
  await ensureLoaded();
  const t0 = performance.now();
  const out = await tts(text);
  const ms = Math.round(performance.now() - t0);
  const audio = out.audio instanceof Float32Array ? out.audio : Float32Array.from(out.audio);
  const rate = out.sampling_rate || 16000;
  post({ type: "audio", id, audio, rate, ms, samples: audio.length, device }, [audio.buffer]);
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await speak(id, e.data.text);
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
