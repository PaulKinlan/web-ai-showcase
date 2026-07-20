// Speaker-diarization worker — pyannote segmentation off the main thread.
// Model: onnx-community/pyannote-segmentation-3.0 (audio-frame-classification), WASM, fp32.
// Architecture: pyannote/segmentation-3.0 — a SincNet + LSTM segmentation model exported to ONNX and
// exposed to Transformers.js as `AutoModelForAudioFrameClassification`. For a sliding audio window it
// emits, per frame, logits over a 7-class "powerset" of up to 3 concurrent speakers (silence, spk A,
// spk B, spk C, A+B, A+C, B+C). The processor's `post_process_speaker_diarization` decodes those
// powerset frames into speaker-labelled segments with start/end times and a confidence — i.e. "who
// spoke when", including overlapped speech. This is DISTINCT from the built silero-VAD (which only
// answers speech-vs-silence): diarization assigns *identities* to the speech.
//
// The page decodes audio to a 16 kHz mono Float32Array (Web Audio) and transfers it here; we run the
// real model and return the decoded segments + per-frame activity for the "see inside" surface. Small
// (~6 MB) and CPU-only — no GPU required. Verified in headless Chrome before shipping (real segments,
// distinct speaker ids, timestamps).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/pyannote-segmentation-3.0";
let model = null;
let processor = null;
let mod = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (model) return;
  mod = await import(TRANSFORMERS_URL);
  const { AutoProcessor, AutoModelForAudioFrameClassification } = mod;
  console.log(`[diarization worker] loading ${MODEL_ID} (wasm, fp32)`);
  model = await AutoModelForAudioFrameClassification.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[diarization worker] ready");
  post({ type: "ready", device: "wasm" });
}

// Reduce the per-frame powerset logits to a small activity summary for the "see inside" surface:
// for each frame, the argmax powerset class and whether >1 speaker is active (overlap).
function frameActivity(logits) {
  const [, frames, C] = logits.dims;
  const data = logits.data;
  // Powerset class → number of active speakers (class 0 = silence). For 3 speakers the 7 classes are
  // {}, {A}, {B}, {C}, {A,B}, {A,C}, {B,C}; count of active speakers per class:
  const activeCount = [0, 1, 1, 1, 2, 2, 2];
  const out = new Array(frames);
  let speech = 0, overlap = 0;
  for (let f = 0; f < frames; f++) {
    let mi = 0;
    for (let c = 1; c < C; c++) if (data[f * C + c] > data[f * C + mi]) mi = c;
    const n = activeCount[mi] ?? 0;
    out[f] = n;
    if (n >= 1) speech++;
    if (n >= 2) overlap++;
  }
  return { activity: out, frames, speechFrames: speech, overlapFrames: overlap };
}

async function run(id, audio, sampleRate) {
  await ensureLoaded();
  const t0 = performance.now();
  const inputs = await processor(audio);
  const { logits } = await model(inputs);
  const result = processor.post_process_speaker_diarization(logits, audio.length);
  const segments = (result[0] || []).map((s) => ({
    id: s.id,
    start: s.start,
    end: s.end,
    confidence: s.confidence,
  }));
  const act = frameActivity(logits);
  const ms = Math.round(performance.now() - t0);
  const durationSec = audio.length / (sampleRate || 16000);
  const speakerIds = [...new Set(segments.map((s) => s.id))].sort((a, b) => a - b);
  post({
    type: "result",
    id,
    segments,
    speakerIds,
    durationSec,
    frames: act.frames,
    activity: act.activity,
    speechFrames: act.speechFrames,
    overlapFrames: act.overlapFrames,
    ms,
    device: "wasm",
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.audio, e.data.sampleRate);
  } catch (err) {
    console.error("[diarization worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
