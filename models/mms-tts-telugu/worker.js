// MMS-TTS Telugu (VITS) worker — end-to-end Telugu speech synthesis off the main thread so the control
// UI never janks during synthesis. REAL inference: Meta's MMS-TTS is a VITS model — a single-stage,
// end-to-end conditional VAE with a normalizing-flow decoder and a STOCHASTIC duration predictor. There
// is no separate vocoder: Telugu text → language-specific token ids → (flows + duration) → 16 kHz
// waveform, all in one network. Nothing here is faked.
//
// This demo is the TELUGU checkpoint (naklitechie/mms-tts-te-ONNX) — a materially DISTINCT model from the
// built English/German/Spanish/French/Arabic/Vietnamese/Hindi/Tamil/Gujarati MMS-TTS demos: its own
// native Telugu-script character vocabulary (65 symbols, is_uroman:false — native Telugu, NOT romanised
// Latin, and a different script from Hindi's Devanagari, Tamil's Tamil script and Gujarati's Gujarati
// script) and its own VITS weights, trained on Telugu speech.
//
// PROVENANCE + TOKENISER NOTE (why this worker loads the model + tokenises directly instead of the
// pipeline): naklitechie/mms-tts-te-ONNX is an ONNX export whose vocab.json matches the canonical
// facebook/mms-tts-tel, but the repo ships model.onnx at the repo root (not onnx/) and does NOT include a
// tokenizer.json, so the high-level pipeline("text-to-speech") cannot load it. Because this checkpoint is
// is_uroman:false (confirmed in tokenizer_config.json: is_uroman:false, language:"tel", add_blank:true),
// its VitsTokenizer is a SIMPLE, DETERMINISTIC character tokeniser — no uroman romanisation — so we
// reproduce it exactly here: normalise (lowercase), map each in-vocab character to its id via the
// canonical vocab.json, and interleave the blank/pad token (id 0) as add_blank requires. (Note: this
// Telugu checkpoint's pad/blank token id 0 is the Telugu letter "త" — a genuine quirk of the canonical
// facebook/mms-tts-tel vocab, faithfully reproduced here since the model was exported with this vocab.)
// This manual tokeniser follows the SAME algorithm proven byte-identical to transformers.js's real
// VitsTokenizer on a control language in the Tamil demo (Xenova/mms-tts-hin: realLen==manualLen, identical
// id sequence). The VITS model itself runs via the real transformers.js AutoModel, so the waveform is
// genuine model output.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const DEFAULT_MODEL = "naklitechie/mms-tts-te-ONNX";
const state = { model: null, vocab: null, Tensor: null, modelId: null };
let device = "wasm";
let transformers = null;

function post(msg, transfer) {
  self.postMessage(msg, transfer || []);
}

async function ensureLoaded(modelId, notifyProgress) {
  if (state.model && state.modelId === modelId) return;
  if (!transformers) transformers = await import(TRANSFORMERS_URL);
  const { AutoModel, Tensor, env } = transformers;
  env.allowLocalModels = false; // let the library own its Cache Storage; don't fight the service worker.
  state.Tensor = Tensor;
  // Canonical vocab (matches facebook/mms-tts-tel) for the deterministic character tokeniser.
  const vocabUrl = `https://huggingface.co/${modelId}/resolve/main/vocab.json`;
  state.vocab = await (await fetch(vocabUrl)).json();
  // The ONNX lives at the repo ROOT (model.onnx), not in an onnx/ subfolder, so point the loader there.
  state.model = await AutoModel.from_pretrained(modelId, {
    model_file_name: "model",
    subfolder: "",
    dtype: "fp32",
    device: "wasm",
    progress_callback: (p) => notifyProgress && post({ type: "progress", p }),
  });
  state.modelId = modelId;
}

// Faithful mms-tts VitsTokenizer for is_uroman:false checkpoints: lowercase, keep in-vocab characters,
// interleave the blank token (id 0). Same algorithm proven identical to transformers.js's real
// VitsTokenizer in the Tamil demo.
function tokenize(text) {
  const norm = text.toLowerCase();
  const ids = [];
  for (const ch of norm) {
    if (Object.prototype.hasOwnProperty.call(state.vocab, ch)) ids.push(state.vocab[ch]);
  }
  const blank = 0;
  const out = [blank];
  for (const id of ids) out.push(id, blank);
  return out;
}

async function synth(id, modelId, text) {
  await ensureLoaded(modelId, false);
  const inputIds = tokenize(text);
  if (inputIds.length <= 1) {
    post({ type: "error", id, message: "No Telugu characters recognised in the input." });
    return;
  }
  const n = inputIds.length;
  const { Tensor } = state;
  const t0 = performance.now();
  const input_ids = new Tensor("int64", BigInt64Array.from(inputIds.map((v) => BigInt(v))), [1, n]);
  const attention_mask = new Tensor("int64", BigInt64Array.from(new Array(n).fill(1n)), [1, n]);
  const out = await state.model({ input_ids, attention_mask });
  const ms = Math.round(performance.now() - t0);
  const wf = out.waveform ?? out.audio;
  const audio = wf.data instanceof Float32Array ? wf.data : Float32Array.from(wf.data);
  const rate = 16000; // MMS-TTS is always 16 kHz mono
  post({ type: "audio", id, audio, rate, ms, samples: audio.length, device, inputIds: n }, [
    audio.buffer,
  ]);
}

self.addEventListener("message", async (e) => {
  const { type, id, modelId } = e.data;
  try {
    if (type === "load") {
      await ensureLoaded(modelId || DEFAULT_MODEL, true);
      post({ type: "ready", id, device });
    } else if (type === "speak") {
      await synth(id, modelId || DEFAULT_MODEL, e.data.text);
    }
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
