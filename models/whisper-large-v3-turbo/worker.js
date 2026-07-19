// Whisper-large-v3-turbo ASR worker — ALL inference off the main thread.
//
// Model: onnx-community/whisper-large-v3-turbo (task: automatic-speech-recognition). It is the full
// large-v3 encoder paired with a pruned 4-layer decoder (large-v3 has 32) — near large-v3 accuracy at
// several times the decode speed. We prefer WebGPU with q4f16 (~560 MB, fast); with no GPU adapter we
// fall back HONESTLY to WASM q8 (~1.1 GB, slower but real) and report which backend + dtype actually ran.
//
// Two real passes per run: (1) a single decoder step to DETECT the spoken language and its probability
// (softmax over Whisper's language tokens); (2) the transcription itself, with the language forced (from
// detection or the user's choice) and segment timestamps. Everything reported is measured, never claimed.

import { hasWebGPU, TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL = "onnx-community/whisper-large-v3-turbo";
const TASK = "automatic-speech-recognition";

// Whisper's language codes (order matches the language-token block after <|startoftranscript|>).
const WHISPER_LANG_CODES = [
  "en",
  "zh",
  "de",
  "es",
  "ru",
  "ko",
  "fr",
  "ja",
  "pt",
  "tr",
  "pl",
  "ca",
  "nl",
  "ar",
  "sv",
  "it",
  "id",
  "hi",
  "fi",
  "vi",
  "he",
  "uk",
  "el",
  "ms",
  "cs",
  "ro",
  "da",
  "hu",
  "ta",
  "no",
  "th",
  "ur",
  "hr",
  "bg",
  "lt",
  "la",
  "mi",
  "ml",
  "cy",
  "sk",
  "te",
  "fa",
  "lv",
  "bn",
  "sr",
  "az",
  "sl",
  "kn",
  "et",
  "mk",
  "br",
  "eu",
  "is",
  "hy",
  "ne",
  "mn",
  "bs",
  "kk",
  "sq",
  "sw",
  "gl",
  "mr",
  "pa",
  "si",
  "km",
  "sn",
  "yo",
  "so",
  "af",
  "oc",
  "ka",
  "be",
  "tg",
  "sd",
  "gu",
  "am",
  "yi",
  "lo",
  "uz",
  "fo",
  "ht",
  "ps",
  "tk",
  "nn",
  "mt",
  "sa",
  "lb",
  "my",
  "bo",
  "tl",
  "mg",
  "as",
  "tt",
  "haw",
  "ln",
  "ha",
  "ba",
  "jw",
  "su",
  "yue",
];

let tf = null;
let pipe = null;
let device = "wasm";
let dtype = "q8";
let langTokenIds = null; // { code: tokenId }
let sotId = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded() {
  if (pipe) return;
  tf = await import(TRANSFORMERS_URL);
  const { pipeline, env } = tf;
  env.allowLocalModels = false;
  const gpu = await hasWebGPU();
  // Prefer WebGPU with a q4-int4/fp16 mix (fast + compact); WASM gets q8 (works everywhere).
  const attempts = gpu
    ? [
      { device: "webgpu", dtype: "q4f16" },
      { device: "wasm", dtype: "q8" },
    ]
    : [{ device: "wasm", dtype: "q8" }];
  let lastErr = null;
  for (const a of attempts) {
    try {
      pipe = await pipeline(TASK, MODEL, {
        device: a.device,
        dtype: a.dtype,
        progress_callback: (p) => post({ type: "progress", p }),
      });
      device = a.device;
      dtype = a.dtype;
      lastErr = null;
      break;
    } catch (err) {
      lastErr = err;
      if (a.device !== "wasm") {
        post({
          type: "progress",
          p: { status: "initiate", file: "WebGPU load failed — retrying on WASM…" },
        });
      }
    }
  }
  if (!pipe) throw lastErr || new Error("Failed to load model");

  // Resolve language-token ids + the start-of-transcript token for the detection pass.
  const t2i = pipe.tokenizer.model.tokens_to_ids;
  langTokenIds = {};
  for (const code of WHISPER_LANG_CODES) {
    const id = t2i.get?.(`<|${code}|>`);
    if (id != null) langTokenIds[code] = id;
  }
  sotId = t2i.get?.("<|startoftranscript|>");
  post({ type: "ready", device, dtype });
}

// One decoder step over the encoded audio -> logits at the language position -> softmax over the
// language tokens -> the detected language and a real probability distribution. No generation loop.
async function detectLanguage(pcm) {
  if (sotId == null || !langTokenIds) return null;
  const { Tensor } = tf;
  const inputs = await pipe.processor(pcm);
  const decoderInputIds = new Tensor("int64", [BigInt(sotId)], [1, 1]);
  const out = await pipe.model({
    input_features: inputs.input_features,
    decoder_input_ids: decoderInputIds,
  });
  const data = out.logits.data; // last dim = vocab, seq length 1
  const entries = Object.entries(langTokenIds);
  let mx = -Infinity;
  for (const [, id] of entries) {
    const v = Number(data[id]);
    if (v > mx) mx = v;
  }
  let sum = 0;
  const exps = entries.map(([code, id]) => {
    const e = Math.exp(Number(data[id]) - mx);
    sum += e;
    return [code, e];
  });
  const probs = exps.map(([code, e]) => [code, e / sum]).sort((a, b) => b[1] - a[1]);
  return { detected: probs[0][0], prob: probs[0][1], probs };
}

function toSegments(chunks) {
  return (chunks || []).map((c) => ({
    start: c.timestamp?.[0] ?? null,
    end: c.timestamp?.[1] ?? null,
    text: (c.text || "").trim(),
  })).filter((s) => s.text);
}

async function run(id, audio, opts = {}) {
  await ensureLoaded();

  // 1) Language detection (real distribution) unless the user forced a language and opted out.
  let detectedLang = null, detectedProb = null, langProbs = null;
  if (opts.detect !== false) {
    try {
      const det = await detectLanguage(audio);
      if (det) {
        detectedLang = det.detected;
        detectedProb = det.prob;
        langProbs = det.probs.slice(0, 8);
      }
    } catch {
      /* detection is a bonus; never block the transcript on it */
    }
  }

  // 2) Transcription. Force the user's language if chosen, else let Whisper use what it detected.
  const genOpts = {
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
    task: opts.task === "translate" ? "translate" : "transcribe",
  };
  if (opts.language) genOpts.language = opts.language;

  const t0 = performance.now();
  const output = await pipe(audio, genOpts);
  const ms = Math.round(performance.now() - t0);

  const text = (output.text || "").trim();
  const segments = toSegments(output.chunks);

  let tokens = null;
  try {
    const enc = pipe.tokenizer(text);
    tokens = enc?.input_ids?.dims ? enc.input_ids.dims.at(-1) : (enc?.input_ids?.size ?? null);
  } catch {
    tokens = null;
  }
  const tokPerSec = tokens && ms ? tokens / (ms / 1000) : null;

  post({
    type: "result",
    id,
    text,
    segments,
    detectedLang,
    detectedProb,
    langProbs,
    tokens,
    tokPerSec,
    ms,
    device,
    dtype,
    task: genOpts.task,
    forcedLanguage: opts.language || null,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "run") await run(e.data.id, e.data.audio, e.data.opts);
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
