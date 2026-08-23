// Ultravox worker — a NATIVE audio-in language model, off the main thread.
//
// Model: onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX (task: audio-text-to-text).
//
// The point of this demo, and the thing that makes it different from an ASR pipeline: the raw 16 kHz
// PCM never becomes text. UltravoxProcessor turns it into `audio_values`; UltravoxModel's
// _merge_input_ids_with_audio_features splices the resulting audio embeddings into the token stream
// at the <|audio|> positions, and the Llama-3.2 decoder attends to THOSE. There is no transcript
// anywhere in this file — nothing to grep for, because nothing produces one. Tone, hesitation and
// overlapping speech reach the model instead of being flattened away by a speech-to-text stage.
//
// Transformers.js 3.7.5 (the shared pin) already registers `ultravox` in
// MODEL_FOR_AUDIO_TEXT_TO_TEXT_MAPPING_NAMES, so no version escape hatch is needed. There is no
// `audio-text-to-text` PIPELINE task in 3.7.5, so we drive the processor and model directly — the
// exact call shape published on the model card.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/ultravox-v0_5-llama-3_2-1b-ONNX";

// The per-module dtypes the model card documents as a working combination. Stated sizes are the real
// download: embed_tokens q8 263 MB + audio_encoder q4 454 MB + decoder_model_merged q4 806 MB.
const DTYPE = {
  embed_tokens: "q8",
  audio_encoder: "q4",
  decoder_model_merged: "q4",
};

let mod = null;
let processor = null;
let model = null;
let device = null;

function post(msg) {
  self.postMessage(msg);
}

// navigator.gpu existing is NOT enough — headless and locked-down browsers expose the object but
// return no adapter. Ask for one so the page can show an honest unsupported state.
async function probeGPU() {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { ok: false, reason: "no-adapter" };
    return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
}

async function ensureLoaded() {
  if (model) return;
  const gpu = await probeGPU();
  if (!gpu.ok) {
    // ~1.5 GB of 4-bit weights through the WASM EP is not a slow path, it is an unusable one, and a
    // WASM fallback here would exceed the wasm32 4 GB address space during session creation. Refuse
    // honestly rather than start a download that cannot finish.
    throw new Error(
      "Ultravox needs WebGPU — this device reports no usable GPU adapter, and ~1.5 GB of 4-bit " +
        "weights cannot run on the WebAssembly backend.",
    );
  }
  mod = await import(TRANSFORMERS_URL);
  const { UltravoxProcessor, UltravoxModel } = mod;
  console.log(`[ultravox worker] loading ${MODEL_ID} on webgpu`, DTYPE);
  processor = await UltravoxProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await UltravoxModel.from_pretrained(MODEL_ID, {
    device: "webgpu",
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  device = "webgpu";
  console.log("[ultravox worker] ready");
  post({ type: "ready", device, modelId: MODEL_ID, dtype: DTYPE });
}

/**
 * Build the templated prompt. `messages` carries the audio placeholder inside the user turn; the
 * Llama-3.2 template serialises `tools` into the first user message and documents the
 * {"name", "parameters"} reply format.
 */
function template(messages, tools) {
  return processor.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
    ...(tools?.length ? { tools } : {}),
  });
}

/**
 * One generation. `audio` is a 16 kHz mono Float32Array (or null for a follow-up turn that carries
 * no new audio). Returns the decoded continuation plus the numbers that prove audio really entered
 * the context: how many embedding frames it occupied and how long the prompt became.
 */
async function generate(id, { messages, tools, audio, maxTokens }) {
  await ensureLoaded();
  const text = template(messages, tools);
  post({ type: "prompt", id, template: text });

  const t0 = performance.now();
  // processor(text, audio) → { input_ids, attention_mask, audio_values, audio_token_len }. The
  // placeholder is expanded to one <|audio|> per embedding frame BEFORE tokenisation, so the audio
  // occupies real positions in the sequence.
  const inputs = await processor(text, audio ?? null);
  const prepMs = Math.round(performance.now() - t0);

  const promptLen = inputs.input_ids.dims.at(-1);
  const audioFrames = inputs.audio_token_len ? Number(inputs.audio_token_len[0]) : 0;

  const t1 = performance.now();
  // Stream. A 1B model producing up to 192 tokens on WebGPU is several seconds of silence
  // otherwise, and a stage label that never changes is indistinguishable from a hang. The final
  // decode below is still what gets parsed — the stream is purely for the visible progress.
  let streamed = 0;
  const streamer = mod.TextStreamer
    ? new mod.TextStreamer(processor.tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false,
      callback_function: (token) => {
        streamed++;
        post({ type: "token", id, token, n: streamed, t: performance.now() - t1 });
      },
    })
    : undefined;
  const outputIds = await model.generate({
    ...inputs,
    max_new_tokens: Math.max(1, Math.min(512, maxTokens ?? 192)),
    do_sample: false, // greedy — sampling makes a 1B model mangle the call JSON
    ...(streamer ? { streamer } : {}),
  });
  const genMs = Math.round(performance.now() - t1);

  const decoded = processor.batch_decode(
    outputIds.slice(null, [promptLen, null]),
    { skip_special_tokens: false },
  );
  const raw = decoded[0] ?? "";
  const newTokens = outputIds.dims.at(-1) - promptLen;

  post({
    type: "result",
    id,
    text: raw,
    prepMs,
    genMs,
    promptTokens: promptLen,
    audioFrames,
    newTokens,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "probe") {
      post({ type: "probe-result", gpu: await probeGPU() });
    } else if (d.type === "load") {
      await ensureLoaded();
    } else if (d.type === "generate") {
      await generate(d.id, d);
    }
  } catch (err) {
    console.error("[ultravox worker] error", err);
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
