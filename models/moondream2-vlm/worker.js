// Moondream2 worker — small vision-language model, generation off the main thread, honest WebGPU gating.
// Model: Xenova/moondream2 (image-text-to-text), WebGPU, decoder dtype q4f16.
// Canonical Transformers.js Moondream path (verified against v3.7.5):
//   Moondream1ForConditionalGeneration + AutoProcessor + AutoTokenizer + RawImage
//   -> tokenizer(`<image>\n\nQuestion: ${prompt}\n\nAnswer:`) + processor(image)
//   -> model.generate({ ...text_inputs, ...vision_inputs }) with a TextStreamer for live tokens.
// Note: the exported class is Moondream1ForConditionalGeneration (it also serves moondream2).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/moondream2";
let model = null;
let processor = null;
let tokenizer = null;
let mod = null;

function post(msg) {
  self.postMessage(msg);
}

// Real capability check — navigator.gpu existing is NOT enough; the adapter must resolve.
async function probeGPU() {
  if (!("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
  if (!adapter) return { ok: false, reason: "no-adapter" };
  const shaderF16 = adapter.features?.has?.("shader-f16") ?? false;
  return { ok: true, shaderF16 };
}

async function ensureLoaded() {
  if (model) return;
  mod = await import(TRANSFORMERS_URL);
  const { AutoProcessor, AutoTokenizer, Moondream1ForConditionalGeneration } = mod;
  console.log(`[moondream worker] loading ${MODEL_ID} on webgpu (decoder q4f16)`);
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await Moondream1ForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      decoder_model_merged: "q4f16",
    },
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[moondream worker] ready on webgpu");
  post({ type: "ready", device: "webgpu" });
}

async function run(id, imageURL, prompt, maxTokens) {
  await ensureLoaded();
  const { RawImage, TextStreamer } = mod;

  // Moondream's prompt format — image placeholder, then a Q/A frame.
  const text = `<image>\n\nQuestion: ${prompt}\n\nAnswer:`;
  post({ type: "prompt", id, template: text }); // "See inside": the exact constructed prompt.
  const text_inputs = tokenizer(text);

  const image = await RawImage.fromURL(imageURL);
  const vision_inputs = await processor(image);

  const t0 = performance.now();
  let count = 0;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (tok) => {
      count++;
      post({ type: "token", id, token: tok, t: performance.now() - t0 });
    },
  });

  await model.generate({
    ...text_inputs,
    ...vision_inputs,
    do_sample: false,
    max_new_tokens: maxTokens ?? 200,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  const promptLen = text_inputs.input_ids?.dims?.at(-1) ?? null;
  post({ type: "done", id, ms, tokens: count, promptLen });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "probe") {
      post({ type: "probe-result", gpu: await probeGPU() });
    } else if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.prompt, e.data.maxTokens);
    }
  } catch (err) {
    console.error("[moondream worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
