// PaliGemma 2 worker — task-prefixed vision-language generation off the main thread, honest WebGPU gating.
// Model: onnx-community/paligemma2-3b-pt-224 (image-text-to-text), WebGPU, decoder dtype q4f16.
// Canonical Transformers.js PaliGemma path (verified against the model card / v3.7.5):
//   PaliGemmaForConditionalGeneration + AutoProcessor + processor(image, prompt)
//   -> model.generate({ ...inputs }) with a TextStreamer for live tokens.
// PaliGemma is a base (pt) checkpoint steered by TASK PREFIXES, not a chat template:
//   "caption en" (caption), "ocr" (read text), "detect {obj}" (boxes via <locXXXX> tokens),
//   "segment {obj}" (masks via <segXXX> tokens), "answer en {question}" (VQA).
// The model card includes the "<image>" marker in the prompt string, so we prepend it here.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/paligemma2-3b-pt-224";
let processor = null;
let model = null;
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
  const { AutoProcessor, PaliGemmaForConditionalGeneration } = mod;
  console.log(`[paligemma worker] loading ${MODEL_ID} on webgpu (decoder q4f16)`);
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await PaliGemmaForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: {
      embed_tokens: "q8",
      vision_encoder: "fp16",
      decoder_model_merged: "q4f16",
    },
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[paligemma worker] ready on webgpu");
  post({ type: "ready", device: "webgpu" });
}

// `task` is the PaliGemma prefix (e.g. "caption en", "detect car"). skipSpecial=false keeps the
// <locXXXX>/<segXXX> tokens visible so detection/segmentation pages can decode them.
async function run(id, imageURL, task, maxTokens, skipSpecial) {
  await ensureLoaded();
  const { RawImage, TextStreamer } = mod;
  const image = await RawImage.read(imageURL);

  const prompt = `<image>${task}`;
  post({ type: "prompt", id, template: prompt }); // "See inside": the exact constructed prompt.

  const inputs = await processor(image, prompt);

  const t0 = performance.now();
  let count = 0;
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: skipSpecial !== false,
    callback_function: (tok) => {
      count++;
      post({ type: "token", id, token: tok, t: performance.now() - t0 });
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: maxTokens ?? 100,
    do_sample: false,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  const promptLen = inputs.input_ids?.dims?.at(-1) ?? null;
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
      await run(e.data.id, e.data.image, e.data.prompt, e.data.maxTokens, e.data.skipSpecial);
    }
  } catch (err) {
    console.error("[paligemma worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
