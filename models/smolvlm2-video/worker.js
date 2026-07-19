// SmolVLM2 worker — multi-image / video-frame vision-language generation off the main thread, with
// honest WebGPU gating. Model: HuggingFaceTB/SmolVLM2-256M-Video-Instruct (image-text-to-text),
// WebGPU, dtype q4f16. Distinct from SmolVLM v1 (idefics3): SmolVLM2 (model_type "smolvlm",
// SmolVLMForConditionalGeneration) natively accepts MULTIPLE images/frames in one turn, so it can
// compare stills and reason across a sequence of video frames.
//
// Uses the canonical Transformers.js VLM path: AutoProcessor + AutoModelForVision2Seq +
// apply_chat_template + TextStreamer. Shared 3.7.5 pin (SmolVLMForConditionalGeneration +
// SmolVLMProcessor ship in 3.7.5 — verified) — no version-pin escape hatch needed.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct";
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
  const { AutoProcessor, AutoModelForVision2Seq } = mod;
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForVision2Seq.from_pretrained(MODEL_ID, {
    dtype: "q4f16",
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "webgpu" });
}

// Load one or more images (data URLs) into RawImage objects for the processor.
async function loadImages(urls) {
  const { load_image } = mod;
  return Promise.all(urls.map((u) => load_image(u)));
}

async function run(id, imageURLs, prompt, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const images = await loadImages(imageURLs);

  // One {type:"image"} per frame, then the text — this is the multi-image turn shape.
  const content = images.map(() => ({ type: "image" }));
  content.push({ type: "text", text: prompt });
  const messages = [{ role: "user", content }];

  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  post({ type: "prompt", id, template: text, imageCount: images.length });

  const inputs = await processor(text, images, { do_image_splitting: false });
  // input_ids length = the full templated token count (text + expanded image tokens). pixel_values
  // dims = [batch, numImages, channels, H, W] — surface both so "See inside" can show the real shape.
  const promptTokens = inputs.input_ids?.dims?.at(-1) ?? null;
  const pvDims = inputs.pixel_values?.dims ? Array.from(inputs.pixel_values.dims) : null;
  post({ type: "shape", id, promptTokens, pixelValues: pvDims });

  const t0 = performance.now();
  let count = 0;
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (tok) => {
      count++;
      post({ type: "token", id, token: tok, t: performance.now() - t0 });
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: maxTokens ?? 200,
    do_sample: false,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  post({ type: "done", id, ms, tokens: count, promptTokens, imageCount: images.length });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "probe") {
      post({ type: "probe-result", gpu: await probeGPU() });
    } else if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.images, e.data.prompt, e.data.maxTokens);
    }
  } catch (err) {
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
