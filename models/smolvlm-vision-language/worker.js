// SmolVLM worker — vision-language generation off the main thread, with honest WebGPU gating.
// Model: HuggingFaceTB/SmolVLM-256M-Instruct (image-text-to-text), WebGPU, dtype q4f16.
// Uses the canonical Transformers.js VLM path: AutoProcessor + AutoModelForVision2Seq +
// apply_chat_template + TextStreamer. We import the shared TRANSFORMERS_URL from lib/webai.js.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct";
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

async function run(id, imageURL, prompt, maxTokens) {
  await ensureLoaded();
  const { TextStreamer, load_image } = mod;
  const image = await load_image(imageURL);

  const messages = [
    { role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] },
  ];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  post({ type: "prompt", id, template: text }); // for "See inside": the constructed chat template

  const inputs = await processor(text, [image], { do_image_splitting: false });

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
  const imageTokens = inputs.input_ids?.dims?.at(-1) ?? null;
  post({ type: "done", id, ms, tokens: count, promptLen: imageTokens });
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
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
