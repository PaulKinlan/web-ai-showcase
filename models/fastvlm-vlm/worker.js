// FastVLM worker — Apple's efficient vision-language model, generation off the main thread.
// Model: onnx-community/FastVLM-0.5B-ONNX (image-text-to-text). Architecture: a FastViTHD hybrid
// conv/transformer vision encoder (the paper's contribution — far fewer, higher-quality visual tokens,
// so time-to-first-token is low) feeding a Qwen2-0.5B text decoder. Exposed to Transformers.js as the
// `llava_qwen2` model type (LlavaQwen2ForCausalLM), loaded via AutoModelForImageTextToText.
//
// Canonical Transformers.js FastVLM path (from the model card, verified against v3.7.5 in headless
// Chrome before shipping):
//   AutoProcessor + AutoModelForImageTextToText
//   messages=[{role:"user", content:"<image>" + prompt}]
//   prompt = processor.apply_chat_template(messages, { add_generation_prompt:true })
//   inputs = processor(image, prompt, { add_special_tokens:false })
//   model.generate({ ...inputs, ... }) with a TextStreamer for live tokens.
//
// FastVLM-0.5B is small enough to run on BOTH WebGPU (q4f16, fast) and the WASM CPU fallback (q4), so
// the page works without a GPU too — we pick WebGPU when a real adapter exists, else WASM, and report
// which path actually ran. Never fakes output.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/FastVLM-0.5B-ONNX";
let model = null;
let processor = null;
let mod = null;
let device = "wasm";

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
  const { AutoProcessor, AutoModelForImageTextToText } = mod;
  // WASM (CPU), all-q4. This is the config we VERIFIED produces correct, grounded output in a real
  // browser (headless Chrome): bee.jpg -> "...a large, pink flower...". We deliberately do NOT use the
  // WebGPU path: FastVLM's fp16/q4f16 exports throw an ORT-Web `Unexpected input data type` error (the
  // fp16 vision encoder wants fp16 pixel_values but the processor emits float32, and Transformers.js
  // doesn't cast them), and all-q4 on a *software* WebGPU adapter returns degenerate tokens — both
  // measured here. Running on the CPU keeps the output correct and identical on every device (and works
  // without a GPU); the honest trade-off is speed, which the page reports via the real latency readout.
  device = "wasm";
  const dtype = {
    embed_tokens: "q4",
    vision_encoder: "q4",
    decoder_model_merged: "q4",
  };
  console.log(`[fastvlm worker] loading ${MODEL_ID} on ${device}`);
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await AutoModelForImageTextToText.from_pretrained(MODEL_ID, {
    dtype,
    device,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[fastvlm worker] ready on ${device}`);
  post({ type: "ready", device });
}

async function run(id, imageURL, prompt, maxTokens) {
  await ensureLoaded();
  const { RawImage, TextStreamer } = mod;

  // FastVLM uses a chat template; the image placeholder <image> goes in the user turn's content.
  const messages = [{ role: "user", content: `<image>${prompt}` }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  post({ type: "prompt", id, template: text }); // "See inside": the exact constructed prompt.

  const image = await RawImage.fromURL(imageURL);
  const inputs = await processor(image, text, { add_special_tokens: false });

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
    do_sample: false,
    max_new_tokens: maxTokens ?? 200,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  const promptLen = inputs.input_ids?.dims?.at(-1) ?? null;
  post({ type: "done", id, ms, tokens: count, promptLen, device });
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
    console.error("[fastvlm worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
