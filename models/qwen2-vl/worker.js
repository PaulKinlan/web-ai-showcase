// Qwen2-VL worker — vision-language generation off the main thread, with honest WebGPU gating.
// Model: onnx-community/Qwen2-VL-2B-Instruct (image-text-to-text), WebGPU, decoder dtype q4f16.
// Canonical Transformers.js Qwen2-VL path (verified against the model card / v3.7.5):
//   Qwen2VLForConditionalGeneration + AutoProcessor + apply_chat_template + processor(text, image)
//   -> model.generate({ ...inputs }) with a TextStreamer for live tokens.
// The processor resizes the image (we clamp to 448px) so the number of vision tokens stays sane.
// Supports multi-turn: the image is attached to the FIRST user turn only; later turns are text.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Qwen2-VL-2B-Instruct";
const MAX_SIDE = 448; // keep the vision-token budget (and memory) in check
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
  const { AutoProcessor, Qwen2VLForConditionalGeneration } = mod;
  console.log(`[qwen2-vl worker] loading ${MODEL_ID} on webgpu (decoder q4f16)`);
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      decoder_model_merged: "q4f16",
    },
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[qwen2-vl worker] ready on webgpu");
  post({ type: "ready", device: "webgpu" });
}

// Build a Qwen2-VL conversation. `history` is prior turns [{role,text}] (image lives on the first
// user turn only); `prompt` is the current user question.
function buildConversation(history, prompt) {
  const conv = [];
  let imgAttached = false;
  for (const turn of history ?? []) {
    if (turn.role === "user" && !imgAttached) {
      conv.push({ role: "user", content: [{ type: "image" }, { type: "text", text: turn.text }] });
      imgAttached = true;
    } else {
      conv.push({ role: turn.role, content: [{ type: "text", text: turn.text }] });
    }
  }
  if (!imgAttached) {
    conv.push({ role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] });
  } else {
    conv.push({ role: "user", content: [{ type: "text", text: prompt }] });
  }
  return conv;
}

async function run(id, imageURL, prompt, maxTokens, history) {
  await ensureLoaded();
  const { TextStreamer, RawImage } = mod;
  let image = await RawImage.read(imageURL);
  // Clamp the longest side to MAX_SIDE to bound vision tokens + memory (aspect kept roughly square).
  const side = Math.min(MAX_SIDE, Math.max(image.width, image.height));
  image = await image.resize(side, side);

  const conversation = buildConversation(history, prompt);
  const text = processor.apply_chat_template(conversation, { add_generation_prompt: true });
  post({ type: "prompt", id, template: text }); // "See inside": the constructed chat template.

  const inputs = await processor(text, image);

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
      await run(e.data.id, e.data.image, e.data.prompt, e.data.maxTokens, e.data.history);
    }
  } catch (err) {
    console.error("[qwen2-vl worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
