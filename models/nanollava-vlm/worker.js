// nanoLLaVA worker — a ~1B LLaVA-style vision-language model off the main thread, honest WebGPU gating.
// Model: Xenova/nanoLLaVA (image-text-to-text), WebGPU, decoder dtype q4f16.
// Canonical Transformers.js LLaVA path (verified against v3.7.5):
//   LlavaForConditionalGeneration + AutoProcessor + AutoTokenizer + RawImage
//   -> tokenizer.apply_chat_template([{system},{user:`<image>\n${prompt}`}]) -> tokenizer(text)
//   -> processor(image) -> model.generate({ ...text_inputs, ...vision_inputs }) with a TextStreamer.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

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
  const { AutoProcessor, AutoTokenizer, LlavaForConditionalGeneration } = mod;
  console.log(`[nanollava worker] loading Xenova/nanoLLaVA on webgpu (decoder q4f16)`);
  tokenizer = await AutoTokenizer.from_pretrained("Xenova/nanoLLaVA", {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  processor = await AutoProcessor.from_pretrained("Xenova/nanoLLaVA", {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await LlavaForConditionalGeneration.from_pretrained("Xenova/nanoLLaVA", {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      decoder_model_merged: "q4f16",
    },
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[nanollava worker] ready on webgpu");
  post({ type: "ready", device: "webgpu" });
}

async function run(id, imageURL, prompt, system, maxTokens) {
  await ensureLoaded();
  const { RawImage, Tensor, TextStreamer } = mod;

  // LLaVA-style ChatML (the Xenova/nanoLLaVA card's shape): the user turn carries the `<image>`
  // placeholder as TEXT and the tokenizer's chat template builds the ChatML prompt.
  const messages = [
    { role: "system", content: system && system.trim() ? system.trim() : "Answer the question." },
    { role: "user", content: `<image>\n${prompt}` },
  ];
  const text = tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });
  post({ type: "prompt", id, template: text }); // "See inside": the templated ChatML prompt.

  const image = await RawImage.fromURL(imageURL);
  const vision_inputs = await processor(image);
  const text_inputs = tokenizer(text);

  // transformers.js 3.7.5's Llava processor carries NO tokenizer, so its joint text+image call
  // cannot run and the model's merge step sees ONE placeholder token against the vision tower's
  // 729 features ("Number of tokens and features do not match"), found by driving this route.
  // Expand the placeholder HERE, at the count the vision tower will produce — the patch grid is
  // floor(image_size / patch_size) squared — exactly as default_merge_input_ids_with_features
  // expects. `tolist()` yields BigInts for int64 tensors, so the id comparison is numeric.
  const vision = model.config.vision_config ?? {};
  const imageTokenCount = Math.floor((vision.image_size ?? 384) / (vision.patch_size ?? 14)) ** 2;
  const imageTokenId = Number(model.config.image_token_index);
  const ids = text_inputs.input_ids.tolist()[0];
  const mask = text_inputs.attention_mask.tolist()[0];
  const expandedIds = [];
  const expandedMask = [];
  for (let i = 0; i < ids.length; i++) {
    if (Number(ids[i]) === imageTokenId) {
      for (let k = 0; k < imageTokenCount; k++) {
        expandedIds.push(imageTokenId);
        expandedMask.push(1);
      }
    } else {
      expandedIds.push(Number(ids[i]));
      expandedMask.push(Number(mask[i]));
    }
  }
  const inputs = {
    input_ids: new Tensor("int64", BigInt64Array.from(expandedIds.map(BigInt)), [
      1,
      expandedIds.length,
    ]),
    attention_mask: new Tensor("int64", BigInt64Array.from(expandedMask.map(BigInt)), [
      1,
      expandedMask.length,
    ]),
    pixel_values: vision_inputs.pixel_values,
  };

  const t0 = performance.now();
  let count = 0;
  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    // TextStreamer buffers decoded words, so callback_function fires per VISIBLE CHUNK, not per
    // generated token. Count the generated IDs instead (prompt excluded, special generated IDs
    // included), then attach that count to each visible text chunk (web-ai-showcase-0ly; db2).
    token_callback_function: (ids) => {
      count += ids.length;
    },
    callback_function: (tok) => {
      post({ type: "token", id, token: tok, tokens: count, t: performance.now() - t0 });
    },
  });

  await model.generate({
    ...inputs,
    do_sample: false,
    max_new_tokens: maxTokens ?? 200,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  const promptLen = inputs.input_ids.dims?.at(-1) ?? null;
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
      await run(e.data.id, e.data.image, e.data.prompt, e.data.system, e.data.maxTokens);
    }
  } catch (err) {
    console.error("[nanollava worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
