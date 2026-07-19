// Phi-3.5-vision worker — multi-image vision-language generation off the main thread, honest WebGPU gate.
//
// Model:   onnx-community/Phi-3.5-vision-instruct  (task: image-text-to-text)
// Classes: Phi3VForCausalLM + AutoProcessor (Phi3VProcessor)  — present in @huggingface/transformers 3.7.5
//          (model_type "phi3_v" → Phi3VForCausalLM in the causal-LM mapping; AutoProcessor → Phi3VProcessor).
// dtype:   { vision_encoder, prepare_inputs_embeds, model } all q4f16 (needs shader-f16) — ~2.5 GB total.
// Backend: WebGPU (required — a GPU adapter + shader-f16).
//
// Canonical Transformers.js Phi-3.5-vision path (verified against the CDN bundle + model card / v3.7.5):
//   - The repo splits into three ONNX sessions: vision_encoder, prepare_inputs_embeds, model — matching
//     the dtype keys below.
//   - Prompt uses <|image_N|> placeholders; the Phi3VProcessor expands each to <|image|> repeated
//     num_img_tokens times, tokenizes, and NEGATES the image-token ids so the model splices in vision
//     features. We count those negative ids to report the true image-token count ("see inside").
//   - apply_chat_template wraps the turn as <|user|>\n…<|end|>\n<|assistant|>\n, then processor(text,
//     images) → model.generate({ ...inputs }) with a TextStreamer for live tokens.
//   - MULTI-IMAGE: the Phi3VProcessor reuses the first image's token count for every <|image_N|> tag in
//     one text, so all images in a single turn must share dimensions. We resize every image in a
//     multi-image turn to the SAME square so their num_img_tokens match (single-image keeps aspect).

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Phi-3.5-vision-instruct";
const MAX_SIDE = 1344; // bound vision tokens/memory for a single image (4 × 336 crop base)
const MULTI_SIDE = 756; // common square for multi-image turns so per-image token counts match
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
  const { AutoProcessor, Phi3VForCausalLM } = mod;
  console.log(`[phi-3.5-vision worker] loading ${MODEL_ID} on webgpu (all sessions q4f16)`);
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await Phi3VForCausalLM.from_pretrained(MODEL_ID, {
    dtype: {
      vision_encoder: "q4f16",
      prepare_inputs_embeds: "q4f16",
      model: "q4f16",
    },
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[phi-3.5-vision worker] ready on webgpu");
  post({ type: "ready", device: "webgpu" });
}

// Proportionally clamp the longest side (keeps aspect) — for a single image.
async function clampImage(image, maxSide) {
  const longest = Math.max(image.width, image.height);
  if (longest <= maxSide) return image;
  const scale = maxSide / longest;
  return await image.resize(Math.round(image.width * scale), Math.round(image.height * scale));
}

async function run(id, imageURLs, prompt, maxTokens) {
  await ensureLoaded();
  const { TextStreamer, RawImage } = mod;

  const urls = Array.isArray(imageURLs) ? imageURLs : [imageURLs];
  const images = [];
  for (const url of urls) {
    let image = await RawImage.read(url);
    // Multi-image: force a common square so each image's num_img_tokens matches (Phi3VProcessor reuses
    // the first image's token count for every placeholder in a turn). Single image: keep aspect.
    image = urls.length > 1
      ? await image.resize(MULTI_SIDE, MULTI_SIDE)
      : await clampImage(image, MAX_SIDE);
    images.push(image);
  }

  // Build the <|image_1|>…<|image_N|> placeholder block + the question, then apply the chat template.
  const placeholders = images.map((_, i) => `<|image_${i + 1}|>`).join("\n");
  const content = images.length ? `${placeholders}\n${prompt}` : prompt;
  const messages = [{ role: "user", content }];
  const text = processor.tokenizer.apply_chat_template(messages, {
    tokenize: false,
    add_generation_prompt: true,
  });
  post({ type: "prompt", id, template: text }); // "See inside": the constructed chat template.

  const inputs = await processor(text, images);

  // The processor negates image-token ids; count them for the honest image-token readout.
  let imageTokens = 0;
  const idData = inputs.input_ids?.data;
  if (idData) {
    for (const v of idData) {
      const n = typeof v === "bigint" ? v : BigInt(Math.trunc(Number(v)));
      if (n < 0n) imageTokens++;
    }
  }
  const promptLen = inputs.input_ids?.dims?.at(-1) ?? null;
  post({ type: "meta", id, imageTokens, promptLen, images: images.length });

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
    max_new_tokens: maxTokens ?? 256,
    do_sample: false,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  post({ type: "done", id, ms, tokens: count, promptLen, imageTokens });
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
    console.error("[phi-3.5-vision worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
