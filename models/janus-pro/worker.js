// Janus-Pro worker — the UNIFIED multimodal model runs off the main thread, WebGPU-gated honestly.
// Model: onnx-community/Janus-Pro-1B-ONNX (any-to-any). ONE model does BOTH:
//   • understanding  — image + text  -> streamed text  (model.generate + TextStreamer)
//   • generation     — text          -> a decoded raster image (model.generate_images)
//
// Verified against transformers.js v3.7.5 + the model card + the official janus-pro-webgpu example:
//   AutoProcessor.from_pretrained            -> resolves VLChatProcessor (processor_class in the repo)
//   MultiModalityCausalLM.from_pretrained    -> config.model_type "multi_modality" is registered
//   model.generate(...)                      -> text (understanding)
//   processor(conv, { chat_template:"text_to_image" }) + model.generate_images(...) -> RawImage[]
// The two paths use decoupled vision encoders: SigLIP for understanding, a VQ tokenizer + image_decode
// head for generation — the whole reason a single 1B model can both read AND draw.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Janus-Pro-1B-ONNX";
let mod = null;
let processor = null;
let model = null;
let device = "webgpu";

const post = (msg) => self.postMessage(msg);

// Real capability probe — navigator.gpu existing is not enough; the adapter must resolve.
async function probeGPU() {
  if (!("gpu" in navigator)) return { ok: false, reason: "no-gpu" };
  let adapter = null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch (e) {
    return { ok: false, reason: "adapter-error", detail: String(e?.message ?? e) };
  }
  if (!adapter) return { ok: false, reason: "no-adapter" };
  return { ok: true, shaderF16: adapter.features?.has?.("shader-f16") ?? false };
}

async function ensureLoaded() {
  if (model) return;
  mod = await import(TRANSFORMERS_URL);
  const { AutoProcessor, MultiModalityCausalLM } = mod;

  // shader-f16 lets us use the lighter fp16/q4f16 weights; without it we fall back to fp32/q4.
  const gpu = await probeGPU();
  const f16 = gpu.ok && gpu.shaderF16;
  console.log(`[janus worker] loading ${MODEL_ID} (shader-f16=${f16})`);

  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });

  model = await MultiModalityCausalLM.from_pretrained(MODEL_ID, {
    // Mirrors the official janus-pro-webgpu example's dtype selection.
    dtype: f16
      ? {
        prepare_inputs_embeds: "q4",
        language_model: "q4f16",
        lm_head: "fp16",
        gen_head: "fp16",
        gen_img_embeds: "fp16",
        image_decode: "fp32",
      }
      : {
        prepare_inputs_embeds: "fp32",
        language_model: "q4",
        lm_head: "fp32",
        gen_head: "fp32",
        gen_img_embeds: "fp32",
        image_decode: "fp32",
      },
    device: {
      // prepare_inputs_embeds stays on wasm (a known WebGPU op bug upstream); the rest is WebGPU.
      prepare_inputs_embeds: "wasm",
      language_model: "webgpu",
      lm_head: "webgpu",
      gen_head: "webgpu",
      gen_img_embeds: "webgpu",
      image_decode: "webgpu",
    },
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[janus worker] ready");
  post({ type: "ready", device, shaderF16: f16 });
}

// ---- Understanding: image (+ text) -> streamed text ---------------------------------------------
async function understand(id, imageURL, prompt, maxTokens) {
  await ensureLoaded();
  const { TextStreamer } = mod;

  const conversation = imageURL
    ? [{ role: "<|User|>", content: "<image_placeholder>\n" + prompt, images: [imageURL] }]
    : [
      {
        role: "<|System|>",
        content: "You are a helpful assistant. Answer the user's questions concisely.",
      },
      { role: "<|User|>", content: prompt },
    ];

  const inputs = await processor(conversation);
  post({ type: "prompt", id, template: describeConversation(conversation) });

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

  post({ type: "done", id, ms: Math.round(performance.now() - t0), tokens: count });
}

// ---- Generation: text -> a decoded image --------------------------------------------------------
async function generate(id, prompt) {
  await ensureLoaded();
  const { BaseStreamer } = mod;

  const conversation = [{ role: "<|User|>", content: prompt }];
  const inputs = await processor(conversation, { chat_template: "text_to_image" });
  const numImageTokens = processor.num_image_tokens; // 576 = 24x24 image-token grid
  post({ type: "gen-start", id, total: numImageTokens });

  // Report each generated image token as the autoregressive stream fills the 24x24 grid.
  let started = null;
  let seen = 0;
  const streamer = new (class extends BaseStreamer {
    put() {
      if (seen === 0) {
        // first call is the prompt batch — start the clock and skip it
        started = performance.now();
        seen = 1;
        return;
      }
      const step = seen++; // 1..numImageTokens
      const t = performance.now() - started;
      post({ type: "gen-progress", id, step, total: numImageTokens, t, tps: (step / t) * 1000 });
    }
    end() {}
  })();

  const t0 = performance.now();
  const outputs = await model.generate_images({
    ...inputs,
    min_new_tokens: numImageTokens,
    max_new_tokens: numImageTokens,
    do_sample: true,
    streamer,
  });

  const blob = await outputs[0].toBlob();
  post({
    type: "gen-done",
    id,
    blob,
    width: outputs[0].width,
    height: outputs[0].height,
    ms: Math.round(performance.now() - t0),
    tokens: numImageTokens,
  });
}

// A readable rendering of the templated turn for the "see inside" surface.
function describeConversation(conv) {
  return conv
    .map((t) =>
      `${t.role}\n${t.content}${t.images ? "\n[+ image → 576 SigLIP vision tokens]" : ""}`
    )
    .join("\n\n");
}

self.addEventListener("message", async (e) => {
  const d = e.data;
  try {
    if (d.type === "probe") {
      post({ type: "probe-result", gpu: await probeGPU() });
    } else if (d.type === "load") {
      await ensureLoaded();
    } else if (d.type === "understand") {
      await understand(d.id, d.image, d.prompt, d.maxTokens);
    } else if (d.type === "generate") {
      await generate(d.id, d.prompt);
    }
  } catch (err) {
    console.error("[janus worker] error", err);
    post({ type: "error", id: d?.id, message: String(err?.message ?? err) });
  }
});
