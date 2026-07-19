// Qwen2.5-VL worker — vision-language generation off the main thread, with honest WebGPU gating.
// Model: onnx-community/Qwen2.5-VL-3B-Instruct-ONNX (image-text-to-text), WebGPU, decoder q4f16.
//
// DISTINCT from the built Qwen2-VL demo: Qwen2.5-VL keeps the image at its NATIVE aspect ratio
// (dynamic resolution — no forced square crop), so the number of vision tokens scales with the image;
// it has stronger document/OCR parsing and native visual GROUNDING (it can emit bounding-box
// coordinates in the resized-image pixel space). We surface both: the live image-token count and,
// for the grounding page, the processed dims so boxes can be scaled back onto the display.
//
// VERSION-PIN escape hatch (CLAUDE.md invariant 9): the Qwen2_5_VLForConditionalGeneration class /
// qwen2_5_vl model type exists ONLY in @huggingface/transformers >= 4.2.0 (absent from the shared
// 3.7.5 pin in lib/webai.js — grep-verified). We import 4.2.0 LOCALLY here and NOWHERE else; the
// shared lib/model-cache.js is version-agnostic (scans Cache Storage by modelId) so createModelLoader
// auto-init still works. Precedent: models/sam2-segmentation/worker.js pins 4.2.0.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "onnx-community/Qwen2.5-VL-3B-Instruct-ONNX";
const IMAGE_TOKEN_ID = 151655; // <|image_pad|> — from the model config.json
const FACTOR = 28; // patch(14) × spatial-merge(2): valid image sides are multiples of 28
const MAX_SIDE = 980; // bound vision tokens + GPU memory while keeping the native aspect ratio
const MIN_SIDE = 112;

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
  const { AutoProcessor, Qwen2_5_VLForConditionalGeneration } = mod;
  console.log(
    `[qwen2.5-vl worker] loading ${MODEL_ID} on webgpu (decoder q4f16) via transformers@4.2.0`,
  );
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: (p) => post({ type: "progress", p }),
  });
  model = await Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: {
      embed_tokens: "fp16",
      vision_encoder: "fp16",
      decoder_model_merged: "q4f16",
    },
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log("[qwen2.5-vl worker] ready on webgpu");
  post({ type: "ready", device: "webgpu" });
}

// Fit the image to a multiple-of-28 size that keeps its NATIVE aspect ratio (Qwen2.5-VL dynamic
// resolution) while bounding the longest side. Because the size is already a valid multiple of 28 and
// within the processor's pixel budget, the processor's smart-resize is a no-op — so the coordinates
// the model emits for grounding are in exactly these dims, and we can scale boxes back to the display.
function fitDims(w, h) {
  const scale = Math.min(1, MAX_SIDE / Math.max(w, h));
  const nw = Math.max(MIN_SIDE, Math.round((w * scale) / FACTOR) * FACTOR);
  const nh = Math.max(MIN_SIDE, Math.round((h * scale) / FACTOR) * FACTOR);
  return { nw, nh };
}

// Build a Qwen2.5-VL conversation. `history` is prior turns [{role,text}] (image lives on the first
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

function countImageTokens(inputIds) {
  try {
    const data = inputIds?.data;
    if (!data) return null;
    let n = 0;
    for (let i = 0; i < data.length; i++) {
      const v = typeof data[i] === "bigint" ? Number(data[i]) : data[i];
      if (v === IMAGE_TOKEN_ID) n++;
    }
    return n;
  } catch {
    return null;
  }
}

async function run(id, imageURL, prompt, maxTokens, history) {
  await ensureLoaded();
  const { TextStreamer, RawImage } = mod;
  let image = await RawImage.read(imageURL);
  const { nw, nh } = fitDims(image.width, image.height);
  image = await image.resize(nw, nh); // native aspect ratio preserved (dynamic resolution)

  const conversation = buildConversation(history, prompt);
  const text = processor.apply_chat_template(conversation, { add_generation_prompt: true });
  post({ type: "prompt", id, template: text }); // "See inside": the constructed chat template.

  const inputs = await processor(text, image);
  const imageTokens = countImageTokens(inputs.input_ids);
  const gridThw = inputs.image_grid_thw?.data
    ? Array.from(inputs.image_grid_thw.data, (x) => (typeof x === "bigint" ? Number(x) : x))
    : null;
  post({ type: "meta", id, imageTokens, gridThw, procW: nw, procH: nh });

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
  const promptLen = inputs.input_ids?.dims?.at(-1) ?? null;
  post({ type: "done", id, ms, tokens: count, promptLen, imageTokens, procW: nw, procH: nh });
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
    console.error("[qwen2.5-vl worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
