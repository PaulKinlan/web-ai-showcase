// Florence-2-large worker — task-prompted vision inference off the main thread, honest WebGPU gating.
// Model: onnx-community/Florence-2-large (image-text-to-text), WebGPU, dtype fp16.
//
// This is the LARGER Florence-2 checkpoint (0.77B params vs 0.23B for the base page). Same unified
// task-prompt API, the accuracy/detail step up. Canonical Transformers.js Florence-2 path (v3.7.5):
//   Florence2ForConditionalGeneration + AutoProcessor + load_image
//   -> processor.construct_prompts(task) -> processor(image, prompts) -> model.generate
//   -> processor.batch_decode(..., { skip_special_tokens:false }) -> processor.post_process_generation.
// One model, many tasks: <CAPTION>, <DETAILED_CAPTION>, <OD>, <OCR>, <REFERRING_EXPRESSION_SEGMENTATION>…

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "onnx-community/Florence-2-large";
let model = null;
let processor = null;
let mod = null;
let activeDevice = "webgpu";
let activeDtype = "fp16";

function post(msg) {
  self.postMessage(msg);
}

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
  const { Florence2ForConditionalGeneration, AutoProcessor } = mod;
  
  processor = await AutoProcessor.from_pretrained("onnx-community/Florence-2-large", {
    progress_callback: (p) => post({ type: "progress", p }),
  });

  const gpu = await probeGPU();
  if (gpu.ok) {
    // Attempt primary fp16 path on WebGPU
    try {
      console.log(`[florence worker] loading ${MODEL_ID} on webgpu (fp16)`);
      model = await Florence2ForConditionalGeneration.from_pretrained("onnx-community/Florence-2-large", {
        dtype: "fp16",
        device: "webgpu",
        progress_callback: (p) => post({ type: "progress", p }),
      });
      activeDevice = "webgpu";
      activeDtype = "fp16";
    } catch (errFp16) {
      console.warn(`[florence worker] webgpu fp16 failed (${errFp16?.message || errFp16}), falling back to webgpu q4`);
      post({ type: "progress", p: { status: "initiate", file: "retrying on WebGPU (q4)…" } });
      try {
        model = await Florence2ForConditionalGeneration.from_pretrained("onnx-community/Florence-2-large", {
          dtype: "q4",
          device: "webgpu",
          progress_callback: (p) => post({ type: "progress", p }),
        });
        activeDevice = "webgpu";
        activeDtype = "q4";
      } catch (errQ4) {
        console.warn(`[florence worker] webgpu q4 failed (${errQ4?.message || errQ4}), falling back to wasm q8`);
        post({ type: "progress", p: { status: "initiate", file: "retrying on WASM (q8)…" } });
        model = await Florence2ForConditionalGeneration.from_pretrained("onnx-community/Florence-2-large", {
          dtype: "q8",
          device: "wasm",
          progress_callback: (p) => post({ type: "progress", p }),
        });
        activeDevice = "wasm";
        activeDtype = "q8";
      }
    }
  } else {
    // No GPU adapter — honest WASM q8 fallback
    console.log(`[florence worker] no WebGPU adapter, loading ${MODEL_ID} on wasm (q8)`);
    model = await Florence2ForConditionalGeneration.from_pretrained("onnx-community/Florence-2-large", {
      dtype: "q8",
      device: "wasm",
      progress_callback: (p) => post({ type: "progress", p }),
    });
    activeDevice = "wasm";
    activeDtype = "q8";
  }

  console.log(`[florence worker] ready on ${activeDevice} (${activeDtype})`);
  post({ type: "ready", device: activeDevice, dtype: activeDtype });
}

async function run(id, imageURL, task, text, maxTokens) {
  await ensureLoaded();
  const { load_image } = mod;
  const image = await load_image(imageURL);

  // Tasks that take an extra text input (e.g. the phrase to segment) append it to the task token.
  const promptText = text && text.trim() ? task + text.trim() : task;
  const prompts = processor.construct_prompts(promptText);
  post({ type: "progress", p: { status: "generating" } });

  const inputs = await processor(image, prompts);
  const t0 = performance.now();
  const generated_ids = await model.generate({
    ...inputs,
    max_new_tokens: Math.max(64, Math.min(1024, maxTokens ?? 512)),
    do_sample: false,
    num_beams: 1,
  });
  const ms = Math.round(performance.now() - t0);
  const tokens = generated_ids?.dims ? generated_ids.dims[generated_ids.dims.length - 1] : null;

  // The RAW decoded string (special tokens kept) — this is what the "See inside" surface shows,
  // BEFORE Florence's task-specific post-processing turns it into boxes / polygons / clean text.
  const raw = processor.batch_decode(generated_ids, { skip_special_tokens: false })[0];
  const parsed = processor.post_process_generation(raw, task, image.size);

  post({
    type: "result",
    id,
    task,
    prompt: promptText,
    raw,
    parsed,
    imageSize: image.size, // [width, height]
    tokens,
    ms,
    device: activeDevice,
    dtype: activeDtype,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "probe") {
      post({ type: "probe-result", gpu: await probeGPU() });
    } else if (type === "load") {
      await ensureLoaded();
    } else if (type === "run") {
      await run(e.data.id, e.data.image, e.data.task, e.data.text, e.data.maxTokens);
    }
  } catch (err) {
    console.error("[florence-large worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
