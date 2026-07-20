// SmolDocling worker — document-image → structured DocTags, ALL inference off the main thread, with
// honest WebGPU gating.
//
// DISTINCT from the built document demos. Nougat (models/nougat-ocr) turns a page into Markdown; Donut
// (models/donut-docvqa) answers a question about a form; TrOCR does line OCR. SmolDocling does something
// different: it converts a whole document image into **DocTags** — a compact structured markup that
// captures LAYOUT + TEXT + TABLES + FORMULAS together, each element tagged with its type and its
// bounding box (<text>, <section_header_level_1>, <otsl> tables with <fcel>/<ched> cells, <formula> as
// LaTeX, <picture>, <caption>, …). It's the model behind IBM's Docling document-conversion toolkit.
//
// Model:   docling-project/SmolDocling-256M-preview  (task: image-text-to-text)
// Family:  Idefics3ForConditionalGeneration (model_type "idefics3") — a SmolVLM-family VLM: SigLIP
//          vision encoder + SmolLM2 decoder, fine-tuned on synthetic document/table/formula/chart data.
// Classes: AutoProcessor + AutoModelForVision2Seq. idefics3 loads in 3.7.5 for SmolVLM v1, BUT
//          SmolDocling's ONNX only loads under Transformers.js v4+ (in 3.7.5 the session build aborts
//          with a bare numeric error — verified). So this worker PINS v4.2.0 locally, exactly like the
//          SAM2 worker pins v4 for Sam2Model. The shared lib/webai.js stays 3.7.5; model-cache is
//          version-agnostic so the auto-init loader still works.
// dtype:   q4 — VERIFIED to produce coherent DocTags on WebGPU. The q4f16 / fp16 / int8 exports are all
//          DEGENERATE for this model (they emit repeated-token garbage — measured), so q4 is the honest
//          runnable export.
// Backend: WebGPU (required) — a 256M-param VLM over a split document image needs a GPU to be usable.

// SmolDocling's ONNX needs Transformers.js v4+; the shared webai.js is pinned to 3.7.5. Pin v4 LOCALLY
// here only — a per-model choice, not a shared-file change. Precedent: models/sam2-segmentation/worker.js
const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";

const MODEL_ID = "docling-project/SmolDocling-256M-preview";
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
    dtype: "q4", // q4 is the honest runnable export; q4f16/fp16/int8 are degenerate for this model
    device: "webgpu",
    progress_callback: (p) => post({ type: "progress", p }),
  });
  post({ type: "ready", device: "webgpu" });
}

// Strip the model's chat/end special tokens from the DocTags stream (keep the DocTags themselves).
function cleanTag(tok) {
  return tok
    .replace(/<end_of_utterance>/g, "")
    .replace(/<\|im_end\|>/g, "")
    .replace(/<eos>/g, "");
}

async function run(id, imageURL, promptText, maxTokens) {
  await ensureLoaded();
  const { TextStreamer, load_image } = mod;
  const image = await load_image(imageURL);

  const prompt = promptText || "Convert this page to docling.";
  const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: prompt }] }];
  const text = processor.apply_chat_template(messages, { add_generation_prompt: true });
  post({ type: "prompt", id, template: text, prompt });

  const inputs = await processor(text, [image], {});
  // input_ids length = full templated token count (text + expanded image tokens). pixel_values dims =
  // [batch, numTiles, channels, H, W] — surface both so "See inside" shows the real document tiling.
  const promptTokens = inputs.input_ids?.dims?.at(-1) ?? null;
  const pvDims = inputs.pixel_values?.dims ? Array.from(inputs.pixel_values.dims) : null;
  post({ type: "shape", id, promptTokens, pixelValues: pvDims });

  const t0 = performance.now();
  let count = 0;
  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: false, // keep the DocTags (<text>, <loc_>, <otsl>, <formula>, …)
    callback_function: (tok) => {
      const clean = cleanTag(tok);
      count++;
      if (clean) post({ type: "token", id, token: clean, t: performance.now() - t0 });
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: maxTokens ?? 512,
    do_sample: false,
    streamer,
  });

  const ms = Math.round(performance.now() - t0);
  post({ type: "done", id, ms, tokens: count, promptTokens });
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
