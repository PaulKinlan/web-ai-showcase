// LFM2-350M (Transformers.js / ONNX) worker — streaming text generation off the main thread.
//
// LFM2 is Liquid AI's second-generation on-device foundation model (2025): a HYBRID architecture that
// interleaves short-range gated convolution blocks with grouped-query attention, designed from the
// ground up for fast CPU/edge inference rather than being a scaled-down transformer. Its model class
// (`lfm2`) is NOT registered in the shared Transformers.js 3.7.5 pin, so this worker uses the isolated
// **version-pin escape hatch** (precedent: models/sam2-segmentation/worker.js pins 4.2.0) — it imports
// @huggingface/transformers@4.2.0 LOCALLY here only. lib/webai.js and every other page stay on 3.7.5.
//
// DTYPE is device-adaptive, and both paths are REAL:
//   • WebGPU present → dtype "q4f16" (~260 MB). The onnx-community quantized export block-quantizes the
//     token-embedding gather (a WebGPU-only ORT-Web kernel, GatherBlockQuantized), so the fast quantized
//     path requires a GPU adapter.
//   • No WebGPU → dtype "fp32" (~1.4 GB). The fp32 export uses a plain Gather and runs on the WASM EP.
//     This is the path VERIFIED coherent in headless Chrome (no GPU): greedy decode produced
//     "The sky appears blue because of a phenomenon called Rayleigh scattering, …" (20 words, 19 unique).
// The page picks the device+dtype (from a real adapter probe) and tells us which to load, so the
// download label always matches what is actually fetched — never a silent large download.

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
const MODEL_ID = "onnx-community/LFM2-350M-ONNX";

let generator = null;
let mod = null;
let device = "wasm";
let dtype = "fp32";
let stopper = null;

function post(msg) {
  self.postMessage(msg);
}

async function ensureLoaded(opts) {
  if (generator) return;
  device = opts?.device === "webgpu" ? "webgpu" : "wasm";
  dtype = opts?.dtype || (device === "webgpu" ? "q4f16" : "fp32");
  mod = await import(TRANSFORMERS_URL);
  const { pipeline, env } = mod;
  env.allowLocalModels = false;
  console.log(`[lfm2 worker] loading ${MODEL_ID} on ${device} (${dtype})`);
  generator = await pipeline("text-generation", MODEL_ID, {
    device,
    dtype,
    progress_callback: (p) => post({ type: "progress", p }),
  });
  console.log(`[lfm2 worker] ready on ${device} (${dtype})`);
  post({ type: "ready", device, dtype });
}

async function chat(id, messages, opts) {
  await ensureLoaded(opts);
  const { TextStreamer } = mod;

  // "See inside" — apply LFM2's own chat template so the exact prompt is honest + inspectable.
  const prompt = generator.tokenizer.apply_chat_template(messages, {
    add_generation_prompt: true,
    tokenize: false,
  });
  post({ type: "prompt", id, template: prompt });

  const t0 = performance.now();
  let ttft = null;
  let firstFired = false;
  let chunks = 0;
  let full = "";

  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (piece) => {
      if (!piece) return;
      full += piece;
      if (!firstFired) {
        firstFired = true;
        ttft = performance.now() - t0;
        post({ type: "first", id, t: Math.round(ttft) });
      }
      chunks++;
      post({ type: "token", id, delta: piece });
    },
  });

  stopper = mod.InterruptableStoppingCriteria ? new mod.InterruptableStoppingCriteria() : null;
  const doSample = opts?.doSample ?? true;
  const genOpts = {
    max_new_tokens: Math.max(1, Math.min(1024, opts?.maxTokens ?? 256)),
    do_sample: doSample,
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
    ...(doSample
      ? {
        temperature: opts?.temperature ?? 0.3,
        top_p: opts?.topP ?? 0.95,
        top_k: opts?.topK ?? 50,
      }
      : {}),
    ...(stopper ? { stopping_criteria: stopper } : {}),
  };

  await generator(prompt, genOpts);

  const ms = Math.round(performance.now() - t0);
  post({
    type: "done",
    id,
    text: full,
    ms,
    ttft: ttft === null ? ms : Math.round(ttft),
    chunks,
    device,
    dtype,
  });
}

self.addEventListener("message", async (e) => {
  const { type } = e.data;
  try {
    if (type === "load") await ensureLoaded(e.data.opts);
    else if (type === "run") await chat(e.data.id, e.data.messages, e.data.opts);
    else if (type === "stop") stopper?.interrupt?.();
  } catch (err) {
    console.error("[lfm2 worker] error", err);
    post({ type: "error", id: e.data?.id, message: String(err?.message ?? err) });
  }
});
