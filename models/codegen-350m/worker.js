// CodeGen-350M-mono worker — code generation off the main thread, WASM q8.
// Model: Xenova/codegen-350M-mono (task: text-generation).
//
// Salesforce's CodeGen is one of the first open autoregressive code language models: a 350M decoder
// trained on The Pile + BIGQUERY (public GitHub code), mono = single-language (Python). Given a
// function signature + docstring (or a partial body), it completes the code left-to-right, one token
// at a time. We use the text-generation pipeline with a TextStreamer so every emitted token streams
// to the page as it is produced — real on-device inference, nothing canned.

import { TRANSFORMERS_URL } from "/web-ai-showcase/lib/webai.js";

const MODEL_ID = "Xenova/codegen-350M-mono";
const DTYPE = "q8";
let generator = null;
let mod = null;
let device = "wasm";

function post(msg) {
  self.postMessage(msg);
}

async function makePipeline() {
  const { pipeline } = mod;
  return pipeline("text-generation", MODEL_ID, {
    device,
    dtype: DTYPE,
    progress_callback: (p) => post({ type: "progress", p }),
  });
}

async function ensureLoaded() {
  if (generator) return;
  mod = await import(TRANSFORMERS_URL);
  generator = await makePipeline();
  post({ type: "ready", device });
}

async function generate(id, prompt, opts) {
  await ensureLoaded();
  const { TextStreamer } = mod;
  const maxNew = Math.min(160, Math.max(1, opts.maxNew ?? 80));
  const greedy = !!opts.greedy;
  post({ type: "prompt", id, prompt });

  let count = 0;
  const t0 = performance.now();
  const streamer = new TextStreamer(generator.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token) => {
      count++;
      post({ type: "token", id, piece: token, t: performance.now() - t0 });
    },
  });
  const out = await generator(prompt, {
    max_new_tokens: maxNew,
    do_sample: !greedy,
    temperature: greedy ? 1 : (opts.temperature ?? 0.7),
    top_k: greedy ? undefined : (opts.topK ?? 40),
    repetition_penalty: 1.1,
    streamer,
    return_full_text: false,
  });
  const ms = Math.round(performance.now() - t0);
  post({
    type: "done",
    id,
    text: prompt + String(out?.[0]?.generated_text ?? ""),
    generated: String(out?.[0]?.generated_text ?? ""),
    ms,
    tokens: count,
    device,
  });
}

self.addEventListener("message", async (e) => {
  const { type, id } = e.data;
  try {
    if (type === "load") await ensureLoaded();
    else if (type === "generate") await generate(id, e.data.prompt, e.data.opts || {});
  } catch (err) {
    post({ type: "error", id, message: String(err?.message ?? err) });
  }
});
